"""Export and verify every frozen full-structure prediction; no refitting.

Creates all-case galleries, compact result tables, a provenance receipt, and
the small training-only calibration dependency needed by portable reruns.
The numeric predictions and original experiment receipts are never rewritten.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from threadpoolctl import threadpool_limits

from microc.data import area_resample
from scripts.task5_full_structure_train_20261014 import PREFIX
from scripts.task5_full_structure_verification_20261014 import ROOT, METRICS, digest, image_metrics, morphology, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"outputs/task5_full_structure_20261014/neural_adaptation")
    parser.add_argument("--calibration",type=Path,default=ROOT/"outputs/task5_full_structure_20261014/scale_repair/trained_scale_parameters.json")
    parser.add_argument("--results-dir",type=Path,default=ROOT/"results")
    args = parser.parse_args()
    if not (args.run/"summary.json").exists():
        raise FileNotFoundError("Frozen test summary is required")
    destination = args.run/"complete_evidence"
    if (destination/"export_receipt.json").exists():
        raise FileExistsError("Export refuses to overwrite an existing receipt")
    destination.mkdir(parents=True,exist_ok=True)
    threadpool_limits(limits=2)
    summary = json.loads((args.run/"summary.json").read_text(encoding="utf-8"))
    freeze = json.loads((args.run/"selection_freeze.json").read_text(encoding="utf-8"))
    if digest(args.run/"selected_model.pt") != freeze["model_sha256"]:
        raise ValueError("Frozen model drift")
    if digest(args.calibration) != freeze["source_and_input_sha256"]["calibration"]:
        raise ValueError("Wrong portable calibration dependency")
    selected = summary["selected_method"]
    cohort = pd.read_csv(args.run/"cohort_manifest.csv")
    test = cohort[cohort.split=="test"].sort_values("center")
    records,aggregate = [],[]
    max_error = 0.
    for replicate in summary["replicates"]:
        shown_morphology = set()
        scores = pd.read_csv(args.run/replicate/"per_structure.csv")
        chosen = scores[scores.method==selected].set_index("id")
        if set(chosen.index) != set(test.id) or chosen.index.duplicated().any():
            raise ValueError("Prediction cohort differs from fixed test cohort")
        for method,group in scores.groupby("method"):
            for kind,subset in (("all",group),*list(group.groupby("type"))):
                aggregate.append({"replicate":replicate,"method":method,"type":kind,"n":len(subset),
                                  **{f"mean_{metric}":float(subset[metric].mean()) for metric in METRICS},
                                  "median_shape_r":float(subset.shape_r.median()),"median_profile_r":float(subset.profile_r.median())})
        for number in range(0,len(test),12):
            block = test.iloc[number:number+12]
            fig,axes = plt.subplots(len(block),4,figsize=(12,2.05*len(block)),dpi=120,squeeze=False)
            for row,(record,axrow) in enumerate(zip(block.itertuples(),axes)):
                path = args.run/replicate/f"{record.id}_full_maps.npz"
                with np.load(path) as source:
                    matrices = {key:source[key] for key in ("bilinear","calibrated_bilinear","prediction","truth")}
                    start,end = int(source["start_bp"]),int(source["end_bp"])
                shape = matrices["truth"].shape
                if shape != (record.native_bins,record.native_bins) or (start,end)!=(record.window_start_bp,record.window_end_bp):
                    raise ValueError("Full map extent does not match full annotation window")
                for image in matrices.values():
                    if not np.isfinite(image).all() or not np.allclose(image,image.T,atol=2e-6,rtol=2e-6):
                        raise ValueError("Full map nonfinite or asymmetric")
                recomputed = image_metrics(matrices["truth"],matrices["prediction"])
                errors = {metric:abs(float(chosen.loc[record.id,metric])-recomputed[metric]) for metric in METRICS}
                max_error = max(max_error,*errors.values())
                if max(errors.values()) > 1e-9:
                    raise ValueError(f"Stored map/source-table mismatch for {record.id}: {errors}")
                records.append({"replicate":replicate,"id":record.id,"type":record.type,"full_native_bins":record.native_bins,
                                "start_bp":start,"end_bp":end,"maps_file":str(path.relative_to(args.run)),
                                "maps_sha256":digest(path),
                                "native_png_file":str((args.run/replicate/f"{record.id}_full.png").relative_to(args.run)),
                                "native_png_sha256":digest(args.run/replicate/f"{record.id}_full.png"),
                                "all_metric_recomputed_absolute_error_max":max(errors.values()),
                                "shape_r":recomputed["shape_r"],"profile_r":recomputed["profile_r"]})
                if record.type not in shown_morphology:
                    figure = plt.figure(figsize=(12,5),dpi=150)
                    grid = figure.add_gridspec(2,4,height_ratios=(3,1.7))
                    labels = (("bilinear","Bilinear","#8b8b8b"),
                              ("calibrated_bilinear","Train-calibrated bilinear","#b08256"),
                              ("prediction","Final CNN","#087f8c"),
                              ("truth","Micro-C target","#172c38"))
                    profile_axis = figure.add_subplot(grid[1,:])
                    for column,(key,title,color) in enumerate(labels):
                        ax = figure.add_subplot(grid[0,column])
                        smooth = gaussian_filter(matrices[key],sigma=4)
                        coarse = area_resample(smooth,64)/(len(smooth)/64)**2
                        ax.imshow(coarse,origin="lower",cmap="magma",vmin=-2,vmax=4,
                                  extent=[start,end]*2)
                        ax.set_title(title,fontsize=10)
                        ax.tick_params(labelsize=7)
                        ax.set(xlabel="position bp",ylabel="position bp")
                        _,profile,_ = morphology(matrices[key])
                        x = start+np.arange(len(profile))*10
                        profile_axis.plot(x,profile,color=color,lw=1.2,label=title)
                    profile_axis.set(xlabel="position bp",ylabel="positive contact row profile")
                    profile_axis.legend(loc="upper right",fontsize=8,ncol=4)
                    figure.suptitle(f"{replicate} {record.id} complete span; identical40bp smoothing +64-bin area view",fontsize=11)
                    figure.tight_layout()
                    figure.savefig(destination/f"{replicate}_{record.type}_morphology.png")
                    plt.close(figure)
                    shown_morphology.add(record.type)
                for ax,(key,title) in zip(axrow,(("bilinear","Bilinear"),("calibrated_bilinear","Train-calibrated bilinear"),("prediction","Final adapted CNN"),("truth","Micro-C target"))):
                    ax.imshow(matrices[key],origin="lower",cmap="magma",vmin=-2,vmax=4)
                    ax.set_xticks([]);ax.set_yticks([])
                    ax.set_title(title if row==0 else "",fontsize=9)
                axrow[0].set_ylabel(f"{record.id}\n{record.length_bp}bp\nshape r={recomputed['shape_r']:.3f}",fontsize=8)
            fig.suptitle(f"{replicate} all structures {number+1}-{number+len(block)} of {len(test)}",fontsize=12)
            fig.tight_layout(rect=(0,0,1,.985))
            fig.savefig(destination/f"{replicate}_all_cases_{number//12+1:02d}.png")
            plt.close(fig)
            print(f"Verified {replicate} cases {number+1}-{number+len(block)}",flush=True)
    pd.DataFrame(aggregate).to_csv(destination/"method_type_summary.csv",index=False)
    write_json(destination/"all_case_manifest.json",{"fixed_cohort":len(test),"replicates":2,"records":records,
               "maximum_numeric_recomputation_error":max_error,"not_best_example_selection":True,
               "gallery_rule":"All68 complete test structures in genomic order for each repeat,12per page; identical color scale",
               "morphology_representatives":"First genomic-position structure of each type,not selected by result",
               "morphology_display_rule":"Identical40bp Gaussian smoothing and64-bin count-area mean for every map;[-2,4]color scale;unrescaled positive-row profile;native10bp maps and galleries retained"})
    cal_target = args.results_dir/(PREFIX+"trained_scale_parameters.json")
    if cal_target.exists():
        raise FileExistsError("Portable dependency already exists; refusing overwrite")
    shutil.copyfile(args.calibration,cal_target)
    for name in ("method_type_summary.csv","all_case_manifest.json"):
        target = args.results_dir/(PREFIX+name)
        if target.exists():
            raise FileExistsError("Refusing existing export result overwrite")
        shutil.copyfile(destination/name,target)
    for image in destination.glob("*_morphology.png"):
        target = args.results_dir/(PREFIX+image.name)
        if target.exists():
            raise FileExistsError("Refusing existing morphology figure overwrite")
        shutil.copyfile(image,target)
    write_json(destination/"export_receipt.json",{"exporter_sha256":digest(Path(__file__)),
               "model_sha256":freeze["model_sha256"],"summary_sha256":digest(args.run/"summary.json"),
               "source_tables_sha256":{name:digest(args.run/name/"per_structure.csv") for name in summary["replicates"]},
               "verified_full_structures":len(records),"maximum_numeric_recomputation_error":max_error,
               "portable_calibration_sha256":digest(cal_target),
               "output_sha256":{p.name:digest(p) for p in destination.iterdir() if p.is_file()},
               "no_models_refit":True,"no_frozen_receipts_changed":True})
    shutil.copyfile(destination/"export_receipt.json",args.results_dir/(PREFIX+"export_receipt.json"))
    print(json.dumps({"verified_complete_structures":len(records),"max_error":max_error,"output":str(destination)},indent=2),flush=True)


if __name__ == "__main__":
    main()

"""Full-window distribution adaptation of the CNN, frozen before test.

TRAIN pairs use globally normalized complete intervals and random off-diagonal
tiles. A halo core loss combines MSE, gradients, and local SSIM. Validation
chooses among predeclared epoch/fraction candidates; test cannot alter weights.
Train-calibrated bilinear and bicubic controls are retained. Publishing refuses
to overwrite any existing final-run result, including independent reruns.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from microc.data import annotations, open_cool
from microc.deep import SuperResolutionCNN, _gradient_loss
from scripts.evaluate_sr_ensemble import load_model
from scripts.task5_full_structure_repair_20261014 import calibrate
from scripts.task5_full_structure_verification_20261014 import (
    ROOT, SEED, PRIMARY_TYPES, cohort_manifest, criterion, digest, image_metrics,
    morphology, prepare_full_pair, summarize, utc, whole_cnn, write_json,
)


EPOCHS = (2, 4, 6)
PATCHES_PER_EPOCH = 800
FRACTIONS = (.25, .5, .75, 1.)
PREFIX = "task5_full_structure_20261014_final_"


class FullWindowCNN(nn.Module):
    def __init__(self, base: SuperResolutionCNN, scale: float = 1., bias: float = 0.):
        super().__init__()
        self.base = base
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(scale, 1e-4)), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        # Off-diagonal tiles are not locally symmetric. Projection is global.
        raw = x + self.base.tail(self.base.body(self.base.head(x)))
        return (self.log_scale.exp() * raw + self.bias).clamp(-5, 5)


def ssim_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    pool = lambda x: torch.nn.functional.avg_pool2d(x, 7, stride=1)
    a, b = pool(prediction), pool(truth)
    va = (pool(prediction.square()) - a.square()).clamp_min(0)
    vb = (pool(truth.square()) - b.square()).clamp_min(0)
    cov = pool(prediction * truth) - a * b
    score = ((2*a*b + .01) * (2*cov + .09)) / ((a.square()+b.square()+.01)*(va+vb+.09))
    return 1 - score.mean()


def final_image(model: FullWindowCNN, image: np.ndarray) -> np.ndarray:
    if not np.any(image):
        return np.zeros_like(image)
    raw = whole_cnn(model.base, image)
    return np.clip(float(model.log_scale.detach().exp()) * raw + float(model.bias.detach()), -5, 5).astype(np.float32)


def training_cache(path: Path, frame: pd.DataFrame, cache: Path) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    files = []
    with open_cool(path) as cooler:
        chrom = cooler.chromnames[0]
        for i, record in enumerate(frame.sort_values("center").itertuples(), 1):
            target = cache / f"{record.id}.npz"
            raw = cooler.matrix(balance=False, sparse=True).fetch((chrom, int(record.window_start_bp), int(record.window_end_bp))).toarray().astype(np.float32)
            base, _, truth = prepare_full_pair(raw)
            np.savez_compressed(target, input=base, truth=truth)
            files.append(target)
            if i % 40 == 0 or i == len(frame):
                print(f"Cache training complete windows {i}/{len(frame)}", flush=True)
    return files


def train_epoch(model: FullWindowCNN, optimizer, files: list[Path], epoch: int) -> dict:
    rng = np.random.default_rng(SEED + epoch)
    torch.manual_seed(SEED + epoch)
    cache = OrderedDict()
    totals = []
    batch_x, batch_y = [], []
    model.train()
    for step in range(PATCHES_PER_EPOCH):
        index = int(rng.integers(len(files)))
        if index not in cache:
            with np.load(files[index]) as source:
                cache[index] = (source["input"], source["truth"])
            if len(cache) > 4:
                cache.popitem(last=False)
        cache.move_to_end(index)
        x, y = cache[index]
        a, b = rng.integers(0, len(x)-96+1, size=2)
        px, py = x[a:a+96, b:b+96].copy(), y[a+16:a+80, b+16:b+80].copy()
        if rng.random() < .5:
            px, py = px.T.copy(), py.T.copy()
        if rng.random() < .5:
            px, py = px[::-1, ::-1].copy(), py[::-1, ::-1].copy()
        batch_x.append(px)
        batch_y.append(py)
        if len(batch_x) == 8 or step == PATCHES_PER_EPOCH-1:
            inputs = torch.from_numpy(np.stack(batch_x)[:, None])
            truth = torch.from_numpy(np.stack(batch_y)[:, None])
            output = model(inputs)[:, :, 16:80, 16:80]
            mse = torch.nn.functional.mse_loss(output, truth)
            gradient = _gradient_loss(output, truth)
            structural = ssim_loss(output, truth)
            loss = mse + .15 * gradient + .5 * structural
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            totals.append([float(loss.detach()), float(mse.detach()), float(structural.detach())])
            batch_x, batch_y = [], []
    model.eval()
    average = np.mean(totals, axis=0)
    report = {"epoch": epoch, "loss": float(average[0]), "mse": float(average[1]), "ssim_loss": float(average[2]),
              "scale": float(model.log_scale.detach().exp()), "bias": float(model.bias.detach())}
    print(json.dumps(report), flush=True)
    return report


def evaluate(path: Path, frame: pd.DataFrame, model: FullWindowCNN, old_base,
             calibration: dict, out: Path, fractions: tuple = FRACTIONS,
             chosen_fraction: float | None = None, save: bool = False) -> pd.DataFrame:
    rows, shown = [], set()
    out.mkdir(parents=True, exist_ok=True)
    with open_cool(path) as cooler:
        chrom = cooler.chromnames[0]
        for i, record in enumerate(frame.sort_values("center").itertuples(), 1):
            raw = cooler.matrix(balance=False, sparse=True).fetch((chrom, int(record.window_start_bp), int(record.window_end_bp))).toarray().astype(np.float32)
            base, cubic, truth = prepare_full_pair(raw)
            prediction = final_image(model, base)
            cal_base = calibrate(base, calibration["bilinear"])
            images = {"bilinear": base, "bicubic": cubic,
                      "train_calibrated_bilinear": cal_base,
                      "train_calibrated_bicubic": calibrate(cubic, calibration["bicubic"])}
            images.update({f"adapted_cnn_fraction_{a:g}": a*prediction + (1-a)*cal_base for a in fractions})
            if save:
                images["frozen_main_whole"] = whole_cnn(old_base, base)
            target_parts = morphology(truth)
            for method, image in images.items():
                rows.append({"id": record.id, "type": record.type, "center_bp": record.center,
                             "start_bp": record.start, "end_bp": record.end,
                             "window_start_bp": record.window_start_bp, "window_end_bp": record.window_end_bp,
                             "native_bins": record.native_bins, "overlap_component": record.overlap_component,
                             "method": method, **image_metrics(truth, image, target_parts)})
            if save:
                final = images[f"adapted_cnn_fraction_{chosen_fraction:g}"]
                np.savez_compressed(out/f"{record.id}_full_maps.npz", truth=truth, bilinear=base, bicubic=cubic,
                                    prediction=final, calibrated_bilinear=cal_base,
                                    calibrated_bicubic=images["train_calibrated_bicubic"],
                                    start_bp=record.window_start_bp, end_bp=record.window_end_bp)
                fig, axes = plt.subplots(1,4,figsize=(12,3),dpi=140)
                for ax, image, title in zip(axes, (base,cal_base,final,truth),
                                            ("Bilinear","Train-calibrated bilinear","Full-window adapted CNN","Micro-C target")):
                    ax.imshow(image, origin="lower", cmap="magma", vmin=-2, vmax=4,
                              extent=[record.window_start_bp,record.window_end_bp]*2)
                    ax.set_title(title,fontsize=8)
                    ax.tick_params(labelsize=6)
                    ax.set(xlabel="position bp",ylabel="position bp")
                    for boundary in (record.start,record.end):
                        ax.axvline(boundary,color="cyan",lw=.35,alpha=.65)
                        ax.axhline(boundary,color="cyan",lw=.35,alpha=.65)
                fig.suptitle(f"{record.id} full span {record.length_bp} bp +160bp flanks",fontsize=10)
                fig.tight_layout()
                fig.savefig(out/f"{record.id}_full.png")
                plt.close(fig)
                if record.type not in shown:
                    shutil.copyfile(out/f"{record.id}_full.png",out/f"example_{record.type}.png")
                    shown.add(record.type)
            print(f"{out.name}: {i}/{len(frame)} {record.id}",flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out/"per_structure.csv",index=False)
    return result


def candidate_grid(frame: pd.DataFrame, epoch: int) -> list[dict]:
    grid = []
    for a in FRACTIONS:
        method = f"adapted_cnn_fraction_{a:g}"
        report = summarize(frame,method)
        feasible = report["all"]["psnr"]["mean_gain"] > 0 and report["all"]["ssim"]["mean_gain"] > 0
        feasible = feasible and all(report[k]["shape_r"]["mean_gain"] >= -.03 and report[k]["profile_r"]["mean_gain"] >= -.03 for k in PRIMARY_TYPES)
        grid.append({"epoch":epoch,"fraction":a,"method":method,"feasible":bool(feasible),
                     "psnr_gain":report["all"]["psnr"]["mean_gain"],"ssim_gain":report["all"]["ssim"]["mean_gain"],
                     "shape_gain":report["all"]["shape_r"]["mean_gain"],"profile_gain":report["all"]["profile_r"]["mean_gain"]})
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("select","test"),required=True)
    parser.add_argument("--rep1",type=Path,default=ROOT/"data/raw/GSE272159_37C_rep1.mapq_30.10.cool")
    parser.add_argument("--rep2",type=Path,default=ROOT/"data/raw/GSE272159_37C_rep2.mapq_30.10.cool")
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"results/task5_super_resolution.pt")
    parser.add_argument("--annotations",type=Path,default=ROOT/"data/structures.csv")
    parser.add_argument("--calibration",type=Path,default=ROOT/"outputs/task5_full_structure_20261014/scale_repair/trained_scale_parameters.json")
    parser.add_argument("--out",type=Path,default=ROOT/"outputs/task5_full_structure_20261014/neural_adaptation")
    parser.add_argument("--results-dir",type=Path,default=ROOT/"results")
    parser.add_argument("--threads",type=int,default=2)
    args = parser.parse_args()
    if args.phase == "select" and (args.out/"protocol.json").exists():
        raise FileExistsError("Select refuses to overwrite a frozen run; choose a fresh --out")
    if args.phase == "test" and ((args.out/"summary.json").exists() or list(args.results_dir.glob(PREFIX+"*"))):
        raise FileExistsError("Test refuses to overwrite published/frozen final results; use fresh output/results directory")
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    old_base, original = load_model(args.checkpoint)
    calibrated = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration = calibrated["parameters"]
    with open_cool(args.rep1) as cooler:
        genome_bp = int(cooler.chromsizes.iloc[0])
    cohort,split = cohort_manifest(annotations(args.annotations),genome_bp)
    sources = {"script":Path(__file__),"evaluator":ROOT/"scripts/task5_full_structure_verification_20261014.py",
               "scale_script":ROOT/"scripts/task5_full_structure_repair_20261014.py","deep":ROOT/"microc/deep.py",
               "data":ROOT/"microc/data.py","checkpoint":args.checkpoint,"annotations":args.annotations,
               "calibration":args.calibration,"rep1":args.rep1,"rep2":args.rep2}
    hashes = {key:digest(path) for key,path in sources.items()}
    for key in ("checkpoint","annotations","rep1","rep2"):
        if hashes[key] != calibrated["source_and_input_sha256"][key]:
            raise ValueError("Training calibration uses different source data/checkpoint")
    model_base,_ = load_model(args.checkpoint)
    model = FullWindowCNN(model_base,calibration["cnn"]["slope"],calibration["cnn"]["intercept"])
    if args.phase == "select":
        write_json(args.out/"protocol.json",{"experiment":"task5-full-neural-adaptation-20261014","frozen_at_utc":utc(),
                   "code_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                   "source_and_input_sha256":hashes,"split":split,"device":"cpu","threads":args.threads,
                   "dependencies":{n:importlib.metadata.version(n) for n in ("torch","numpy","pandas","scipy","scikit-image","cooler")},
                   "architecture":"Original48-channel4-residual-block CNN + trainable positive affine output scale; no local transpose symmetrization on offdiagonal tiles",
                   "training":"TRAIN complete windows only; random96px context,64px core;800 patches per epoch,batch8,Adam1e-4; MSE+0.15gradient+0.5localSSIM;transpose/genomic-reversal augmentations;norm clip5",
                   "fixed_candidates":{"epochs":list(EPOCHS),"cnn_fractions":list(FRACTIONS)},
                   "selection":"Validation complete windows only;PSNR/SSIM gains>0 and CHIN/OPCID shape/profile mean gains>=-.03; maximize SSIM thenPSNR; fallback finalepoch fraction1",
                   "controls":"Raw bilinear/bicubic and train-affine-calibrated bilinear/bicubic from identical training cohort; frozen original wholeCNN retained on test",
                   "success":"Prerecorded whole-window image/shape/profile operational criteria from first full verification; original-image peaks supplementary, not ranker output",
                   "caution":"Retrospective spatial cohort; initial global inference and scale-only validation failures preserved; current test/rep2 not used to tune"})
        cohort.to_csv(args.out/"cohort_manifest.csv",index=False)
        files = training_cache(args.rep1,cohort[cohort.split=="train"],args.out/"train_cache")
        write_json(args.out/"training_cache_receipt.json",{"files":{str(p.relative_to(args.out)):digest(p) for p in files},"training_ids":cohort[cohort.split=="train"].id.tolist()})
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
        grid,history = [],[]
        for epoch in range(1,max(EPOCHS)+1):
            history.append(train_epoch(model,optimizer,files,epoch))
            if epoch in EPOCHS:
                torch.save({"state_dict":model.state_dict(),"epoch":epoch},args.out/f"epoch{epoch}.pt")
                frame = evaluate(args.rep1,cohort[cohort.split=="validation"],model,old_base,calibration,args.out/f"validation_epoch{epoch}")
                candidates = candidate_grid(frame,epoch)
                grid.extend(candidates)
                print(json.dumps({"epoch":epoch,"validation_grid":candidates}),flush=True)
        usable = [row for row in grid if row["feasible"]]
        chosen = max(usable,key=lambda r:(r["ssim_gain"],r["psnr_gain"])) if usable else grid[-1]
        shutil.copyfile(args.out/f"epoch{chosen['epoch']}.pt",args.out/"selected_model.pt")
        write_json(args.out/"training_history.json",{"history":history})
        write_json(args.out/"selection_freeze.json",{"frozen_at_utc":utc(),"selected":chosen,"all_candidates":grid,
                   "source_and_input_sha256":hashes,"protocol_sha256":digest(args.out/"protocol.json"),
                   "cohort_sha256":digest(args.out/"cohort_manifest.csv"),"model_sha256":digest(args.out/"selected_model.pt"),
                   "training_cache_receipt_sha256":digest(args.out/"training_cache_receipt.json"),"test_read":False,"rep2_matrix_read":False})
        print(json.dumps({"frozen_selected":chosen},indent=2),flush=True)
    else:
        freeze = json.loads((args.out/"selection_freeze.json").read_text(encoding="utf-8"))
        if hashes != freeze["source_and_input_sha256"] or digest(args.out/"protocol.json") != freeze["protocol_sha256"]:
            raise ValueError("Source/protocol drift after model freeze")
        if digest(args.out/"selected_model.pt") != freeze["model_sha256"] or digest(args.out/"cohort_manifest.csv") != freeze["cohort_sha256"]:
            raise ValueError("Frozen model/cohort drift")
        saved = torch.load(args.out/"selected_model.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        a = float(freeze["selected"]["fraction"])
        method = f"adapted_cnn_fraction_{a:g}"
        result = {"selected":freeze["selected"],"selected_method":method,"selection_freeze_sha256":digest(args.out/"selection_freeze.json"),
                  "replicates":{},"scope":"Full known intervals +160bp flanks, native10bp output,simulated40bp input"}
        for name,path in (("rep1_test",args.rep1),("rep2_fixed_test",args.rep2)):
            frame = evaluate(path,cohort[cohort.split=="test"],model,old_base,calibration,args.out/name,(a,),a,True)
            selected = summarize(frame,method)
            result["replicates"][name] = {"selected":selected,"success":criterion(selected),
                                          "controls":{m:summarize(frame,m) for m in frame.method.unique() if m!=method}}
        result["overall_operational_success"] = all(item["success"]["operational_course_success"] for item in result["replicates"].values())
        result["wall_seconds"] = time.perf_counter()-started
        write_json(args.out/"summary.json",result)
        artifacts = {str(p.relative_to(args.out)):digest(p) for p in args.out.rglob("*") if p.is_file() and p.name!="receipt.json"}
        write_json(args.out/"receipt.json",{"completed_at_utc":utc(),"source_and_input_sha256":hashes,"output_sha256":artifacts,
                   "validation_only_selection":True,"test_only_after_freeze":True,"old_checkpoint_unchanged":True,
                   "published_prefix":PREFIX,"results_dir":str(args.results_dir.resolve())})
        args.results_dir.mkdir(parents=True,exist_ok=True)
        for name in ("protocol.json","cohort_manifest.csv","selection_freeze.json","training_history.json","training_cache_receipt.json","summary.json","receipt.json","selected_model.pt"):
            target = args.results_dir/(PREFIX+name)
            if target.exists():
                raise FileExistsError("Refusing result overwrite")
            shutil.copyfile(args.out/name,target)
        for name in result["replicates"]:
            shutil.copyfile(args.out/name/"per_structure.csv",args.results_dir/(PREFIX+name+".csv"))
            for kind in ("CHIN","OPCID","CHID"):
                shutil.copyfile(args.out/name/f"example_{kind}.png",args.results_dir/(PREFIX+name+"_"+kind+".png"))
        print(json.dumps({"overall_operational_success":result["overall_operational_success"],
                          "success":{name:item["success"] for name,item in result["replicates"].items()}},indent=2),flush=True)


if __name__ == "__main__":
    main()

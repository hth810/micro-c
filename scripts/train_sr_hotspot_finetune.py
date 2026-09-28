"""Small, validation-frozen hotspot-aware fine-tuning experiment for task 5.

Run --phase select first. It reads ONLY training/validation genomic segments,
trains one fixed trajectory with checkpoints after 2 and 5 epochs, and writes
selection.json. Run --phase test separately to evaluate that frozen selection
on the previously inspected rep1 test segment and fixed rep2 replicate. This
is exploratory rather than a new blinded test; no test score may select a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from microc.data import open_cool, window
from microc.deep import _gradient_loss, prepare_sr_pair
from scripts.evaluate_sr_ensemble import infer, load_model, split_centers
from scripts.evaluate_sr_peak_guard import against, paired_report, per_patch
from scripts.validate_sr_known_shape import hotspot


SEED = 20260928
CHECKPOINT_EPOCHS = (2, 5)
LEARNING_RATE = 1e-4
HOTSPOT_WEIGHT = 0.05
HOTSPOT_TEMPERATURE = 0.75
HOTSPOT_GAUSSIAN_SIGMA = 3.0
PSNR_TOLERANCE_DB = 0.05
SSIM_TOLERANCE = 0.001


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_spec(cooler, annotations: pd.DataFrame, bins: int, factor: int,
               max_patches: int) -> dict:
    chrom = cooler.chromnames[0]
    patch_bp = bins * int(cooler.binsize)
    centers, train_end, test_start = split_centers(
        int(cooler.chromsizes[chrom]), patch_bp, max_patches)
    # The extra patch-width buffer ensures no annotated train patch reaches
    # the first validation tile. Validation/test annotation rules match prior
    # task-5 reports, so their fixed benchmark populations remain unchanged.
    train_ann = annotations[annotations.center < centers[train_end] - patch_bp]
    val_ann = annotations[(annotations.center >= centers[train_end]) &
                          (annotations.center < centers[test_start])]
    test_ann = annotations[annotations.center >= centers[test_start]]
    spec = {"chrom": chrom, "patch_bp": patch_bp, "bins": bins, "factor": factor,
            "max_patches": max_patches, "centers": centers,
            "train_end": train_end, "test_start": test_start,
            "train_known_ids": train_ann.id.tolist(),
            "validation_known_ids": val_ann.id.tolist(),
            "test_known_ids": test_ann.id.tolist()}
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    spec["split_sha256"] = hashlib.sha256(canonical).hexdigest()
    return spec


def annotation_split(frame: pd.DataFrame, spec: dict, phase: str) -> pd.DataFrame:
    centers, train_end, test_start = spec["centers"], spec["train_end"], spec["test_start"]
    if phase == "train":
        return frame[frame.center < centers[train_end] - spec["patch_bp"]]
    if phase == "validation":
        return frame[(frame.center >= centers[train_end]) & (frame.center < centers[test_start])]
    if phase == "test":
        return frame[frame.center >= centers[test_start]]
    raise ValueError(phase)


def load_pairs(cooler, spec: dict, locations) -> tuple[np.ndarray, np.ndarray]:
    pairs = [prepare_sr_pair(window(cooler, spec["chrom"], int(center),
                                    spec["patch_bp"], spec["bins"]), spec["factor"])
             for center in locations]
    return (np.stack([pair[0] for pair in pairs]),
            np.stack([pair[1] for pair in pairs]))


def peak_density(truth: np.ndarray, sigma: float = HOTSPOT_GAUSSIAN_SIGMA) -> np.ndarray:
    """Soft target around the truth peak on an upper-triangle mask only."""
    n = truth.shape[-1]
    upper = np.triu(np.ones((n, n), dtype=bool), k=3)
    yy, xx = np.indices((n, n))
    densities = []
    for image in truth:
        row, col = hotspot(image, upper)
        gaussian = np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma ** 2))
        weights = gaussian[upper]
        densities.append(weights / weights.sum())
    return np.asarray(densities, dtype=np.float32)


def soft_peak_loss(prediction: torch.Tensor, target_prob: torch.Tensor,
                   temperature: float = HOTSPOT_TEMPERATURE) -> torch.Tensor:
    n = prediction.shape[-1]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=prediction.device), diagonal=3)
    logits = prediction[:, 0][:, upper] / temperature
    return -(target_prob * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def choose_stage(rows: list[dict]) -> dict:
    """Frozen rule, with no CHID selection (only two validation CHIDs)."""
    main = next(row for row in rows if row["stage"] == "main")
    feasible = [row for row in rows if row["stage"] != "main"
                and row["tile_psnr"] >= main["tile_psnr"] - PSNR_TOLERANCE_DB
                and row["tile_ssim"] >= main["tile_ssim"] - SSIM_TOLERANCE
                and row["known_hotspot_error"] < main["known_hotspot_error"]
                and row["CHIN_hotspot_error"] < main["CHIN_hotspot_error"]]
    if not feasible:
        return main
    # Equal importance for common CHIN and all known types, normalized by
    # main-model validation error. Tie-breaking favors fewer training epochs.
    return min(feasible, key=lambda row: (
        row["CHIN_hotspot_error"] / main["CHIN_hotspot_error"]
        + row["known_hotspot_error"] / main["known_hotspot_error"], row["epochs"]))


def score_validation(model, tile_base, tile_truth, tile_centers,
                     known_base, known_truth, known_ann) -> dict:
    tile_prediction = infer(model, tile_base)
    known_prediction = infer(model, known_base)
    tile = per_patch(tile_truth, tile_base, tile_prediction, tile_centers)
    known = per_patch(known_truth, known_base, known_prediction,
                      known_ann.center.to_numpy(), known_ann.type.to_numpy(), known_ann.id.to_numpy())
    chin = known[known.type == "CHIN"]
    return {"tile_psnr": float(tile.prediction_psnr.mean()),
            "tile_ssim": float(tile.prediction_ssim.mean()),
            "known_hotspot_error": float(known.prediction_hotspot_error.mean()),
            "CHIN_hotspot_error": float(chin.prediction_hotspot_error.mean()),
            "known_top5_error": float(known.prediction_top5_error.mean()),
            "CHIN_top5_error": float(chin.prediction_top5_error.mean()),
            "baseline_tile_psnr": float(tile.baseline_psnr.mean()),
            "baseline_tile_ssim": float(tile.baseline_ssim.mean()),
            "baseline_known_hotspot_error": float(known.baseline_hotspot_error.mean()),
            "baseline_CHIN_hotspot_error": float(chin.baseline_hotspot_error.mean())}


def select(args, model, check, spec, ann):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    centers, train_end, test_start = spec["centers"], spec["train_end"], spec["test_start"]
    train_ann = annotation_split(ann, spec, "train")
    val_ann = annotation_split(ann, spec, "validation")
    if len(train_ann) < 100 or len(val_ann) < 15 or (val_ann.type == "CHIN").sum() < 10:
        raise ValueError("Insufficient train/validation annotations")
    with open_cool(args.cool) as cooler:
        train_base, train_truth = load_pairs(cooler, spec, centers[:train_end])
        known_base, known_truth = load_pairs(cooler, spec, train_ann.center)
        val_centers = np.asarray(centers[train_end:test_start])
        val_base, val_truth = load_pairs(cooler, spec, val_centers)
        val_known_base, val_known_truth = load_pairs(cooler, spec, val_ann.center)
    upper = np.triu(np.ones((spec["bins"], spec["bins"]), dtype=bool), k=3)
    informative = np.std(known_truth[:, upper], axis=1) > 1e-8
    known_base, known_truth = known_base[informative], known_truth[informative]
    known_prob = peak_density(known_truth)
    x_train = torch.from_numpy(train_base[:, None])
    y_train = torch.from_numpy(train_truth[:, None])
    x_known = torch.from_numpy(known_base[:, None])
    y_known = torch.from_numpy(known_truth[:, None])
    p_known = torch.from_numpy(known_prob)
    base_row = {"stage": "main", "epochs": 0, "checkpoint": str(args.main_checkpoint),
                "sha256": digest(args.main_checkpoint),
                **score_validation(model, val_base, val_truth, val_centers,
                                   val_known_base, val_known_truth, val_ann)}
    rows = [base_row]
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
        model.train()
        for batch in torch.randperm(len(x_train)).split(16):
            known_idx = torch.randint(len(x_known), (8,))
            output = model(x_train[batch])
            known_output = model(x_known[known_idx])
            tile_loss = nn.functional.mse_loss(output, y_train[batch]) + .15 * _gradient_loss(output, y_train[batch])
            known_loss = (nn.functional.mse_loss(known_output, y_known[known_idx])
                          + .15 * _gradient_loss(known_output, y_known[known_idx]))
            loss = tile_loss + .5 * known_loss + HOTSPOT_WEIGHT * soft_peak_loss(
                known_output, p_known[known_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        if epoch in CHECKPOINT_EPOCHS:
            path = args.out / f"hotspot_finetune_epoch{epoch}.pt"
            torch.save({"model_state_dict": model.state_dict(), "factor": check["factor"],
                        "patch_bins": check["patch_bins"], "blend": 1.0,
                        "seed": SEED, "epochs": epoch, "hotspot_weight": HOTSPOT_WEIGHT}, path)
            row = {"stage": f"epoch{epoch}", "epochs": epoch, "checkpoint": str(path),
                   "sha256": digest(path),
                   **score_validation(model, val_base, val_truth, val_centers,
                                      val_known_base, val_known_truth, val_ann)}
            rows.append(row)
            print(json.dumps({"epoch": epoch, "validation": row}, indent=2), flush=True)
    selected = choose_stage(rows)
    pd.DataFrame(rows).to_csv(args.out / "validation_stages.csv", index=False)
    report = {"selection_rule": "Require validation tile PSNR >= main-0.05 dB, SSIM >= main-0.001, overall known top1 error < main, CHIN top1 error < main; minimize equal-weight sum of relative CHIN and overall top1 error; ties choose fewer epochs; otherwise main",
              "selected_stage": selected["stage"], "selected_checkpoint": selected["checkpoint"],
              "selected_checkpoint_sha256": selected["sha256"],
              "baseline_checkpoint": str(args.main_checkpoint),
              "baseline_checkpoint_sha256": base_row["sha256"],
              "annotation_sha256": digest(args.annotations),
              "split_sha256": spec["split_sha256"],
              "split_counts": {"train_tiles": train_end, "validation_tiles": test_start - train_end,
                               "test_tiles": len(centers) - test_start,
                               "train_known": train_ann.type.value_counts().to_dict(),
                               "validation_known": val_ann.type.value_counts().to_dict(),
                               "test_known": annotation_split(ann, spec, "test").type.value_counts().to_dict(),
                               "informative_train_known": int(informative.sum())},
              "fixed_training": {"learning_rate": LEARNING_RATE, "hotspot_weight": HOTSPOT_WEIGHT,
                                 "hotspot_temperature": HOTSPOT_TEMPERATURE,
                                 "hotspot_target_gaussian_sigma": HOTSPOT_GAUSSIAN_SIGMA,
                                 "epochs_checked": list(CHECKPOINT_EPOCHS),
                                 "regular_batch": 16, "known_batch": 8,
                                 "loss": "regular MSE + 0.15 gradient + 0.5*(known MSE + 0.15 gradient) + 0.05 soft-hotspot cross-entropy"},
              "validation_stages": rows,
              "caution": "Validation CHID n=2 is not used for selection. Test segment has been seen by previous studies; not blind. Low resolution simulated from high-resolution Micro-C."}
    (args.out / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_stage": selected["stage"],
                      "split_sha256": spec["split_sha256"],
                      "selection_file": str(args.out / "selection.json")}, indent=2), flush=True)


def test(args, check, spec, ann):
    selection_path = args.out / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection["split_sha256"] != spec["split_sha256"]
            or selection["baseline_checkpoint_sha256"] != digest(args.main_checkpoint)
            or selection["annotation_sha256"] != digest(args.annotations)):
        raise ValueError("Selection provenance or data split changed")
    selected_path = Path(selection["selected_checkpoint"])
    if digest(selected_path) != selection["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed since validation")
    selected_model, _ = load_model(selected_path)
    main_model, _ = load_model(args.main_checkpoint)
    test_ann = annotation_split(ann, spec, "test")
    test_centers = np.asarray(spec["centers"][spec["test_start"]:])

    def evaluate(cool_path: Path, label: str) -> dict:
        with open_cool(cool_path) as cooler:
            if (cooler.chromnames[0] != spec["chrom"]
                    or int(cooler.binsize) * spec["bins"] != spec["patch_bp"]):
                raise ValueError("External replicate resolution or chromosome mismatch")
            base_tile, truth_tile = load_pairs(cooler, spec, test_centers)
            base_known, truth_known = load_pairs(cooler, spec, test_ann.center)
        tile_selected = per_patch(truth_tile, base_tile, infer(selected_model, base_tile), test_centers)
        tile_main = per_patch(truth_tile, base_tile, infer(main_model, base_tile), test_centers)
        known_selected = per_patch(truth_known, base_known, infer(selected_model, base_known),
                                   test_ann.center.to_numpy(), test_ann.type.to_numpy(), test_ann.id.to_numpy())
        known_main = per_patch(truth_known, base_known, infer(main_model, base_known),
                               test_ann.center.to_numpy(), test_ann.type.to_numpy(), test_ann.id.to_numpy())
        tile_selected.to_csv(args.out / f"{label}_selected_tiles.csv", index=False)
        known_selected.to_csv(args.out / f"{label}_selected_known.csv", index=False)
        stage_name = selection["selected_stage"]
        return {"selected_vs_interpolation_tiles": paired_report(tile_selected, "bilinear interpolation", stage_name),
                "selected_vs_interpolation_known": paired_report(known_selected, "bilinear interpolation", stage_name),
                "selected_vs_main_tiles": paired_report(against(tile_selected, tile_main), "main CNN", stage_name),
                "selected_vs_main_known": paired_report(against(known_selected, known_main), "main CNN", stage_name),
                "by_type_vs_interpolation": {name: paired_report(group, "bilinear interpolation", stage_name)
                                             for name, group in known_selected.groupby("type")},
                "by_type_vs_main": {name: paired_report(against(group, known_main[known_main.type == name]),
                                                  "main CNN", stage_name)
                                    for name, group in known_selected.groupby("type")}}

    result = {"selected_stage": selection["selected_stage"],
              "selection_sha256": digest(selection_path),
              "split_sha256": spec["split_sha256"],
              "rep1_test": evaluate(args.cool, "rep1_test")}
    if args.external_cool:
        result["rep2_fixed"] = evaluate(args.external_cool, "rep2_fixed")
    result["caution"] = selection["caution"]
    (args.out / "test_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selected_stage": result["selected_stage"],
                      "rep1_known_vs_main_hotspot": result["rep1_test"]["selected_vs_main_known"]["hotspot_error"],
                      "rep2_known_vs_main_hotspot": result.get("rep2_fixed", {}).get("selected_vs_main_known", {}).get("hotspot_error")},
                     indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("select", "test"), required=True)
    ap.add_argument("--cool", type=Path, required=True)
    ap.add_argument("--external-cool", type=Path)
    ap.add_argument("--annotations", type=Path, default=Path("data/structures.csv"))
    ap.add_argument("--main-checkpoint", type=Path, default=Path("results/task5_super_resolution.pt"))
    ap.add_argument("--max-patches", type=int, default=1200)
    ap.add_argument("--out", type=Path, default=Path("outputs/task5_hotspot_finetune"))
    args = ap.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, check = load_model(args.main_checkpoint)
    ann = pd.read_csv(args.annotations).sort_values("center")
    with open_cool(args.cool) as cooler:
        spec = split_spec(cooler, ann, int(check["patch_bins"]),
                          int(check["factor"]), args.max_patches)
    if args.phase == "select":
        select(args, model, check, spec, ann)
    else:
        test(args, check, spec, ann)


if __name__ == "__main__":
    main()

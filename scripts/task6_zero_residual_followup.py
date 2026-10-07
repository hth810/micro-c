"""Frozen objective-alignment sensitivity, not independent 3-D validation.

The historical circular MLP fits positive contacts, whereas matched AUC also
depends on clipped-zero residuals. This run adds ONE fixed ranking term using
rep1 positive/zero pairs at the SAME exact ring offset. A zero is not physical
noncontact. The full rep1 target is the model input; no pair holdout is claimed.
All seeds are reported and coordinates are frozen before opening rep2.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from microc.circular import circular_contact_target, circular_pairs, cyclic_chain_variance
from microc.data import open_cool
from microc.deep import GenomeMLP, _coarsen, seed_all
from scripts.task6_circular_sensitivity import evaluate, sha, write_json


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261006, 20261007, 20261008)
EPOCHS = 200
PAIR_COUNT = 20000
ALPHA = .1
MARGIN = .2
MATCH_SEED = 20261006


def matched_ranking_pairs(contact: np.ndarray, seed: int = MATCH_SEED):
    """Fixed rep1-only, without-replacement balanced pairs per ring offset.

    Returns positive endpoints, clipped-zero endpoints and their shared exact
    offset. Unsupported offsets are omitted, not approximated by nearby ones.
    """
    contact = np.asarray(contact, dtype=np.float64)
    if (contact.ndim != 2 or contact.shape[0] != contact.shape[1] or
            not np.isfinite(contact).all() or (contact < 0).any() or
            not np.allclose(contact, contact.T, rtol=1e-8, atol=1e-8)):
        raise ValueError("Need a finite nonnegative symmetric square target")
    i, j, offsets = circular_pairs(len(contact))
    positive = contact[i, j] > 0
    rng = np.random.default_rng(seed)
    pos, zero, shared, receipt = [], [], [], []
    for offset in np.unique(offsets):
        p = np.flatnonzero((offsets == offset) & positive)
        z = np.flatnonzero((offsets == offset) & ~positive)
        count = min(len(p), len(z))
        if count:
            pos.append(rng.choice(p, count, replace=False))
            zero.append(rng.choice(z, count, replace=False))
            shared.append(np.full(count, offset, dtype=np.int64))
            receipt.append({"exact_ring_offset": int(offset), "positive_available": len(p),
                            "zero_available": len(z), "matched_count": count})
    if not pos:
        raise ValueError("No ring offset supports both positive and clipped-zero residuals")
    p, z = np.concatenate(pos), np.concatenate(zero)
    return (i[p], j[p], i[z], j[z], np.concatenate(shared)), receipt


def ranking_loss(coordinates: torch.Tensor, positive_i, positive_j,
                 zero_i, zero_j):
    positive_d = torch.linalg.vector_norm(coordinates[positive_i] - coordinates[positive_j], dim=1)
    zero_d = torch.linalg.vector_norm(coordinates[zero_i] - coordinates[zero_j], dim=1)
    return torch.nn.functional.softplus(MARGIN + positive_d - zero_d).mean()


def train(contact: np.ndarray, matches: tuple, seed: int,
          epochs: int = EPOCHS, pair_count: int = PAIR_COUNT):
    seed_all(seed)
    i, j, _ = circular_pairs(len(contact))
    keep = contact[i, j] > 0
    i, j = i[keep], j[keep]
    if not len(i) or not len(matches[0]):
        raise ValueError("Training requires positive edges and matched residual pairs")
    target = np.clip(1 / (contact[i, j] + .05), 0, 20)
    goal = torch.tensor(target / max(float(np.median(target)), 1e-8), dtype=torch.float32)
    inputs = torch.tensor(contact, dtype=torch.float32)
    model = GenomeMLP(len(contact))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pi, pj, zi, zj = (torch.from_numpy(np.asarray(a, dtype=np.int64)) for a in matches[:4])
    history = []
    for epoch in range(epochs):
        model.train()
        xyz = model(inputs)
        take = torch.randint(len(i), (min(pair_count, len(i)),))
        actual = torch.linalg.vector_norm(xyz[i[take]] - xyz[j[take]], dim=1)
        edge = torch.nn.functional.smooth_l1_loss(actual, goal[take])
        ranking_take = torch.randint(len(pi), (min(pair_count, len(pi)),))
        rank = ranking_loss(xyz, pi[ranking_take], pj[ranking_take], zi[ranking_take], zj[ranking_take])
        chain = cyclic_chain_variance(xyz)
        loss = edge + .1 * chain + ALPHA * rank
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "total_loss": float(loss.detach()),
                            "positive_edge_loss": float(edge.detach()),
                            "cyclic_chain_variance": float(chain.detach()),
                            "matched_ranking_loss": float(rank.detach())})
    model.eval()
    with torch.no_grad():
        coordinates = model(inputs).numpy()
    if not np.isfinite(coordinates).all():
        raise ValueError("Nonfinite trained coordinates")
    return model, coordinates, history


def freeze_protocol(args: argparse.Namespace) -> dict:
    sources = ("scripts/task6_zero_residual_followup.py", "tests/test_task6_zero_residual_followup.py",
               "scripts/task6_circular_sensitivity.py", "microc/circular.py", "microc/deep.py",
               "microc/data.py", "scripts/annotate_3d_domains.py")
    return {"experiment": "task6-zero-residual-objective-20261006",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {p: sha(ROOT / p) for p in sources},
            "input_sha256": {"rep1": sha(args.rep1), "rep2": sha(args.rep2),
                             "published_coordinates": sha(args.published),
                             "published_protocol": sha(args.published_protocol)},
            "input_paths": {k: str(getattr(args, k)) for k in
                            ("rep1", "rep2", "published", "published_protocol")},
            "environment": {k: importlib.metadata.version(k) for k in
                            ("torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "model": "GenomeMLP, full rep1 circular contact target as input, no MDS prior",
            "seeds": list(SEEDS), "epochs": EPOCHS, "pairs_per_term_per_epoch": PAIR_COUNT,
            "optimizer": {"name": "Adam", "learning_rate": .001},
            "objective": "positive inverse-distance smooth-L1 + 0.1 cyclic-chain length variance + 0.1 softplus(0.2+d_positive-d_zero)",
            "alpha": ALPHA, "margin": MARGIN, "ranking_pair_seed": MATCH_SEED,
            "ranking_pairs": "rep1-only exact ring offset >=2, positive/clipped-zero balanced without replacement within offsets; fixed list sampled per epoch",
            "selection": "None. One predeclared alpha/margin; report all three seeds, export first seed not best; no architecture, epoch, seed, or coefficient selection using rep2",
            "coordinate_freeze": "Hash all coordinate models and first-seed checkpoint before opening rep2 as a cooler",
            "evaluation": "Rep2 same circular target: exact-offset-matched AUC plus all-positive, top20, weaker-positive and origin-crossing 2-to-20 Spearman",
            "comparators": "Published circular MLP seeds, circular MDS, same-chain linear-target MLP and legacy MLP evaluated on the same current circular target",
            "caution": "Retrospective objective-alignment sensitivity. Rep2 was historically inspected. Full contact matrix input means no independent pair holdout. Clipped-zero residuals are not physical noncontacts. No external 3-D truth. Final partial bin uses index-offset approximation."}


def figure(metrics: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(metrics["exact_circular_offset_matched_auc"]["auc"])
    names.remove("exact_ring_offset_baseline")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for ax, metric in zip(axes, ("auc", "all_positive", "origin_crossing_2_to_20")):
        source = (metrics["exact_circular_offset_matched_auc"]["auc"] if metric == "auc"
                  else metrics["strata"][metric])
        values = [source[name] for name in names]
        colors = ["#d37b35" if name.startswith("rank_mlp") else
                  ("#477ea1" if "mds" in name else "#999999") for name in names]
        ax.barh(np.arange(len(names)), values, color=colors)
        ax.set_yticks(np.arange(len(names)), names, fontsize=7)
        ax.invert_yaxis()
        ax.set(title=metric, xlabel="AUC" if metric == "auc" else "Spearman")
        ax.axvline(.5 if metric == "auc" else 0, color="black", linewidth=.7)
    fig.suptitle("Fixed positive / zero-residual ranking sensitivity on rep2\nAll seeds retained; zeros are normalized residuals, not physical noncontacts", fontsize=12)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(2)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a fresh empty output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = freeze_protocol(args)
    write_json(args.out / "protocol.json", protocol)
    print(f"Protocol frozen {protocol['frozen_at_utc']}", flush=True)
    # Verify that historical comparator bytes belong to the same source data
    # and normalization implementation rather than trusting a file name.
    old_protocol = json.loads(args.published_protocol.read_text(encoding="utf-8"))
    if (old_protocol["rep1_sha256"] != protocol["input_sha256"]["rep1"] or
            old_protocol["rep2_sha256"] != protocol["input_sha256"]["rep2"] or
            old_protocol["module_sha256"] != sha(ROOT / "microc/circular.py")):
        raise ValueError("Published comparators use a different input or circular target implementation")
    with open_cool(args.rep1) as cool:
        chrom = cool.chromnames[0]
        genome_bp = int(cool.chromsizes[chrom])
        target1 = circular_contact_target(_coarsen(cool, chrom, 5000))
    matches, match_receipt = matched_ranking_pairs(target1)
    pd.DataFrame({"positive_i": matches[0], "positive_j": matches[1],
                  "zero_i": matches[2], "zero_j": matches[3],
                  "exact_ring_offset": matches[4]}).to_csv(args.out / "ranking_pairs.csv", index=False)
    write_json(args.out / "ranking_pair_receipt.json", {
        "selected_using": "rep1 only", "seed": MATCH_SEED,
        "total_matched_rank_pairs": len(matches[0]), "offsets": match_receipt,
        "ranking_pairs_sha256": sha(args.out / "ranking_pairs.csv")})
    with np.load(args.published) as saved:
        coordinates = {name: saved[name].copy() for name in saved.files}
    if any(xyz.shape != (len(target1), 3) or not np.isfinite(xyz).all()
           for xyz in coordinates.values()):
        raise ValueError("Published coordinate dimensions do not match rep1")
    histories = {}
    for seed in SEEDS:
        model, xyz, history = train(target1, matches, seed)
        coordinates[f"rank_mlp_seed{seed}"] = xyz
        histories[str(seed)] = history
        if seed == SEEDS[0]:
            torch.save({"model_state_dict": model.state_dict(), "seed": seed,
                        "bin_bp": 5000, "genome_bp": genome_bp,
                        "protocol_sha256": sha(args.out / "protocol.json")}, args.out / "first_seed.pt")
        print(f"Completed fixed seed {seed}; {time.perf_counter() - started:.1f}s", flush=True)
    np.savez_compressed(args.out / "coordinate_models.npz", **coordinates)
    n = len(target1)
    first = coordinates[f"rank_mlp_seed{SEEDS[0]}"]
    pd.DataFrame({"chrom": chrom, "start": np.arange(n) * 5000,
                  "end": np.minimum((np.arange(n) + 1) * 5000, genome_bp),
                  "x": first[:, 0], "y": first[:, 1], "z": first[:, 2]}).to_csv(
                  args.out / "coordinates.csv", index=False)
    write_json(args.out / "training_history.json", histories)
    freeze = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "rep2_cooler_opened_before_freeze": False,
              "output_sha256": {p: sha(args.out / p) for p in
                ("protocol.json", "ranking_pairs.csv", "ranking_pair_receipt.json",
                 "coordinate_models.npz", "coordinates.csv", "first_seed.pt", "training_history.json")}}
    write_json(args.out / "coordinate_freeze.json", freeze)
    print(f"Coordinates frozen {freeze['frozen_at_utc']}; opening rep2 for fixed evaluation", flush=True)
    with open_cool(args.rep2) as cool:
        if cool.chromnames[0] != chrom or int(cool.chromsizes[chrom]) != genome_bp:
            raise ValueError("Replicate genome dimensions differ")
        target2 = circular_contact_target(_coarsen(cool, chrom, 5000))
    rep1_metrics, rep2_metrics = evaluate(target1, coordinates), evaluate(target2, coordinates)
    rows = []
    for name in coordinates:
        row = {"model": name, "rep2_exact_offset_matched_auc":
               rep2_metrics["exact_circular_offset_matched_auc"]["auc"].get(name)}
        row.update({f"rep2_{s}_rho": rep2_metrics["strata"][s][name]
                    for s in rep2_metrics["strata"]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out / "comparison.csv", index=False)
    figure(rep2_metrics, args.out / "comparison.png")
    # Source drift would invalidate the run even if its metrics looked good.
    for p, digest in protocol["source_sha256"].items():
        if sha(ROOT / p) != digest:
            raise RuntimeError(f"Frozen source drift: {p}")
    for name, path in (("rep1", args.rep1), ("rep2", args.rep2),
                       ("published_coordinates", args.published), ("published_protocol", args.published_protocol)):
        if sha(path) != protocol["input_sha256"][name]:
            raise RuntimeError(f"Frozen input drift: {name}")
    summary = {"experiment": protocol["experiment"], "protocol_sha256": sha(args.out / "protocol.json"),
               "completed_at_utc": datetime.now(timezone.utc).isoformat(), "coordinate_freeze": freeze,
               "bins": n, "genome_bp": genome_bp, "last_bin_bp": genome_bp - (n - 1) * 5000,
               "ranking_pairs": len(matches[0]), "rep1_in_sample": rep1_metrics,
               "rep2_cross_replicate": rep2_metrics,
               "seed_distance_spearman": {f"{a}__{b}": float(spearmanr(pdist(coordinates[a]), pdist(coordinates[b])).statistic)
                    for ix, a in enumerate(coordinates) for b in list(coordinates)[ix + 1:]
                    if a.startswith("rank_mlp") and b.startswith("rank_mlp")},
               "caution": protocol["caution"], "wall_seconds": time.perf_counter() - started,
               "conclusion_rule": "Retain every seed and baseline. Report AUC and all four rho strata together; do not select the best seed or hide objective tradeoffs."}
    write_json(args.out / "summary.json", summary)
    if args.publish:
        # The large deterministic ranking CSV remains local and is byte-hashed
        # in the receipt. Git keeps compact receipts and model/metric evidence.
        files = ("protocol.json", "summary.json", "coordinate_freeze.json", "ranking_pair_receipt.json",
                 "coordinate_models.npz", "coordinates.csv", "first_seed.pt", "training_history.json",
                 "comparison.csv", "comparison.png")
        prefix = "task6_followup_20261006_"
        if any((ROOT / "results" / (prefix + p)).exists() for p in files):
            raise ValueError("Published follow-up files already exist")
        for p in files:
            shutil.copyfile(args.out / p, ROOT / "results" / (prefix + p))
        write_json(ROOT / "results" / (prefix + "evidence_index.json"), {
            "published_at_utc": datetime.now(timezone.utc).isoformat(), "output_directory": str(args.out),
            "source_sha256": protocol["source_sha256"], "input_sha256": protocol["input_sha256"],
            "files": {prefix + p: sha(ROOT / "results" / (prefix + p)) for p in files},
            "command": "python -m scripts.task6_zero_residual_followup --publish",
            "local_ranking_pairs_sha256": sha(args.out / "ranking_pairs.csv")})
    print(json.dumps(rows, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep1", type=Path, default=Path("data/raw/GSE272159_37C_rep1.mapq_30.10.cool"))
    parser.add_argument("--rep2", type=Path, default=Path("data/raw/GSE272159_37C_rep2.mapq_30.10.cool"))
    parser.add_argument("--published", type=Path, default=Path("results/task6_circular_20261006_coordinate_models.npz"))
    parser.add_argument("--published-protocol", type=Path, default=Path("results/task6_circular_20261006_protocol.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task6_followup_20261006"))
    parser.add_argument("--publish", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""RNG-controlled ablation of the frozen positive/zero ranking objective.

The coefficient, margin, epochs and model are unchanged after the first ranking
sensitivity. Only ranking minibatch draws move to an independent torch stream,
preserving historical positive-edge draws and initialization. This corrective
ablation is retrospective, not a fresh independent validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
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
from scripts.task6_zero_residual_followup import (
    ALPHA, EPOCHS, MARGIN, MATCH_SEED, PAIR_COUNT, ROOT, SEEDS,
    figure, freeze_protocol as parent_protocol, matched_ranking_pairs, ranking_loss,
)


RANKING_SEED_OFFSET = 7919


def sample_batches(positive_n: int, ranking_n: int, pair_count: int,
                   ranking_rng: torch.Generator):
    """Default positive stream matches the historical baseline exactly."""
    if min(positive_n, ranking_n, pair_count) < 1:
        raise ValueError("Need positive edge/ranking counts and batch budget")
    positive = torch.randint(positive_n, (min(pair_count, positive_n),))
    ranking = torch.randint(ranking_n, (min(pair_count, ranking_n),), generator=ranking_rng)
    return positive, ranking


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
    ranking_rng = torch.Generator().manual_seed(seed + RANKING_SEED_OFFSET)
    pi, pj, zi, zj = (torch.from_numpy(np.asarray(a, dtype=np.int64)) for a in matches[:4])
    history = []
    for epoch in range(epochs):
        model.train()
        xyz = model(inputs)
        take, ranking_take = sample_batches(len(i), len(pi), pair_count, ranking_rng)
        actual = torch.linalg.vector_norm(xyz[i[take]] - xyz[j[take]], dim=1)
        edge = torch.nn.functional.smooth_l1_loss(actual, goal[take])
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


def freeze_protocol(args) -> dict:
    protocol = parent_protocol(args)
    protocol["source_sha256"].update({p: sha(ROOT / p) for p in
        ("scripts/task6_zero_residual_controlled_followup.py", "tests/test_task6_zero_residual_controlled_followup.py")})
    protocol.update({"experiment": "task6-zero-residual-rng-controlled-20261006",
                     "rng_control": "Historical default torch RNG for initialization and positive minibatches; separate CPU torch.Generator for ranking draws, seed +7919. GenomeMLP has no stochastic layers. NumPy matching uses its independent fixed generator. Evaluation is deterministic with fixed matched-AUC seed.",
                     "ranking_generator_seed_offset": RANKING_SEED_OFFSET,
                     "unchanged_after_first_ranking_sensitivity": {"alpha": ALPHA, "margin": MARGIN,
                         "epochs": EPOCHS, "pairs_per_term_per_epoch": PAIR_COUNT, "seeds": list(SEEDS)},
                     "correction_context": "The preceding sensitivity has already read rep2. This isolates minibatch RNG, does not tune any model parameter or offer new independent validation.",
                     "original_ranking_protocol_sha256": sha(ROOT / "results/task6_followup_20261006_protocol.json"),
                     "caution": protocol["caution"] + " Corrective RNG ablation after first ranking sensitivity; not a new blind validation."})
    return protocol


def run(args) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(2)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a fresh empty output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = freeze_protocol(args)
    write_json(args.out / "protocol.json", protocol)
    print(f"Controlled protocol frozen {protocol['frozen_at_utc']}", flush=True)
    old = json.loads(args.published_protocol.read_text(encoding="utf-8"))
    if (old["rep1_sha256"] != protocol["input_sha256"]["rep1"] or
            old["rep2_sha256"] != protocol["input_sha256"]["rep2"] or
            old["module_sha256"] != sha(ROOT / "microc/circular.py")):
        raise ValueError("Published target/input comparator mismatch")
    with open_cool(args.rep1) as cool:
        chrom = cool.chromnames[0]
        genome_bp = int(cool.chromsizes[chrom])
        target1 = circular_contact_target(_coarsen(cool, chrom, 5000))
    matches, match_receipt = matched_ranking_pairs(target1)
    pd.DataFrame({"positive_i": matches[0], "positive_j": matches[1],
                  "zero_i": matches[2], "zero_j": matches[3],
                  "exact_ring_offset": matches[4]}).to_csv(args.out / "ranking_pairs.csv", index=False)
    write_json(args.out / "ranking_pair_receipt.json", {"selected_using": "rep1 only", "seed": MATCH_SEED,
        "total_matched_rank_pairs": len(matches[0]), "offsets": match_receipt,
        "ranking_pairs_sha256": sha(args.out / "ranking_pairs.csv")})
    with np.load(args.published) as saved:
        coordinates = {name: saved[name].copy() for name in saved.files}
    if any(xyz.shape != (len(target1), 3) or not np.isfinite(xyz).all()
           for xyz in coordinates.values()):
        raise ValueError("Published coordinate dimensions differ")
    histories = {}
    for seed in SEEDS:
        model, xyz, history = train(target1, matches, seed)
        coordinates[f"rank_mlp_seed{seed}"] = xyz
        histories[str(seed)] = history
        if seed == SEEDS[0]:
            torch.save({"model_state_dict": model.state_dict(), "seed": seed,
                        "bin_bp": 5000, "genome_bp": genome_bp,
                        "protocol_sha256": sha(args.out / "protocol.json")}, args.out / "first_seed.pt")
        print(f"Completed controlled fixed seed {seed}; {time.perf_counter() - started:.1f}s", flush=True)
    np.savez_compressed(args.out / "coordinate_models.npz", **coordinates)
    n, first = len(target1), coordinates[f"rank_mlp_seed{SEEDS[0]}"]
    pd.DataFrame({"chrom": chrom, "start": np.arange(n) * 5000,
                  "end": np.minimum((np.arange(n) + 1) * 5000, genome_bp),
                  "x": first[:, 0], "y": first[:, 1], "z": first[:, 2]}).to_csv(args.out / "coordinates.csv", index=False)
    write_json(args.out / "training_history.json", histories)
    freeze = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "rep2_cooler_opened_before_freeze": False,
              "historical_and_first_ranking_rep2_inspection": True,
              "output_sha256": {p: sha(args.out / p) for p in
                ("protocol.json", "ranking_pairs.csv", "ranking_pair_receipt.json",
                 "coordinate_models.npz", "coordinates.csv", "first_seed.pt", "training_history.json")}}
    write_json(args.out / "coordinate_freeze.json", freeze)
    print(f"Controlled coordinates frozen {freeze['frozen_at_utc']}; opening rep2", flush=True)
    with open_cool(args.rep2) as cool:
        if cool.chromnames[0] != chrom or int(cool.chromsizes[chrom]) != genome_bp:
            raise ValueError("Replicate dimensions differ")
        target2 = circular_contact_target(_coarsen(cool, chrom, 5000))
    rep1_metrics, rep2_metrics = evaluate(target1, coordinates), evaluate(target2, coordinates)
    rows = [{"model": name, "rep2_exact_offset_matched_auc":
               rep2_metrics["exact_circular_offset_matched_auc"]["auc"].get(name),
             **{f"rep2_{s}_rho": rep2_metrics["strata"][s][name] for s in rep2_metrics["strata"]}}
            for name in coordinates]
    pd.DataFrame(rows).to_csv(args.out / "comparison.csv", index=False)
    figure(rep2_metrics, args.out / "comparison.png")
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
               "conclusion_rule": "Retain every seed/baseline and rho stratum. RNG correction only; no parameter selection or newly independent validation."}
    write_json(args.out / "summary.json", summary)
    if args.publish:
        files = ("protocol.json", "summary.json", "coordinate_freeze.json", "ranking_pair_receipt.json",
                 "coordinate_models.npz", "coordinates.csv", "first_seed.pt", "training_history.json",
                 "comparison.csv", "comparison.png")
        prefix = "task6_controlled_followup_20261006_"
        if any((ROOT / "results" / (prefix + p)).exists() for p in files):
            raise ValueError("Published controlled files already exist")
        for p in files:
            shutil.copyfile(args.out / p, ROOT / "results" / (prefix + p))
        write_json(ROOT / "results" / (prefix + "evidence_index.json"), {
            "published_at_utc": datetime.now(timezone.utc).isoformat(), "output_directory": str(args.out),
            "source_sha256": protocol["source_sha256"], "input_sha256": protocol["input_sha256"],
            "files": {prefix + p: sha(ROOT / "results" / (prefix + p)) for p in files},
            "command": "python -m scripts.task6_zero_residual_controlled_followup --publish",
            "local_ranking_pairs_sha256": sha(args.out / "ranking_pairs.csv")})
    print(json.dumps(rows, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep1", type=Path, default=Path("data/raw/GSE272159_37C_rep1.mapq_30.10.cool"))
    parser.add_argument("--rep2", type=Path, default=Path("data/raw/GSE272159_37C_rep2.mapq_30.10.cool"))
    parser.add_argument("--published", type=Path, default=Path("results/task6_circular_20261006_coordinate_models.npz"))
    parser.add_argument("--published-protocol", type=Path, default=Path("results/task6_circular_20261006_protocol.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/task6_controlled_followup_20261006"))
    parser.add_argument("--publish", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

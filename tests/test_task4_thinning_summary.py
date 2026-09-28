import json

import pandas as pd

from scripts.summarize_task4_thinning import SEEDS, main, output_dir, summarize_condition


def test_summary_accepts_an_additional_condition(tmp_path, monkeypatch):
    condition = "hns"
    rows = {
        SEEDS[0]: [("A", "CHIN", 10, 30, "weakened"),
                   ("B", "CHID", 20, 40, "strengthened"),
                   ("D", "OPCID", 100, 120, "weakened")],
        SEEDS[1]: [("A", "CHIN", 10, 30, "weakened"),
                   ("B", "CHID", 20, 40, "strengthened"),
                   ("C", "OPCID", 150, 170, "weakened"),
                   ("D", "OPCID", 100, 120, "strengthened")],
        SEEDS[2]: [("A", "CHIN", 10, 30, "weakened"),
                   ("B", "CHID", 20, 40, "strengthened"),
                   ("D", "OPCID", 100, 120, "weakened")],
    }
    for seed, records in rows.items():
        directory = output_dir(tmp_path, condition, seed)
        directory.mkdir(parents=True)
        pd.DataFrame(records, columns=["id", "type", "start", "end", "direction"]).to_csv(
            directory / "reproducible_changes.csv", index=False)

    summary = summarize_condition(tmp_path, condition)
    assert summary["all_seed_intersection_count"] == 3
    assert summary["all_seed_sign_flip_count"] == 1
    assert summary["all_seed_sign_consistent_count"] == 2
    assert summary["all_seed_sign_consistent_merged_loci"] == 1

    monkeypatch.setattr("sys.argv", ["summarize_task4_thinning", "--root", str(tmp_path),
                                "--out", "summary.json", "--conditions", condition])
    main()
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert list(saved) == [condition]
    assert pd.read_csv(tmp_path / "task4_hns_stable_changes.csv").id.tolist() == ["A", "B"]

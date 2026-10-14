import pandas as pd

from scripts import audit_final_experiments_20261014 as audit


def test_published_summary_receipt_path_schema(tmp_path):
    assert audit.task2_check_path(tmp_path, {"kind": "summary_copy", "path": "summary.json"}) == tmp_path / "results/summary.json"
    assert audit.task2_check_path(tmp_path, {"kind": "output", "path": "outputs/run/summary.json"}) == tmp_path / "outputs/run/summary.json"


def test_stronger_controls_use_identical_ids_and_gain_direction(tmp_path, monkeypatch):
    folder = tmp_path / "results"
    folder.mkdir()
    methods = ["adapted_cnn_fraction_1", "bilinear", "bicubic", "train_calibrated_bilinear", "train_calibrated_bicubic"]
    rows = []
    for method in methods:
        for number in range(68):
            selected = method == methods[0]
            row = {"id": f"record_{number:03}", "method": method, "overlap_component": number // 2}
            row.update({metric: 5. if selected else 4. for metric in ("psnr", "ssim", "shape_r", "profile_r", "top5_f1")})
            row["peak_error_bp"] = 2. if selected else 3.
            rows.append(row)
    for replicate in ("rep1_test", "rep2_fixed_test"):
        pd.DataFrame(rows).to_csv(folder / f"task5_full_structure_20261014_final_{replicate}.csv", index=False)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    result = audit.strong_controls()
    assert len(result) == 48
    assert result.n_structures.eq(68).all()
    assert result.gain_mean.eq(1.).all()
    assert result.gain_component_95ci_lower.eq(1.).all()
    assert result.source_csv_sha256.str.len().eq(64).all()

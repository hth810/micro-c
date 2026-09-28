from pathlib import Path

import pandas as pd
import pytest

from scripts.build_annotation_review import prepare_cards, render_html


def test_review_cards_keep_human_decision_separate_from_cnn(tmp_path: Path) -> None:
    evidence = pd.DataFrame([{"chrom": "NC_000913.3", "start": 3_304_960,
                              "end": 3_310_080, "representative_center_bp": 3_307_520,
                              "wt_replicate_r": .53, "known_distance_bp": 13_010,
                              "max_candidate_score": .97}])
    suggestions = pd.DataFrame([{"center_bp": 3_307_520, "suggested_known_shape": "OPCID",
                                 "shape_score_CHIN": .3, "shape_score_OPCID": .5,
                                 "shape_score_CHID": .2}])
    for replicate in ("rep1", "rep2"):
        (tmp_path / f"task2_strict_individual_3307520_{replicate}.png").write_bytes(b"png")
    cards = prepare_cards(evidence, tmp_path, suggestions)
    assert len(cards) == 1
    assert cards[0]["putative_n1"]
    assert cards[0]["suggestion"]["class"] == "OPCID"
    assert cards[0]["images"]["rep1"].startswith("data:image/png;base64,")
    page = render_html(cards)
    assert "人工判定" in page and "导出人工复核 JSON" in page
    assert "非新颖性概率" in page
    assert "artifact" in page


def test_review_rejects_duplicate_loci(tmp_path: Path) -> None:
    evidence = pd.DataFrame([{"chrom": "NC_000913.3", "start": 1, "end": 2,
                              "representative_center_bp": 3, "wt_replicate_r": .5,
                              "known_distance_bp": 50, "max_candidate_score": .9}] * 2)
    with pytest.raises(ValueError, match="independent"):
        prepare_cards(evidence, tmp_path, None)

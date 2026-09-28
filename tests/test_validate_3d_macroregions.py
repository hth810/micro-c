import numpy as np

from scripts.validate_3d_macroregions import core_mask, offset_effect, offset_matched_auc


def test_offset_matched_region_signal_detects_geometry_not_genome_distance():
    n = 48
    labels = np.repeat(np.arange(6), 8)
    i, j = np.indices((n, n))
    separation = np.minimum(abs(i - j), n - abs(i - j))
    region_geometry = separation * .01 + (labels[i] != labels[j]).astype(float)
    genome_only = separation.astype(float)
    core = np.ones(n, dtype=bool)

    result = offset_matched_auc(labels, core, region_geometry, seed=7)
    control = offset_matched_auc(labels, core, genome_only, seed=7)
    assert result["same_region_vs_different_auc"] == 1.0
    assert result["matched_pairs_per_class"] > 0
    assert control["same_region_vs_different_auc"] == .5

    upper_i, upper_j = np.triu_indices(n, k=2)
    effect = offset_effect(labels, core, (upper_i, upper_j),
                           separation[upper_i, upper_j],
                           np.array([region_geometry[upper_i, upper_j],
                                     genome_only[upper_i, upper_j]]))
    assert effect[0] > .5
    assert abs(effect[1]) < 1e-12


def test_core_mask_excludes_both_sides_of_circular_boundary():
    centers = np.array([5_000, 55_000, 595_000, 605_000, 700_000, 4_640_000])
    selected = core_mask(centers, trim_bp=50_000, genome_bp=4_645_000)
    assert selected.tolist() == [False, True, False, False, True, False]

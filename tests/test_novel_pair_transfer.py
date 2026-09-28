import numpy as np

from scripts.audit_novel_pair_transfer import independent_pairs, similarity_percentile


def test_independent_pairs_excludes_overlapping_windows():
    mask = independent_pairs([10000, 15000, 50000])
    assert mask.sum() == 2
    assert not mask[0, 1]
    assert mask[0, 2] and mask[1, 2]


def test_similarity_percentile_ties_do_not_count_as_smaller():
    assert similarity_percentile(.5, np.array([.1, .5, .8])) == 1 / 3

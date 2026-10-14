import numpy as np
import pytest

from scripts.task6_literature_regions_20261014 import REGIONS, labels_and_core


def test_published_ori_wrap_and_half_open_boundaries():
    x = np.array([0, 9900, 49999, 50000, 599999, 600000, 1209999, 1210000, 3759999, 3760000, 4641651])
    labels, core = labels_and_core(x, 4641652, 0)
    assert labels.tolist() == ["Ori", "Ori", "Ori", "NSR", "NSR", "Right", "Right", "Ter", "NSL", "Ori", "Ori"]
    assert core.all()


def test_zero_not_artificial_boundary_in_ori():
    labels, core = labels_and_core(np.array([0, 2500, 50000, 100000, 3760000, 3810000, 4639000]), 4641652, 50000)
    assert labels.tolist() == ["Ori", "Ori", "NSR", "NSR", "Ori", "Ori", "Ori"]
    assert core.tolist() == [True, False, False, True, False, True, True]


def test_circular_trim_wraps_across_zero():
    regions = (("A", 950, 100), ("B", 100, 950))
    _, core = labels_and_core(np.array([0, 9, 10, 90, 100, 110, 940, 950, 960, 999]), 1000, 10, regions)
    assert core.tolist() == [True, True, True, True, False, True, True, False, True, True]


@pytest.mark.parametrize("positions,genome,trim", [(np.array([-1]), 1000, 0), (np.array([1000]), 1000, 0), (np.array([0]), 1000, -1)])
def test_invalid_position_or_trim(positions, genome, trim):
    with pytest.raises(ValueError):
        labels_and_core(positions, genome, trim)


def test_partition_rejects_overlap_and_unassigned():
    with pytest.raises(ValueError):
        labels_and_core(np.array([10, 800]), 1000, 0, (("A", 0, 500), ("B", 250, 900)))
    with pytest.raises(ValueError):
        labels_and_core(np.array([10, 950]), 1000, 0, (("A", 0, 500), ("B", 500, 900)))


def test_frozen_whole_coordinate_version():
    assert REGIONS == (("Ori", 3760000, 50000), ("NSR", 50000, 600000), ("Right", 600000, 1210000),
                       ("Ter", 1210000, 2180000), ("Left", 2180000, 2880000), ("NSL", 2880000, 3760000))

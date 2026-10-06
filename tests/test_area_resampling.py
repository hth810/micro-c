import numpy as np
import pytest

from microc.data import area_resample


def test_sparse_counts_survive_integer_pooling_at_all_native_positions():
    raw = np.zeros((16, 16))
    raw[3, 11] = raw[11, 3] = 7
    pooled = area_resample(raw, 4)
    assert pooled[0, 2] == pooled[2, 0] == 7
    assert pooled.sum() == raw.sum()


@pytest.mark.parametrize("native,output", [(13, 5), (5, 13), (8, 4), (7, 7)])
def test_fractional_area_boundaries_preserve_counts_and_symmetry(native, output):
    rng = np.random.default_rng(6)
    raw = rng.uniform(0, 10, (native, native))
    raw = (raw + raw.T) / 2
    result = area_resample(raw, output)
    assert result.shape == (output, output)
    assert np.allclose(result, result.T)
    assert np.isclose(result.sum(), raw.sum(), rtol=1e-12)


def test_area_pooling_rejects_invalid_measurements():
    for raw, output in ((np.ones((2, 3)), 2), (np.full((3, 3), np.nan), 2),
                        (np.ones((3, 3)), 0), (np.ones((3, 3)), True)):
        with pytest.raises(ValueError):
            area_resample(raw, output)

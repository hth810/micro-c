import numpy as np
import pytest

from scripts.audit_3d_seed_stability import distance_agreement, train_coordinates


def test_seed_audit_trains_finite_3d_coordinates_without_mds():
    contact = np.zeros((12, 12), np.float32)
    contact[np.triu_indices(12, k=2)] = .5
    contact += contact.T
    first = train_coordinates(contact, seed=12, epochs=2, pairs_per_epoch=25)
    again = train_coordinates(contact, seed=12, epochs=2, pairs_per_epoch=25)
    assert first.shape == (12, 3)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, again)
    assert distance_agreement({"first": first, "again": again})["first__again"] == pytest.approx(1)


def test_seed_audit_rejects_invalid_contact():
    with pytest.raises(ValueError, match="square"):
        train_coordinates(np.ones((3, 4)), seed=1, epochs=1)

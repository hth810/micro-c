import numpy as np
import pytest
from scripts.run_improvement_trials import ContextSR, block_gain, sr_metrics
import torch


def test_context_control_starts_at_same_interpolation_baseline():
    model = ContextSR().eval()
    x = torch.randn(2, 2, 8, 8)
    x[:, 0] = (x[:, 0] + x[:, 0].transpose(-1, -2)) / 2
    with torch.no_grad():
        wide = model(x)
        control_x = x.clone()
        control_x[:, 1] = 0
        control = model(control_x)
    assert torch.equal(wide, control)
    assert torch.equal(wide[:, 0], x[:, 0])


def test_context_network_preserves_symmetry_after_nonzero_residual():
    model = ContextSR().eval()
    torch.nn.init.normal_(model.tail.weight)
    x = torch.randn(2, 2, 8, 8)
    with torch.no_grad():
        output = model(x)
    assert torch.equal(output, output.transpose(-1, -2))


def test_no_observed_local_signal_does_not_hallucinate():
    model = ContextSR().eval()
    torch.nn.init.normal_(model.tail.weight)
    x = torch.ones(1, 2, 8, 8)
    x[:, 0] = 0
    with torch.no_grad():
        assert torch.count_nonzero(model(x)) == 0


def test_constant_paired_gain_has_constant_block_interval():
    result = block_gain(np.zeros(4), np.ones(4), np.array([0, 1, 100000, 100001]), 7)
    assert result["difference"] == 1
    assert result["ci95"] == [1, 1]
    assert result["genomic_blocks"] == 2


def test_saved_sr_image_metrics_are_finite_and_exclude_zero_rank_signal():
    image = np.zeros((1, 64, 64), np.float32)
    with pytest.warns(RuntimeWarning):
        detail, score = sr_metrics(image, image)
    assert detail[0]["rho"] is None
    assert score["psnr"] == 100
    assert score["ssim"] == 1

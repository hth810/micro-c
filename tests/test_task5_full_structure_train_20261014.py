import numpy as np
import torch

from microc.deep import SuperResolutionCNN
from scripts.task5_full_structure_train_20261014 import FullWindowCNN, final_image, ssim_loss


def test_structural_loss_prefers_identical_image():
    torch.manual_seed(4)
    x = torch.randn(2,1,64,64)
    assert abs(float(ssim_loss(x,x))) < 1e-6
    assert float(ssim_loss(torch.zeros_like(x),x)) > .5


def test_full_final_zero_input_remains_zero_after_output_bias():
    base = SuperResolutionCNN(channels=8,blocks=4).eval()
    model = FullWindowCNN(base,2.,.7).eval()
    assert np.count_nonzero(final_image(model,np.zeros((96,96),np.float32))) == 0


def test_output_scale_is_trainable_and_positive():
    model = FullWindowCNN(SuperResolutionCNN(channels=8,blocks=1),.5,.2)
    out = model(torch.ones(1,1,96,96))
    out.mean().backward()
    assert model.log_scale.grad is not None
    assert float(model.log_scale.detach().exp()) > 0

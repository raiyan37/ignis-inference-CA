import torch

from ignisca.inference.mc_dropout import mc_dropout_predict
from ignisca.models.resunet import ResUNet


def test_mc_dropout_zero_dropout_has_zero_variance(synthetic_batch):
    x, _ = synthetic_batch
    model = ResUNet(in_channels=12, base=4, dropout=0.0)
    mean, var = mc_dropout_predict(model, x, n_samples=5)
    assert mean.shape == (4, 1, 32, 32)
    assert var.shape == (4, 1, 32, 32)
    assert torch.allclose(var, torch.zeros_like(var), atol=1e-8)


def test_mc_dropout_nonzero_dropout_has_positive_variance(tiny_resunet, synthetic_batch):
    x, _ = synthetic_batch
    _, var = mc_dropout_predict(tiny_resunet, x, n_samples=8)
    assert var.max().item() > 0.0


def test_mc_dropout_restores_eval_mode(tiny_resunet, synthetic_batch):
    x, _ = synthetic_batch
    tiny_resunet.eval()
    _ = mc_dropout_predict(tiny_resunet, x, n_samples=3)
    assert tiny_resunet.training is False
    for m in tiny_resunet.modules():
        if isinstance(m, torch.nn.Dropout2d):
            assert m.training is False


def test_mc_dropout_leaves_groupnorm_untouched(tiny_resunet, synthetic_batch):
    x, _ = synthetic_batch
    gn_modules = [m for m in tiny_resunet.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert len(gn_modules) > 0, "ResUNet should contain GroupNorm layers"
    for m in gn_modules:
        m.train(False)
    _ = mc_dropout_predict(tiny_resunet, x, n_samples=3)
    for m in gn_modules:
        assert m.training is False

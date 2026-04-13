import torch

from ignisca.models.resunet import ResUNet


def test_resunet_forward_shape_matches_input():
    model = ResUNet(in_channels=12, base=16, dropout=0.2)
    x = torch.randn(2, 12, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_resunet_backward_produces_gradients():
    model = ResUNet(in_channels=12, base=16, dropout=0.2)
    x = torch.randn(2, 12, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"missing grad for {name}"


def test_resunet_dropout_active_when_train_mode_set():
    """MC Dropout requires dropout to remain stochastic when the module is in train mode."""
    torch.manual_seed(0)
    model = ResUNet(in_channels=12, base=16, dropout=0.5).train()
    x = torch.randn(1, 12, 32, 32)
    out_a = model(x)
    out_b = model(x)
    assert not torch.allclose(out_a, out_b)

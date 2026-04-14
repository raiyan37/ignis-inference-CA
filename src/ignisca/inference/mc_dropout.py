from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MC Dropout inference: n forward passes with Dropout2d enabled.

    GroupNorm and other normalization layers stay in eval mode — flipping the
    whole model to ``.train()`` would contaminate normalization statistics
    across MC samples. Only ``nn.Dropout2d`` submodules are toggled.

    Returns
    -------
    mean : Tensor
        (B, 1, H, W) element-wise mean of ``sigmoid(model(x))`` over n samples.
    var : Tensor
        (B, 1, H, W) element-wise population variance over the same samples.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train(True)
    try:
        with torch.no_grad():
            samples = torch.stack(
                [torch.sigmoid(model(x)) for _ in range(n_samples)], dim=0
            )
        mean = samples.mean(dim=0)
        var = samples.var(dim=0, unbiased=False)
    finally:
        model.eval()
    return mean, var

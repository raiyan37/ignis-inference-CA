"""Synthetic in-memory Dataset for evaluation/runner tests.

Not a test file — no ``test_`` prefix so pytest does not collect it.
Yields 8 samples total, meant to exercise slice classification: half the
samples have Santa Ana winds, half have SW-origin winds; half have small
input masks (early), half have large ones (mature).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset


class FakeFireDataset(Dataset):
    def __init__(self, n_samples: int = 8, hw: int = 32) -> None:
        torch.manual_seed(42)
        self.n = n_samples
        self.hw = hw
        self._x = torch.zeros(n_samples, 12, hw, hw)
        self._y = torch.zeros(n_samples, hw, hw)
        for i in range(n_samples):
            if i % 2 == 0:
                # Santa Ana: wind flowing SW, coming FROM NE
                self._x[i, 7] = -7.5
                self._x[i, 8] = -7.5
            else:
                # Not Santa Ana: wind flowing NE, coming FROM SW
                self._x[i, 7] = 7.5
                self._x[i, 8] = 7.5

            if i < n_samples // 2:
                # Early fire — 2 positive pixels
                self._x[i, 0, 0, 0] = 1.0
                self._x[i, 0, 0, 1] = 1.0
                self._y[i, 0, 0:3] = 1.0
            else:
                # Mature fire — 100 positive pixels
                self._x[i, 0, :10, :10] = 1.0
                self._y[i, :10, :11] = 1.0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx]

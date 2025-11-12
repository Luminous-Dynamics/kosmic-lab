from __future__ import annotations

import numpy as np

from core.te_bridge import gaussian_transfer_entropy


def test_gaussian_te_directionality() -> None:
    rng = np.random.default_rng(0)
    n = 500
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.normal(scale=0.1)
        y[t] = 0.5 * y[t - 1] + 0.4 * x[t - 1] + rng.normal(scale=0.1)
    te_forward = gaussian_transfer_entropy(x[:-1], y[:-1], y[1:])
    te_shuffled = gaussian_transfer_entropy(rng.permutation(x[:-1]), y[:-1], y[1:])
    assert te_forward > te_shuffled

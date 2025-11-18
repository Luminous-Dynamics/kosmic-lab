from __future__ import annotations

import numpy as np

from core.bioelectric import BioelectricGrid, compute_iou, mask_from_voltage


def test_iou_fraction() -> None:
    target = np.array([[True, False], [False, True]])
    current = np.array([[True, True], [False, False]])
    assert abs(compute_iou(target, current) - (1 / 3)) < 1e-9


def test_diffusion_spread() -> None:
    grid = BioelectricGrid((16, 16), D=0.2, g=0.0, dt=0.1)
    # Use biological voltage scale (-100 to 0 mV)
    grid.V[8, 8] = -50.0  # Depolarized voltage
    for _ in range(20):
        grid.step()
    # Lower threshold appropriate for biological scale
    mask = mask_from_voltage(grid.V, threshold=5.0)
    # Should spread beyond the original position
    assert mask.sum() > 1


from __future__ import annotations

import numpy as np

from core.reciprocity_coupling import (
    CouplerConfig,
    PolicySnapshot,
    ReciprocityCoupler,
)


def make_snapshot(value: float, energy: float = 0.0) -> PolicySnapshot:
    grad = np.array([value], dtype=float)
    params = np.array([0.0], dtype=float)
    return PolicySnapshot(params={"actor": params.copy()}, grads={"actor": grad}, energy=energy)


def test_reciprocity_positive_coupling() -> None:
    cfg = CouplerConfig(window=10, eta_min=0.0, eta_max=0.05, te_weight=1.0, k_neighbors=3)
    coupler = ReciprocityCoupler(cfg)

    prev_g1 = 0.0
    prev_g2 = 0.0
    for t in range(12):
        g1 = np.sin(0.2 * t)
        g2 = 0.4 * prev_g1 + 0.5 * prev_g2
        snap1 = make_snapshot(g1)
        snap2 = make_snapshot(g2)
        if t == 0:
            coupler.register_universe("U1", snap1)
            coupler.register_universe("U2", snap2)
        else:
            coupler.update_snapshot("U1", snap1)
            coupler.update_snapshot("U2", snap2)
        prev_g1, prev_g2 = g1, g2

    deltas = coupler.compute_coupling()
    assert deltas["U2"]["actor"].sum() != 0.0


def test_energy_gate_blocks_updates() -> None:
    cfg = CouplerConfig(window=5, eta_min=0.0, eta_max=0.1)
    coupler = ReciprocityCoupler(cfg)
    for t in range(6):
        snap_high = make_snapshot(0.1 * t, energy=1.0)
        snap_low = make_snapshot(0.2 * t, energy=0.0)
        if t == 0:
            coupler.register_universe("A", snap_high)
            coupler.register_universe("B", snap_low)
        else:
            coupler.update_snapshot("A", snap_high)
            coupler.update_snapshot("B", snap_low)

    deltas = coupler.compute_coupling()
    assert np.allclose(deltas["A"]["actor"], 0.0)


def test_mismatched_shapes_raise_error() -> None:
    cfg = CouplerConfig(window=5)
    coupler = ReciprocityCoupler(cfg)
    snap1 = PolicySnapshot(
        params={"actor": np.zeros((2,))}, grads={"actor": np.ones((2,))}, energy=0.0
    )
    snap2 = PolicySnapshot(
        params={"actor": np.zeros((3,))}, grads={"actor": np.ones((3,))}, energy=0.0
    )
    coupler.register_universe("X", snap1)
    coupler.register_universe("Y", snap2)
    coupler.update_snapshot("X", snap1)
    coupler.update_snapshot("Y", snap2)
    try:
        coupler.compute_coupling()
    except ValueError as exc:
        assert "Gradient shapes differ" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched shapes")

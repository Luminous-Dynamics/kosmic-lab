# Tests for bioelectric rescue helpers.
from __future__ import annotations

import numpy as np
import pytest

from core.bioelectric import compute_iou
from fre import rescue


class DummyAgent:
    def __init__(self) -> None:
        self.prediction_errors = {"sensory": 0.8}
        self.voltage = -70.0
        self.gap_junctions = {"n1": 1.0, "n2": 0.5}
        self.boundary_integrity = 0.3
        self.internal_state = {"membrane": 0.2, "ATP": 1.0}


def test_fep_to_bioelectric_trigger() -> None:
    agent = DummyAgent()
    rescue.fep_to_bioelectric(agent, timestep=0)
    assert agent.voltage == -62.0
    assert agent.gap_junctions["n1"] == 1.1


def test_bioelectric_to_autopoiesis() -> None:
    agent = DummyAgent()
    agent.voltage = -70.0
    rescue.bioelectric_to_autopoiesis(agent, {"voltage": -70.0})
    assert pytest.approx(agent.boundary_integrity, rel=1e-3) == 0.3035
    assert pytest.approx(agent.internal_state["ATP"], rel=1e-3) == 0.9993


def test_compute_iou() -> None:
    current = np.zeros((2, 2, 2), dtype=bool)
    target = np.zeros_like(current)
    current[0, 0, 0] = True
    target[0, 0, 0] = True
    target[1, 1, 1] = True
    iou = compute_iou(target, current)
    assert iou == 0.5

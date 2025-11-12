from __future__ import annotations

import threading
import time

import numpy as np

from core.shared_buffer import Experience, SharedReplayBuffer


def make_exp(val: float, state_dim: int = 3, action_dim: int = 2) -> Experience:
    state = np.full(state_dim, val, dtype=np.float32)
    next_state = state + 0.1
    action = np.full(action_dim, val * 0.5, dtype=np.float32)
    return Experience(state=state, action=action, reward=float(val), next_state=next_state, done=False, universe_id=int(val))


def test_push_and_len() -> None:
    buf = SharedReplayBuffer(capacity=5, seed=0)
    for i in range(3):
        buf.push(make_exp(float(i)))
    assert len(buf) == 3


def test_overwrite_when_full() -> None:
    buf = SharedReplayBuffer(capacity=3, seed=0)
    for i in range(5):
        buf.push(make_exp(float(i)))
    assert len(buf) == 3
    states, _, _, _, _, ids = buf.sample(batch_size=3)
    # Only last three inserts remain
    assert set(ids.tolist()) == {2, 3, 4}
    assert states.shape == (3, 3)


def test_sample_requires_enough_entries() -> None:
    buf = SharedReplayBuffer(capacity=10, seed=0)
    buf.push(make_exp(0.0))
    try:
        buf.sample(2)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when sampling too many experiences")


def test_sample_returns_arrays() -> None:
    buf = SharedReplayBuffer(capacity=10, seed=123)
    for i in range(6):
        buf.push(make_exp(float(i)))
    states, actions, rewards, next_states, dones, ids = buf.sample(4)
    assert states.shape == (4, 3)
    assert actions.shape == (4, 2)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 3)
    assert dones.shape == (4,)
    assert ids.shape == (4,)


def test_sample_thread_safe_under_concurrency() -> None:
    buffer = SharedReplayBuffer(capacity=64, seed=7)
    stop_event = threading.Event()
    errors: list[Exception] = []

    def producer() -> None:
        val = 0.0
        while not stop_event.is_set():
            buffer.push(make_exp(val))
            val = (val + 1.0) % 5.0
            time.sleep(0.001)

    def consumer() -> None:
        while not stop_event.is_set():
            if len(buffer) < 5:
                time.sleep(0.001)
                continue
            try:
                batch = buffer.sample(5)
                states = batch[0]
                assert states.shape[0] == 5
            except ValueError:
                continue
            except Exception as exc:  # pragma: no cover - diagnostic
                errors.append(exc)
                stop_event.set()

    prod = threading.Thread(target=producer, daemon=True)
    cons = threading.Thread(target=consumer, daemon=True)
    prod.start()
    cons.start()

    time.sleep(0.1)
    stop_event.set()
    prod.join(timeout=1.0)
    cons.join(timeout=1.0)

    assert not errors

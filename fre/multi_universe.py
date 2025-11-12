"""Multi-universe simulator with adaptive coupling strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import SAC

from core.reciprocity_coupling import CouplerConfig, PolicySnapshot, ReciprocityCoupler
from core.te_bridge import knn_transfer_entropy
from fre.universe import UniverseSimulator
from harmonics_module import HarmonyCalculator, HarmonyScores
from fre.phase4_coordinator import BufferConfig, CriticConfig, Phase4Coordinator, TrainerConfig
from core.phase4_actor import ActorConfig, Phase4Actor


@dataclass
class UniverseState:
    simulator: UniverseSimulator
    seed: int
    params: Dict[str, float]
    calculator: HarmonyCalculator
    uid: str
    history: List[HarmonyScores] = field(default_factory=list)
    ema_metrics: Optional[np.ndarray] = None
    last_grad: Optional[np.ndarray] = None
    prev_grad: Optional[np.ndarray] = None
    last_snapshot: Optional[PolicySnapshot] = None
    last_coherence: Optional[float] = None
    last_state_summary: Optional[np.ndarray] = None
    last_action: Optional[np.ndarray] = None
    last_k: Optional[float] = None
    actor: Optional[Phase4Actor] = None


class MultiUniverseSimulator:
    """Couples universes via TE bridges or reciprocal policy coupling."""

    def __init__(
        self,
        universe_count: int,
        base_params: Dict[str, float],
        param_bounds: Dict[str, Tuple[float, float]],
        seeds: List[int],
        *,
        coupling_strength: float = 0.1,
        harmonics_calculator: HarmonyCalculator | None = None,
        coupling_mode: str = "rpc",
        te_config: Dict[str, float] | None = None,
        rpc_config: Dict[str, float] | None = None,
        phase4_config: Dict[str, Dict] | None = None,
    ) -> None:
        assert len(seeds) == universe_count, "Need one seed per universe"
        calc = harmonics_calculator or HarmonyCalculator()
        self.universes = [
            UniverseState(
                simulator=UniverseSimulator(),
                seed=seeds[i],
                params=base_params.copy(),
                calculator=HarmonyCalculator(
                    baseline_phi=calc.baseline_phi,
                    baseline_fe=calc.baseline_fe,
                ),
                uid=f"U{i}",
            )
            for i in range(universe_count)
        ]
        self.param_bounds = param_bounds
        self.coupling_strength = coupling_strength
        self.coupling_mode = coupling_mode.lower()
        self.uid_to_index = {state.uid: idx for idx, state in enumerate(self.universes)}

        self.driver_history: Optional[List[List[np.ndarray]]]
        self.driven_history: Optional[List[List[np.ndarray]]]
        self.weights_matrix: Optional[np.ndarray]
        self.te_cfg: Optional[Dict[str, float]]
        self.rpc_coupler: Optional[ReciprocityCoupler] = None

        self.rpc_ema_alpha = None
        self.rpc_diffusion_strength = 0.0

        self.rpc_policy_model: Optional[SAC] = None
        self.phase4: Optional[Phase4Coordinator] = None
        self.phase4_last_loss: Optional[float] = None
        self.phase4_loss_log: List[Tuple[int, float]] = []
        self.actor_config: Optional[ActorConfig] = None

        if self.coupling_mode == "te":
            self.driver_history = [list() for _ in range(universe_count)]
            self.driven_history = [list() for _ in range(universe_count)]
            self.weights_matrix = np.zeros((universe_count, universe_count))
            self.te_cfg = {
                "window": 64,
                "update_interval": 4,
                "symmetry_lambda": 0.1,
                "epsilon": 1e-6,
                "knn_k": 5,
            }
            if te_config:
                self.te_cfg.update(te_config)
        elif self.coupling_mode == "rpc":
            self.driver_history = None
            self.driven_history = None
            self.weights_matrix = None
            self.te_cfg = None
            rpc_kwargs = {}
            if rpc_config:
                allowed = set(CouplerConfig.__annotations__.keys())
                rpc_kwargs = {k: v for k, v in rpc_config.items() if k in allowed}
                self.rpc_ema_alpha = float(rpc_config.get("ema_alpha", 0.2))
                self.rpc_diffusion_strength = float(rpc_config.get("diffusion_strength", 0.01))
            else:
                self.rpc_ema_alpha = 0.2
                self.rpc_diffusion_strength = 0.01
            self.rpc_coupler = ReciprocityCoupler(CouplerConfig(**rpc_kwargs))
            policy_path = rpc_config.get("policy_path") if rpc_config else None
            if policy_path:
                self.rpc_policy_model = SAC.load(policy_path)
        else:
            raise ValueError(f"Unsupported coupling mode: {coupling_mode}")

        if phase4_config and phase4_config.get("enable", False):
            buffer_cfg_dict = phase4_config.get("buffer", {})
            critic_cfg_dict = phase4_config.get("critic", {})
            trainer_cfg_dict = phase4_config.get("trainer", {})
            buffer_cfg = BufferConfig(
                capacity=int(buffer_cfg_dict.get("capacity", 200_000)),
                batch_size=int(buffer_cfg_dict.get("batch_size", 512)),
                seed=buffer_cfg_dict.get("seed"),
            )
            state_dim = int(critic_cfg_dict.get("state_dim", 6))
            critic_cfg = CriticConfig(
                state_dim=state_dim,
                learning_rate=float(critic_cfg_dict.get("learning_rate", 3e-4)),
                gamma=float(critic_cfg_dict.get("gamma", 0.99)),
                weight_decay=float(critic_cfg_dict.get("weight_decay", 0.0)),
            )
            trainer_cfg = TrainerConfig(
                warmup_steps=int(trainer_cfg_dict.get("warmup_steps", 1_000)),
                critic_updates_per_step=int(trainer_cfg_dict.get("critic_updates_per_step", 4)),
                log_interval=int(trainer_cfg_dict.get("log_interval", 100)),
            )
            self.phase4 = Phase4Coordinator(buffer_cfg, critic_cfg, trainer_cfg)
            actor_cfg_dict = phase4_config.get("actor", {})
            self.actor_config = ActorConfig(
                state_dim=int(actor_cfg_dict.get("state_dim", state_dim)),
                action_dim=int(actor_cfg_dict.get("action_dim", 2)),
                learning_rate=float(actor_cfg_dict.get("learning_rate", 1e-3)),
            )
            for uni in self.universes:
                uni.actor = Phase4Actor(self.actor_config)

    def step(self, timestep: int, weights: np.ndarray) -> Dict[str, HarmonyScores]:
        harmonies: Dict[str, HarmonyScores] = {}

        for idx, uni in enumerate(self.universes):
            metrics = uni.simulator.run(uni.params, uni.seed + timestep)
            inputs = self._to_harmony_inputs(metrics, uni.seed, timestep)
            scores = uni.calculator.compute_all(**inputs)
            uni.history.append(scores)
            uid = uni.uid
            harmonies[uid] = scores

            if self.coupling_mode == "te" and self.driver_history is not None and self.driven_history is not None:
                driver_value = np.array(
                    [scores.resonant_coherence, scores.infinite_play], dtype=float
                )
                self.driver_history[idx].append(driver_value)
                driven_value = np.array(
                    [
                        uni.params.get("energy_gradient", 0.0),
                        uni.params.get("plasticity_rate", 0.0),
                    ],
                    dtype=float,
                )
                self.driven_history[idx].append(driven_value)

            if self.coupling_mode == "rpc" and self.rpc_coupler is not None:
                prev_state = uni.last_state_summary
                prev_action = uni.last_action
                prev_k = uni.last_k

                metric_vector = np.array(
                    [
                        float(metrics.get("cohesion", 0.5)),
                        float(metrics.get("reciprocity", 0.5)),
                        float(metrics.get("playfulness", 0.5)),
                        float(metrics.get("te_mutual", 0.0)),
                        float(metrics.get("te_symmetry", 0.0)),
                        float(metrics.get("K", 0.0)),
                    ],
                    dtype=float,
                )
                alpha = float(self.rpc_ema_alpha or 0.2)
                if uni.ema_metrics is None:
                    ema = metric_vector.copy()
                    grad_vec = np.zeros_like(metric_vector)
                else:
                    ema = alpha * metric_vector + (1.0 - alpha) * uni.ema_metrics
                    grad_vec = ema - uni.ema_metrics
                if uni.last_grad is None:
                    lagged_grad = grad_vec.copy()
                elif uni.prev_grad is None:
                    lagged_grad = grad_vec - uni.last_grad
                else:
                    lagged_grad = grad_vec - uni.prev_grad
                uni.ema_metrics = ema
                uni.prev_grad = uni.last_grad
                uni.last_grad = grad_vec

                state_summary = self._build_state_summary(ema, scores, weights)

                if uni.actor is not None:
                    action_vec = uni.actor.predict(state_summary)
                elif self.rpc_policy_model is not None:
                    obs = self._build_policy_observation(uni, metrics, scores)
                    action, _ = self.rpc_policy_model.predict(obs, deterministic=True)
                    action_vec = np.asarray(action, dtype=float).reshape(-1)
                else:
                    action_vec = np.zeros(2, dtype=float)

                params = {
                    "energy_gradient": np.array([uni.params.get("energy_gradient", 0.0)], dtype=float),
                    "plasticity_rate": np.array([uni.params.get("plasticity_rate", 0.0)], dtype=float),
                }
                grads = {
                    # ΔK plus lagged change to capture acceleration in coherence
                    "energy_gradient": np.array([grad_vec[5] + lagged_grad[5]], dtype=float),
                    # Combine Δcohesion, Δplayfulness, Δte_symmetry to reflect structural shifts
                    "plasticity_rate": np.array([grad_vec[0] + grad_vec[2] + grad_vec[4]], dtype=float),
                }
                if action_vec is not None:
                    grads["policy_action"] = action_vec.copy()
                    params["policy_action"] = np.zeros_like(action_vec)
                cohesion = ema[0]
                energy_metric = max(0.0, 1.0 - cohesion)
                energy = float(np.clip(energy_metric, 0.0, self.rpc_coupler.cfg.energy_cap))
                snapshot = PolicySnapshot(params=params, grads=grads, energy=energy)
                if uni.last_snapshot is None:
                    self.rpc_coupler.register_universe(uid, snapshot)
                else:
                    self.rpc_coupler.update_snapshot(uid, snapshot)
                uni.last_snapshot = snapshot

                current_k = float(scores.kosmic_signature(weights=weights))
                if self.phase4 and prev_state is not None and prev_action is not None and prev_k is not None:
                    reward = current_k - prev_k
                    self.phase4.record_transition(
                        state=prev_state,
                        action=prev_action,
                        reward=reward,
                        next_state=state_summary,
                        done=False,
                        universe_id=idx,
                    )
                uni.last_state_summary = state_summary
                uni.last_action = action_vec.astype(np.float32, copy=False)
                uni.last_k = current_k

        if self.coupling_mode == "te":
            if timestep % int(self.te_cfg["update_interval"]) == 0:
                self._update_coupling_weights()

            driver_latest = [np.asarray(hist[-1], dtype=float) for hist in self.driver_history]  # type: ignore[arg-type]
            driver_mean = np.mean(driver_latest, axis=0)

            for j, uni in enumerate(self.universes):
                delta = np.zeros_like(driver_mean, dtype=float)
                for i, w in enumerate(self.weights_matrix[:, j]):  # type: ignore[index]
                    if i == j:
                        continue
                    delta += w * (driver_latest[i] - driver_mean)

                if "energy_gradient" in uni.params:
                    current_energy = uni.params.get("energy_gradient", 0.0)
                    low_e, high_e = self.param_bounds.get("energy_gradient", (None, None))
                    new_energy = current_energy + self.coupling_strength * float(delta[0])
                    if low_e is not None and high_e is not None:
                        new_energy = float(np.clip(new_energy, low_e, high_e))
                    uni.params["energy_gradient"] = float(new_energy)

                if "plasticity_rate" in uni.params:
                    current_plasticity = uni.params.get("plasticity_rate", 0.0)
                    low_p, high_p = self.param_bounds.get("plasticity_rate", (None, None))
                    new_plasticity = current_plasticity + self.coupling_strength * float(delta[1])
                    if low_p is not None and high_p is not None:
                        new_plasticity = float(np.clip(new_plasticity, low_p, high_p))
                    uni.params["plasticity_rate"] = float(new_plasticity)

        elif self.coupling_mode == "rpc" and self.rpc_coupler is not None:
            updates = self.rpc_coupler.compute_coupling()
            if self.rpc_coupler.trace:
                for entry in self.rpc_coupler.trace:
                    entry["timestep"] = timestep
                self._append_trace_rows(self.rpc_coupler.trace)
                self.rpc_coupler.trace.clear()
            for uid, group_updates in updates.items():
                idx = self.uid_to_index.get(uid)
                if idx is None:
                    continue
                uni = self.universes[idx]
                for group, delta_arr in group_updates.items():
                    if delta_arr.size == 0:
                        continue
                    delta = float(delta_arr.reshape(-1)[0])
                    current = uni.params.get(group, 0.0)
                    new_val = current + delta
                    low, high = self.param_bounds.get(group, (None, None))
                    if low is not None and high is not None:
                        new_val = float(np.clip(new_val, low, high))
                    uni.params[group] = float(new_val)

            if self.rpc_diffusion_strength > 0.0:
                groups = {"energy_gradient", "plasticity_rate"}
                best_params: Dict[str, float] = {}
                best_k = -float("inf")
                best_uni: Optional[UniverseState] = None
                for uid, hist in harmonies.items():
                    k_val = hist.kosmic_signature(weights=weights)
                    idx = self.uid_to_index[uid]
                    uni_state = self.universes[idx]
                    if k_val > best_k:
                        best_k = k_val
                        best_uni = uni_state
                if best_uni is not None:
                    for group in groups:
                        if group in best_uni.params:
                            best_params[group] = best_uni.params[group]
                    for uni in self.universes:
                        for group, target_val in best_params.items():
                            if group not in uni.params:
                                continue
                            current = uni.params[group]
                            new_val = current + self.rpc_diffusion_strength * (target_val - current)
                            low, high = self.param_bounds.get(group, (None, None))
                            if low is not None and high is not None:
                                new_val = float(np.clip(new_val, low, high))
                            uni.params[group] = float(new_val)

            if self.phase4 is not None:
                loss, actor_batches = self.phase4.maybe_train()
                if loss is not None:
                    self.phase4_last_loss = loss
                    self.phase4_loss_log.append((timestep, loss))
                if actor_batches:
                    for uid, (states_batch, actions_batch, advantages_batch) in actor_batches.items():
                        idx = self.uid_to_index.get(f"U{uid}") if isinstance(uid, int) else None
                        if idx is None and isinstance(uid, str):
                            idx = self.uid_to_index.get(uid)
                        if idx is None:
                            continue
                        actor = self.universes[idx].actor
                        if actor is None:
                            continue
                        actor.update(states_batch, actions_batch, advantages_batch)

        return harmonies

    def _append_trace_rows(self, rows: List[Dict[str, float]]) -> None:
        import csv
        from pathlib import Path

        path = Path("logs/rpc_trace.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        fieldnames = [
            "timestep",
            "source",
            "target",
            "group",
            "reciprocity",
            "eta",
            "delta_norm",
            "energy_gate",
            "energy",
            "energy_cap",
        ]
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _build_policy_observation(self, uni: UniverseState, metrics: Dict[str, float], scores: HarmonyScores) -> np.ndarray:
        coherence = float(np.clip(scores.resonant_coherence, -1.0, 1.0))
        last_coherence = getattr(uni, "last_coherence", None)
        delta = 0.0 if last_coherence is None else coherence - float(np.clip(last_coherence, -1.0, 1.0))
        cohesion = float(np.clip(metrics.get("cohesion", 0.5), 0.0, 1.0))
        energy_metric = float(np.clip(1.0 - cohesion, 0.0, 1.0))
        obs = np.array([
            coherence,
            np.clip(delta, -1.0, 1.0),
            np.clip(2.0 * energy_metric - 1.0, -1.0, 1.0),
        ], dtype=np.float32)
        uni.last_coherence = coherence
        return obs

    def _build_state_summary(
        self,
        ema_metrics: np.ndarray,
        scores: HarmonyScores,
        weights: np.ndarray,
    ) -> np.ndarray:
        k_val = float(scores.kosmic_signature(weights=weights))
        summary = np.array(
            [
                np.clip(scores.resonant_coherence, -1.0, 1.0),
                np.clip(ema_metrics[0], 0.0, 1.0),
                np.clip(ema_metrics[1], 0.0, 1.0),
                np.clip(ema_metrics[2], 0.0, 1.0),
                np.clip(ema_metrics[3], 0.0, 2.0),
                np.clip(ema_metrics[4], 0.0, 1.0),
                np.clip(k_val, 0.0, 3.0),
            ],
            dtype=np.float32,
        )
        return summary

    def run(self, weights: np.ndarray, steps: int = 50) -> Dict[str, List[HarmonyScores]]:
        for t in range(steps):
            self.step(t, weights)
        return {f"U{i}": uni.history for i, uni in enumerate(self.universes)}

    def _update_coupling_weights(self) -> None:
        if self.coupling_mode != "te" or self.te_cfg is None:
            return

        window = int(self.te_cfg["window"])
        eps = float(self.te_cfg["epsilon"])
        k = int(self.te_cfg.get("knn_k", 5))
        if window < 4:
            return
        if self.driver_history is None or self.driven_history is None:
            return
        if any(len(hist) < window for hist in self.driver_history):
            return

        n = len(self.universes)
        te_raw = np.zeros((n, n))
        for j in range(n):
            y_vals = np.array(self.driven_history[j][-window:], dtype=float)
            y_past = y_vals[:-1]
            y_present = y_vals[1:]
            for i in range(n):
                if i == j:
                    continue
                x_vals = np.array(self.driver_history[i][-window:], dtype=float)
                x_past = x_vals[:-1]
                try:
                    te_val = knn_transfer_entropy(x_past, y_past, y_present, k=k)
                except ValueError:
                    te_val = 0.0
                te_raw[i, j] = float(max(0.0, te_val))

        lam = float(self.te_cfg["symmetry_lambda"])
        te_sym = te_raw.copy()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                te_sym[i, j] = (1 - lam) * te_raw[i, j] + lam * te_raw[j, i]

        for j in range(n):
            col = te_sym[:, j]
            col[j] = 0.0
            s = col.sum()
            if s <= eps:
                te_sym[:, j] = 0.0
            else:
                te_sym[:, j] = col / s
            te_sym[j, j] = 0.0

        self.weights_matrix = te_sym

    def _to_harmony_inputs(
        self,
        sim_metrics: Dict[str, float],
        seed: int,
        timestep: int,
        agent_count: int = 60,
    ) -> Dict:
        rng = np.random.default_rng(seed + timestep)
        cohesion = np.clip(sim_metrics.get("cohesion", 0.5), 0.0, 1.0)
        reciprocity = np.clip(sim_metrics.get("reciprocity", 0.5), 0.0, 1.0)
        playfulness = np.clip(sim_metrics.get("playfulness", 0.5), 0.0, 1.0)

        alive_count = max(5, int(agent_count * (0.4 + 0.5 * cohesion)))
        agent_states = [
            {"alive": True, "type": rng.choice(["explorer", "cooperator", "mediator"])}
            for _ in range(alive_count)
        ]
        agent_states += [{"alive": False, "type": "defector"} for _ in range(agent_count - alive_count)]

        te_matrix = rng.normal(
            loc=sim_metrics.get("te_mutual", 0.5),
            scale=0.05,
            size=(agent_count, agent_count),
        )
        te_matrix = np.clip(te_matrix, 0.0, None)
        np.fill_diagonal(te_matrix, 0.0)

        prediction_errors = {
            "sensory": float(np.clip(1 - cohesion, 0.0, 1.0)),
            "motor": float(np.clip(1 - reciprocity, 0.0, 1.0)),
            "social": float(np.clip(1 - playfulness, 0.0, 1.0)),
        }

        behavioral_history = [
            rng.normal(loc=0.0, scale=0.5 + playfulness, size=(agent_count, 3))
            for _ in range(15)
        ]

        return {
            "phi": sim_metrics.get("phi", 1.0),
            "agent_states": agent_states,
            "te_matrix": te_matrix,
            "prediction_errors": prediction_errors,
            "behavioral_history": behavioral_history,
            "timestep": timestep,
        }

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from core.config import ConfigBundle, load_yaml_config
from fre.controller_sac import SACController, Transition
from fre.simulate import compute_metrics, expand_parameter_grid
from fre.universe import UniverseSimulator
from fre.detailed_logging_patch import (
    should_enable_detailed_logging,
    log_detailed_track_b_step,
    save_track_b_detailed_npz,
)


HARMONY_KEYS = [
    "H1_Coherence",
    "H2_Flourishing",
    "H3_Wisdom",
    "H4_Play",
    "H5_Interconnection",
    "H6_Reciprocity",
    "H7_Evolution",
]


@dataclass
class EpisodeSummary:
    mode: str
    params: Dict[str, float]
    seed: int
    average_k: float
    corridor_rate: float
    final_k: float
    steps: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "params": self.params,
            "seed": self.seed,
            "average_k": self.average_k,
            "corridor_rate": self.corridor_rate,
            "final_k": self.final_k,
            "steps": self.steps,
        }


class TrackBRunner:
    def __init__(self, config_path: Path) -> None:
        self.control_bundle = load_yaml_config(config_path)
        self.control_config = self.control_bundle.payload
        experiment_cfg = self.control_config.get("experiment", {})
        base_config_path = Path(experiment_cfg.get("base_config", "fre/configs/k_config.yaml"))
        self.base_bundle = load_yaml_config(base_config_path)
        self.base_config = self.base_bundle.payload

        controller_cfg = self.control_config.get("controllers", {}).get("sac", {})
        self.horizon = int(controller_cfg.get("horizon", 180))
        self.action_interval = max(1, int(controller_cfg.get("action_interval", 10)))
        self.batch_size = int(controller_cfg.get("batch_size", 256))
        self.warmup_steps = int(controller_cfg.get("warmup_steps", 512))
        self.train_episodes = int(controller_cfg.get("train_episodes", 1))
        self.eval_episodes = int(controller_cfg.get("eval_episodes", 1))
        self.open_loop_episodes = int(experiment_cfg.get("open_loop_episodes", 1))
        self.reward_beta = float(controller_cfg.get("reward_beta", 0.0))

        self.control_params = list(self.control_config.get("parameters", {}).keys())
        if not self.control_params:
            raise ValueError("Track B configuration must specify at least one controllable parameter.")

        self.param_bounds = self._build_param_bounds()
        self.base_params = self._build_base_params()
        self.param_grid = expand_parameter_grid(self.control_config.get("parameters", {}))
        self.seed_base = int(self.base_config.get("seed_base", 44))
        self.threshold = float(self.base_config.get("corridor_threshold", 1.0))
        self.weights = self._weights_from_config(self.base_config.get("k_weights", {}))

        self.action_scales = self._build_action_scales(controller_cfg)
        self.simulator = UniverseSimulator()

        state_dim = len(HARMONY_KEYS) + 2 + 1 + len(self.control_params)
        self.controller = SACController(
            action_dim=len(self.control_params),
            state_dim=state_dim,
            hidden_layers=tuple(controller_cfg.get("hidden_layers", (256, 256))),
            gamma=float(controller_cfg.get("gamma", 0.99)),
            tau=float(controller_cfg.get("tau", 0.005)),
            actor_lr=float(controller_cfg.get("actor_lr", 3e-4)),
            critic_lr=float(controller_cfg.get("critic_lr", 3e-4)),
            alpha_lr=float(controller_cfg.get("alpha_lr", 3e-4)),
            buffer_capacity=int(controller_cfg.get("buffer_capacity", 1_000_000)),
            init_temperature=float(controller_cfg.get("init_temperature", 0.1)),
        )

        self.diagnostics: List[Dict[str, Any]] = []
        self.training_metrics: List[Dict[str, Any]] = []
        self.global_step = 0

        # Detailed logging for Paper 2
        self.enable_detailed = should_enable_detailed_logging(self.control_config)
        self.detailed_records: List[Dict[str, Any]] = []
        self.last_metrics: Dict[str, float] = {}
        self.last_action: np.ndarray = np.zeros(len(self.control_params))
        self.current_params: Dict[str, float] = {}

    def run(self) -> Dict[str, Any]:
        summaries: List[EpisodeSummary] = []
        seed_cursor = self.seed_base

        for combo_idx, combo in enumerate(self.param_grid):
            initial_params = self.base_params.copy()
            initial_params.update(combo)

            for _ in range(self.open_loop_episodes):
                summary, seed_cursor = self._run_episode(
                    mode="open_loop",
                    initial_params=initial_params,
                    seed_start=seed_cursor,
                    train=False,
                    evaluate=False,
                )
                # Save detailed NPZ if enabled
                if self.enable_detailed and self.detailed_records:
                    results_dir = Path(self.control_config.get("results_dir", "logs"))
                    results_dir.mkdir(parents=True, exist_ok=True)
                    episode_num = len(summaries) + 1
                    npz_path = results_dir / f"track_b_ep{episode_num:03d}_{summary.mode}.npz"
                    try:
                        save_track_b_detailed_npz(self.detailed_records, npz_path)
                        print(f"  💾 Saved detailed log: {npz_path}")
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not save detailed NPZ: {e}")
                    self.detailed_records.clear()

                summaries.append(summary)

            for train_idx in range(self.train_episodes):
                summary, seed_cursor = self._run_episode(
                    mode=f"controller_train_{train_idx + 1}",
                    initial_params=initial_params,
                    seed_start=seed_cursor,
                    train=True,
                    evaluate=False,
                )
                # Save detailed NPZ if enabled
                if self.enable_detailed and self.detailed_records:
                    results_dir = Path(self.control_config.get("results_dir", "logs"))
                    results_dir.mkdir(parents=True, exist_ok=True)
                    episode_num = len(summaries) + 1
                    npz_path = results_dir / f"track_b_ep{episode_num:03d}_{summary.mode}.npz"
                    try:
                        save_track_b_detailed_npz(self.detailed_records, npz_path)
                        print(f"  💾 Saved detailed log: {npz_path}")
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not save detailed NPZ: {e}")
                    self.detailed_records.clear()

                summaries.append(summary)

            for eval_idx in range(self.eval_episodes):
                summary, seed_cursor = self._run_episode(
                    mode=f"controller_eval_{eval_idx + 1}",
                    initial_params=initial_params,
                    seed_start=seed_cursor,
                    train=False,
                    evaluate=True,
                )
                # Save detailed NPZ if enabled
                if self.enable_detailed and self.detailed_records:
                    results_dir = Path(self.control_config.get("results_dir", "logs"))
                    results_dir.mkdir(parents=True, exist_ok=True)
                    episode_num = len(summaries) + 1
                    npz_path = results_dir / f"track_b_ep{episode_num:03d}_{summary.mode}.npz"
                    try:
                        save_track_b_detailed_npz(self.detailed_records, npz_path)
                        print(f"  💾 Saved detailed log: {npz_path}")
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not save detailed NPZ: {e}")
                    self.detailed_records.clear()

                summaries.append(summary)

        return self._build_summary_payload(summaries)

    def _run_episode(
        self,
        *,
        mode: str,
        initial_params: Dict[str, float],
        seed_start: int,
        train: bool,
        evaluate: bool,
    ) -> Tuple[EpisodeSummary, int]:
        current_params = initial_params.copy()
        seed = seed_start
        metrics = compute_metrics(current_params, seed, self.simulator, self.weights)
        state = self._build_state(metrics, current_params, delta_k=0.0)
        last_k = float(metrics["K"])

        step_history: List[Dict[str, Any]] = []
        rewards: List[float] = []
        corridor_hits = 0

        raw_action = np.zeros(len(self.control_params), dtype=float)
        next_seed = seed + 1

        for step in range(self.horizon):
            self.global_step += 1
            if train and self.global_step <= self.warmup_steps:
                allow_update = False
            else:
                allow_update = train

            if mode == "open_loop":
                raw_action = np.zeros_like(raw_action)
            elif step % self.action_interval == 0:
                raw_action = np.asarray(self.controller.select_action(state.tolist(), evaluate=evaluate), dtype=float)

            applied_deltas = self._apply_action(raw_action, current_params)
            next_metrics = compute_metrics(current_params, next_seed, self.simulator, self.weights)
            next_k = float(next_metrics["K"])
            delta_k = next_k - last_k
            reward = next_k + self.reward_beta * delta_k
            next_state = self._build_state(next_metrics, current_params, delta_k=delta_k)
            done = step == self.horizon - 1

            if train:
                transition = Transition(
                    state=state.tolist(),
                    action=raw_action.tolist(),
                    reward=float(reward),
                    next_state=next_state.tolist(),
                    done=done,
                )
                self.controller.store_transition(transition)
                if allow_update:
                    metrics_update = self.controller.update(batch_size=self.batch_size)
                    if metrics_update:
                        metrics_update["global_step"] = self.global_step
                        metrics_update["episode_mode"] = mode
                        metrics_update["step"] = step
                        self.training_metrics.append(metrics_update)

            step_record = {
                "global_step": self.global_step,
                "episode_mode": mode,
                "step": step,
                "seed": next_seed,
                "K": next_k,
                "delta_k": delta_k,
                "reward": reward,
                "params": {name: float(current_params[name]) for name in self.control_params},
            }
            for idx, name in enumerate(self.control_params):
                step_record[f"action_{name}"] = float(raw_action[idx])
                step_record[f"delta_{name}"] = float(applied_deltas[idx])
            step_history.append(step_record)

            # Store for detailed logging
            if self.enable_detailed:
                self.last_metrics = metrics
                self.last_action = raw_action
                self.current_params = current_params

            if next_k >= self.threshold:
                corridor_hits += 1
            rewards.append(reward)

            state = next_state
            last_k = next_k
            next_seed += 1

        average_k = float(np.mean([row["K"] for row in step_history])) if step_history else 0.0
        corridor_rate = float(corridor_hits / max(1, len(step_history)))
        summary = EpisodeSummary(
            mode=mode,
            params={name: float(initial_params[name]) for name in self.control_params},
            seed=seed_start,
            average_k=average_k,
            corridor_rate=corridor_rate,
            final_k=float(step_history[-1]["K"]) if step_history else last_k,
            steps=len(step_history),
        )

        self._log_diagnostics(step_history)
        return summary, next_seed

    def _build_state(
        self,
        metrics: Dict[str, float],
        params: Dict[str, float],
        *,
        delta_k: float,
    ) -> np.ndarray:
        features: List[float] = [float(metrics.get(key, 0.0)) for key in HARMONY_KEYS]
        features.append(float(metrics.get("TE_macro_micro", metrics.get("te_mutual", 0.0))))
        features.append(float(metrics.get("TE_symmetry", metrics.get("te_symmetry", 0.0))))
        features.append(float(delta_k))
        for name in self.control_params:
            features.append(float(params.get(name, 0.0)))
        return np.asarray(features, dtype=np.float32)

    def _apply_action(self, raw_action: np.ndarray, params: Dict[str, float]) -> List[float]:
        deltas: List[float] = []
        for idx, name in enumerate(self.control_params):
            scale = self.action_scales[idx]
            delta = float(np.clip(raw_action[idx], -1.0, 1.0) * scale)
            low, high = self.param_bounds[name]
            new_val = float(np.clip(params.get(name, 0.0) + delta, low, high))
            params[name] = new_val
            deltas.append(delta)
        return deltas

    def _build_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
        control_ranges = self.control_config.get("parameters", {})
        for name, value in control_ranges.items():
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                values = [float(v) for v in value]
                bounds[name] = (min(values), max(values))
            else:
                center = float(value)
                bounds[name] = (center - 0.1, center + 0.1)
        base_ranges = self.base_config.get("parameters", {})
        for name, value in base_ranges.items():
            if name in bounds:
                continue
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                values = [float(v) for v in value]
                bounds[name] = (min(values), max(values))
            else:
                center = float(value)
                bounds[name] = (center - 0.1, center + 0.1)
        return bounds

    def _build_base_params(self) -> Dict[str, float]:
        params: Dict[str, float] = {}
        base_ranges = self.base_config.get("parameters", {})
        for name, value in base_ranges.items():
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                values = [float(v) for v in value]
                params[name] = float(np.mean(values))
            else:
                params[name] = float(value)
        return params

    def _weights_from_config(self, weights_cfg: Dict[str, float]) -> np.ndarray:
        if not weights_cfg:
            return np.ones(7, dtype=float) / 7.0
        order = [
            "resonant_coherence",
            "pan_sentient_flourishing",
            "integral_wisdom",
            "infinite_play",
            "universal_interconnectedness",
            "sacred_reciprocity",
            "evolutionary_progression",
        ]
        weights = np.array([float(weights_cfg.get(key, 1.0)) for key in order], dtype=float)
        total = float(weights.sum())
        if not np.isfinite(total) or abs(total) < 1e-12:
            raise ValueError("Invalid K-weight configuration; sum must be finite and non-zero.")
        return weights / total

    def _build_action_scales(self, controller_cfg: Dict[str, Any]) -> List[float]:
        scale_cfg = controller_cfg.get("action_scale", 0.05)
        if isinstance(scale_cfg, dict):
            return [float(scale_cfg.get(name, 0.05)) for name in self.control_params]
        scale = float(scale_cfg)
        return [scale for _ in self.control_params]

    def _log_diagnostics(self, rows: List[Dict[str, Any]]) -> None:
        self.diagnostics.extend(rows)

        # Add detailed logging if enabled
        if self.enable_detailed:
            for row in rows:
                try:
                    enhanced = log_detailed_track_b_step(
                        step_record=row.copy(),
                        metrics=self.last_metrics,
                        current_params=self.current_params,
                        raw_action=self.last_action,
                    )
                    self.detailed_records.append(enhanced)
                except Exception as e:
                    print(f"Warning: Could not create detailed log: {e}")

    def _build_summary_payload(self, summaries: List[EpisodeSummary]) -> Dict[str, Any]:
        open_loop = [s for s in summaries if s.mode.startswith("open_loop")]
        controller = [s for s in summaries if s.mode.startswith("controller_train") or s.mode.startswith("controller_eval")]

        payload = {
            "control_config_sha": self.control_bundle.sha256,
            "base_config_sha": self.base_bundle.sha256,
            "episodes": [s.to_dict() for s in summaries],
            "controller": {
                "replay_size": len(self.controller.replay_buffer),
                "alpha": self.controller.alpha,
            },
            "aggregates": {
                "open_loop_avg_k": float(np.mean([s.average_k for s in open_loop])) if open_loop else 0.0,
                "controller_avg_k": float(np.mean([s.average_k for s in controller])) if controller else 0.0,
                "open_loop_corridor": float(np.mean([s.corridor_rate for s in open_loop])) if open_loop else 0.0,
                "controller_corridor": float(np.mean([s.corridor_rate for s in controller])) if controller else 0.0,
            },
        }
        return payload

    def write_outputs(self, summary: Dict[str, Any]) -> None:
        output_cfg = self.control_config.get("output", {})
        summary_path = Path(output_cfg.get("summary_json", "logs/track_b_summary.json"))
        diagnostics_path = Path(output_cfg.get("diagnostics_csv", "logs/track_b_diagnostics.csv"))
        training_path = Path(output_cfg.get("training_csv", "logs/track_b_training.csv"))

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        training_path.parent.mkdir(parents=True, exist_ok=True)

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if self.diagnostics:
            fieldnames = sorted({key for row in self.diagnostics for key in row.keys()})
            with diagnostics_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.diagnostics:
                    writer.writerow(row)

        if self.training_metrics:
            fieldnames = sorted({key for row in self.training_metrics for key in row.keys()})
            with training_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.training_metrics:
                    writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Track B SAC controller experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fre/configs/track_b_control.yaml"),
        help="Track B control configuration YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = TrackBRunner(args.config)
    summary = runner.run()
    runner.write_outputs(summary)
    print(f"[Track B] Completed {len(summary['episodes'])} episodes. Summary stored at configured output path.")


if __name__ == "__main__":
    main()

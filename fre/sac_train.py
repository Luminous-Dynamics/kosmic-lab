"""Training script for Track C SAC controller using stable-baselines3."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from fre.sac_env import make_env


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC for Track C rescue")
    parser.add_argument("--config", type=Path, default=Path("configs/track_c_sac.yaml"))
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sac_track_c"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = make_env(cfg)

    sac_kwargs = cfg.get("sac", {})
    policy_kwargs = {
        "net_arch": sac_kwargs.get("actor_layers", [256, 256, 128]),
    }

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=sac_kwargs.get("lr", 3e-4),
        batch_size=sac_kwargs.get("batch_size", 256),
        buffer_size=sac_kwargs.get("replay_size", 1_000_000),
        gamma=sac_kwargs.get("gamma", 0.99),
        tau=sac_kwargs.get("tau", 0.005),
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    eval_env = make_env(cfg)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output),
        log_path=str(args.output),
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output / "sac_track_c"))


if __name__ == "__main__":
    main()

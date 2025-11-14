"""Evaluate a trained SAC model on the rescue environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import SAC

from fre.sac_env import make_env


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained SAC model")
    parser.add_argument("--config", type=Path, default=Path("configs/track_c_sac.yaml"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--lesion-scale", type=float, default=None, help="Scale factor for lesion radius"
    )
    parser.add_argument(
        "--lesion-count", type=int, default=None, help="Number of lesions to introduce"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {
        "lesion_radius_scale": args.lesion_scale,
        "lesion_count": args.lesion_count,
    }
    env = make_env(cfg, overrides=overrides)
    model = SAC.load(str(args.model))

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0
        last_info = {}
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            last_info = info
        print(
            f"Episode {ep + 1}: reward={total_reward:.3f}, steps={steps}, final IoU={last_info.get('iou', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from core.seeding import set_global_seed
from core.trainer import Trainer, build_agent, build_env, build_visualizer
from core.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a structural broadcasting RL experiment.")
    parser.add_argument("--config", required=True, help="Path to an application-local YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    rng = set_global_seed(int(config.get("seed", 0)))
    env = build_env(config)
    agent = build_agent(config, env, rng)
    visualizer = build_visualizer(config)
    trainer = Trainer(config, env, agent, rng, visualizer=visualizer, config_path=args.config)
    output_dir = trainer.train()
    print(f"Run complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()

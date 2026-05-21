import pytest

pytest.importorskip("minigrid")

from core.seeding import set_global_seed
from core.trainer import Trainer, build_agent, build_env, build_visualizer
from core.utils import load_yaml


def test_q_learning_keydoor_training_smoke(tmp_path):
    config = load_yaml("applications/key_door/config_qlearning.yaml")
    config["training"]["total_steps"] = 12
    config["training"]["eval_interval"] = 6
    config["training"]["eval_episodes"] = 1
    config["training"]["log_interval"] = 4
    config["training"]["save_interval"] = 0
    config["visualization"]["enabled"] = False
    config["logging"]["output_root"] = str(tmp_path)
    config["logging"]["run_name"] = "smoke_qlearning"

    rng = set_global_seed(config["seed"])
    env = build_env(config)
    agent = build_agent(config, env, rng)
    visualizer = build_visualizer(config)
    trainer = Trainer(config, env, agent, rng, visualizer=visualizer)
    output_dir = trainer.train()

    metrics = output_dir / "metrics.csv"
    eval_metrics = output_dir / "eval_metrics.csv"
    assert metrics.exists()
    assert eval_metrics.exists()
    assert len(eval_metrics.read_text(encoding="utf-8").strip().splitlines()) >= 2

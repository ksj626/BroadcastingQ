from core.seeding import set_global_seed
from core.trainer import Trainer, build_agent, build_env, build_visualizer
from core.utils import load_yaml


def test_q_learning_dna_promoter_training_smoke(tmp_path):
    config = load_yaml("applications/dna_promoter/config_qlearning.yaml")
    config["training"]["total_steps"] = 12
    config["training"]["eval_interval"] = 6
    config["training"]["eval_episodes"] = 1
    config["training"]["log_interval"] = 4
    config["training"]["save_interval"] = 0
    config["training"]["progress_bar"] = False
    config["visualization"]["enabled"] = False
    config["logging"]["output_root"] = str(tmp_path)
    config["logging"]["run_name"] = "smoke_dna_qlearning"

    rng = set_global_seed(config["seed"])
    env = build_env(config)
    agent = build_agent(config, env, rng)
    visualizer = build_visualizer(config)
    trainer = Trainer(config, env, agent, rng, visualizer=visualizer)
    output_dir = trainer.train()

    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "eval_metrics.csv").exists()


def test_q_learning_lights_out_3x3_training_smoke(tmp_path):
    config = load_yaml("applications/lights_out/config_qlearning_3x3.yaml")
    config["training"]["total_steps"] = 12
    config["training"]["eval_interval"] = 6
    config["training"]["eval_episodes"] = 1
    config["training"]["log_interval"] = 4
    config["training"]["save_interval"] = 0
    config["training"]["progress_bar"] = False
    config["visualization"]["enabled"] = False
    config["logging"]["output_root"] = str(tmp_path)
    config["logging"]["run_name"] = "smoke_lights_qlearning"

    rng = set_global_seed(config["seed"])
    env = build_env(config)
    agent = build_agent(config, env, rng)
    visualizer = build_visualizer(config)
    trainer = Trainer(config, env, agent, rng, visualizer=visualizer)
    output_dir = trainer.train()

    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "eval_metrics.csv").exists()


def test_q_learning_lights_out_4x4_training_smoke(tmp_path):
    config = load_yaml("applications/lights_out/config_qlearning_4x4.yaml")
    config["training"]["total_steps"] = 8
    config["training"]["eval_interval"] = 4
    config["training"]["eval_episodes"] = 1
    config["training"]["log_interval"] = 4
    config["training"]["save_interval"] = 0
    config["training"]["progress_bar"] = False
    config["visualization"]["enabled"] = False
    config["logging"]["output_root"] = str(tmp_path)
    config["logging"]["run_name"] = "smoke_lights_4x4_qlearning"

    rng = set_global_seed(config["seed"])
    env = build_env(config)
    agent = build_agent(config, env, rng)
    visualizer = build_visualizer(config)
    trainer = Trainer(config, env, agent, rng, visualizer=visualizer)
    output_dir = trainer.train()

    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "eval_metrics.csv").exists()


def test_q_learning_lights_out_5x5_sparse_training_smoke(tmp_path):
    config = load_yaml("applications/lights_out/config_qlearning_5x5.yaml")
    config["training"]["total_steps"] = 6
    config["training"]["eval_interval"] = 3
    config["training"]["eval_episodes"] = 1
    config["training"]["log_interval"] = 3
    config["training"]["save_interval"] = 0
    config["training"]["progress_bar"] = False
    config["visualization"]["enabled"] = False
    config["logging"]["output_root"] = str(tmp_path)
    config["logging"]["run_name"] = "smoke_lights_5x5_qlearning"

    rng = set_global_seed(config["seed"])
    env = build_env(config)
    agent = build_agent(config, env, rng)
    visualizer = build_visualizer(config)
    trainer = Trainer(config, env, agent, rng, visualizer=visualizer)
    output_dir = trainer.train()

    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "eval_metrics.csv").exists()

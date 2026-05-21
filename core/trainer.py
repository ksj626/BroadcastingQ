from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from core.agent_base import BaseAgent, Transition
from core.logging import CSVLogger
from core.schedules import make_schedule
from core.utils import ensure_dir, import_from_string

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None



AGENT_REGISTRY = {
    "random": "agents.random_agent:RandomAgent",
    "q_learning": "agents.q_learning:QLearningAgent",
    "qlearning": "agents.q_learning:QLearningAgent",
    "sarsa": "agents.sarsa:SarsaAgent",
    "dqn": "agents.dqn:DQNAgent",
    "anova_q": "agents.anova_q:AnovaQAgent",
}


def build_env(config: dict):
    env_cls = import_from_string(config["application"]["entrypoint"])
    return env_cls(config)


def build_visualizer(config: dict):
    path = config.get("application", {}).get("visualizer")
    if not path:
        return None
    visualizer_cls = import_from_string(path)
    return visualizer_cls(config)


def build_agent(config: dict, env, rng: np.random.Generator) -> BaseAgent:
    agent_config = dict(config.get("agent", {}))
    name = agent_config.pop("name")
    epsilon_config = agent_config.pop("epsilon", None)
    initial_epsilon = 0.0
    if epsilon_config is not None:
        initial_epsilon = float(epsilon_config.get("start", epsilon_config.get("value", 0.0)))
    agent_cls = import_from_string(AGENT_REGISTRY[name])
    return agent_cls(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
        epsilon=initial_epsilon,
        **agent_config,
    )


class Trainer:
    def __init__(
        self,
        config: dict,
        env,
        agent: BaseAgent,
        rng: np.random.Generator,
        visualizer=None,
        config_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.env = env
        self.agent = agent
        self.rng = rng
        self.visualizer = visualizer
        self.config_path = Path(config_path) if config_path else None
        self.agent_name = config["agent"]["name"]
        self.epsilon_schedule = make_schedule(config["agent"].get("epsilon"), default=getattr(agent, "epsilon", 0.0))
        self.eval_env = build_env(config)
        self.output_dir = self._prepare_output_dir()
        self.metrics_logger = CSVLogger(
            self.output_dir / "metrics.csv",
            [
                "step",
                "episode",
                "episode_return",
                "episode_length",
                "success",
                "epsilon",
                "td_error",
                "loss",
                "q_value",
                "target",
                "td_error_mean",
                "q_mean",
            ],
        )
        self.eval_logger = CSVLogger(
            self.output_dir / "eval_metrics.csv",
            [
                "step",
                "eval_return_mean",
                "eval_return_std",
                "eval_length_mean",
                "eval_success_rate",
            ],
        )

    def train(self) -> Path:
        training = self.config.get("training", {})
        total_steps = int(training.get("total_steps", 1_000))
        eval_interval = int(training.get("eval_interval", 10_000))
        eval_episodes = int(training.get("eval_episodes", 5))
        log_interval = int(training.get("log_interval", 1_000))
        save_interval = int(training.get("save_interval", 50_000))
        seed = int(self.config.get("seed", 0))

        obs, info = self.env.reset(seed=seed)
        self.agent.begin_episode()
        action = self.agent.act(obs, explore=True) if self.agent_name == "sarsa" else None
        episode = 0
        episode_return = 0.0
        episode_length = 0
        last_update: dict[str, Any] = {}
        last_info = info

        try:
            steps = range(1, total_steps + 1)
            if tqdm is not None and bool(training.get("progress_bar", True)):
                steps = tqdm(steps, desc=f"{self.agent_name} train", dynamic_ncols=True)
            for step in steps:
                epsilon = float(self.epsilon_schedule(step))
                if hasattr(self.agent, "epsilon"):
                    self.agent.epsilon = epsilon

                current_action = action if self.agent_name == "sarsa" else self.agent.act(obs, explore=True)
                next_obs, reward, terminated, truncated, info = self.env.step(int(current_action))
                episode_return += float(reward)
                episode_length += 1
                last_info = info

                if self.agent_name == "sarsa":
                    next_action = None if terminated or truncated else self.agent.act(next_obs, explore=True)
                    transition = Transition(obs, int(current_action), reward, next_obs, terminated, truncated, info, next_action)
                    last_update = self.agent.update(transition)
                    action = next_action
                else:
                    transition = Transition(obs, int(current_action), reward, next_obs, terminated, truncated, info)
                    last_update = self.agent.update(transition)

                obs = next_obs

                if log_interval > 0 and step % log_interval == 0:
                    self._log_train_row(step, episode, episode_return, episode_length, last_info, epsilon, last_update)

                if terminated or truncated:
                    self.agent.end_episode()
                    self._log_train_row(step, episode, episode_return, episode_length, info, epsilon, last_update)
                    episode += 1
                    obs, info = self.env.reset()
                    self.agent.begin_episode()
                    action = self.agent.act(obs, explore=True) if self.agent_name == "sarsa" else None
                    episode_return = 0.0
                    episode_length = 0

                if eval_interval > 0 and step % eval_interval == 0:
                    eval_metrics = self.evaluate(step, eval_episodes)
                    self.eval_logger.write(eval_metrics)

                if save_interval > 0 and step % save_interval == 0:
                    self._save_checkpoint(step)

            if eval_interval > 0 and total_steps % eval_interval != 0:
                self.eval_logger.write(self.evaluate(total_steps, eval_episodes))
            self._save_checkpoint(total_steps)
        finally:
            self.metrics_logger.close()
            self.eval_logger.close()
            self.env.close()
            self.eval_env.close()
        return self.output_dir

    def evaluate(self, step: int, episodes: int) -> dict:
        returns = []
        lengths = []
        successes = []
        visualization = self.config.get("visualization", {})
        vis_enabled = bool(visualization.get("enabled", False))
        vis_interval = int(visualization.get("interval", self.config.get("training", {}).get("eval_interval", 1)))
        vis_episodes = int(visualization.get("episodes", 1))
        vis_max_steps = int(visualization.get("max_steps", 200))
        eval_max_steps = self.config.get("training", {}).get("eval_max_steps")
        eval_max_steps = None if eval_max_steps is None else int(eval_max_steps)
        fps = int(visualization.get("fps", 4))
        save_visuals = vis_enabled and self.visualizer is not None and (vis_interval <= 0 or step % vis_interval == 0)

        episode_iter = range(int(episodes))
        if tqdm is not None and bool(self.config.get("training", {}).get("progress_bar", True)):
            episode_iter = tqdm(
                episode_iter,
                desc=f"{self.agent_name} eval {step}",
                leave=False,
                dynamic_ncols=True,
            )

        for episode_idx in episode_iter:
            obs, _ = self.eval_env.reset()
            episode_return = 0.0
            frames = []
            done = False
            length = 0
            while not done and (eval_max_steps is None or length < eval_max_steps):
                if save_visuals and episode_idx < vis_episodes and len(frames) < vis_max_steps:
                    frames.append(self.eval_env.render(mode="rgb_array"))
                action = self.agent.act(obs, explore=False)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_return += float(reward)
                length += 1
                done = bool(terminated or truncated)
            if save_visuals and episode_idx < vis_episodes and len(frames) < vis_max_steps:
                frames.append(self.eval_env.render(mode="rgb_array"))
                self._save_visualization(frames, step, episode_idx, fps)
            returns.append(episode_return)
            lengths.append(length)
            successes.append(float(info.get("success", bool(done and episode_return > 0.0))))

        return {
            "step": step,
            "eval_return_mean": float(np.mean(returns)),
            "eval_return_std": float(np.std(returns)),
            "eval_length_mean": float(np.mean(lengths)),
            "eval_success_rate": float(np.mean(successes)),
        }

    def _prepare_output_dir(self) -> Path:
        logging_config = self.config.get("logging", {})
        output_root = Path(logging_config.get("output_root", "outputs"))
        run_name = logging_config.get("run_name", f"run_seed{self.config.get('seed', 0)}")
        output_dir = ensure_dir(output_root / run_name)
        ensure_dir(output_dir / "checkpoints")
        ensure_dir(output_dir / "visualizations")
        if self.config_path is not None:
            shutil.copyfile(self.config_path, output_dir / "config.yaml")
        return output_dir

    def _log_train_row(
        self,
        step: int,
        episode: int,
        episode_return: float,
        episode_length: int,
        info: dict,
        epsilon: float,
        update_info: dict,
    ) -> None:
        row = {
            "step": step,
            "episode": episode,
            "episode_return": float(episode_return),
            "episode_length": int(episode_length),
            "success": bool(info.get("success", False)),
            "epsilon": float(epsilon),
        }
        row.update(update_info or {})
        self.metrics_logger.write(row)

    def _save_checkpoint(self, step: int) -> None:
        suffix = ".pt" if self.agent_name == "dqn" else ".npy"
        self.agent.save(str(self.output_dir / "checkpoints" / f"agent_step_{step}{suffix}"))

    def _save_visualization(self, frames: list[np.ndarray], step: int, episode_idx: int, fps: int) -> None:
        if not frames:
            return
        vis_dir = self.output_dir / "visualizations"
        prefix = vis_dir / f"eval_step_{step}_episode_{episode_idx}"
        self.visualizer.save_episode_gif(frames, prefix.with_suffix(".gif"), fps=fps)
        self.visualizer.save_final_frame(frames[-1], prefix.with_suffix(".png"))

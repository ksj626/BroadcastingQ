from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="CUDA initialization: The NVIDIA driver on your system is too old.*",
    category=UserWarning,
)

import torch
from torch import nn

from core.agent_base import BaseAgent, Transition
from core.replay_buffer import ReplayBuffer
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, int(size)))
            layers.append(nn.ReLU())
            prev = int(size)
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        observation_space: MultiDiscreteSpace,
        action_space: DiscreteActionSpace,
        rng: np.random.Generator,
        gamma: float = 0.99,
        learning_rate: float = 5e-4,
        batch_size: int = 64,
        replay_size: int = 50_000,
        warmup_steps: int = 1_000,
        target_update_interval: int = 1_000,
        hidden_sizes: list[int] | None = None,
        epsilon: float = 0.1,
        grad_clip_norm: float | None = None,
        device: str | None = "auto",
        **kwargs,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.rng = rng
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.target_update_interval = int(target_update_interval)
        self.epsilon = float(epsilon)
        self.grad_clip_norm = grad_clip_norm
        self.device = self._resolve_device(device)
        hidden_sizes = [128, 128] if hidden_sizes is None else hidden_sizes

        self.q_network = MLPQNetwork(observation_space.flat_dim, action_space.n, hidden_sizes).to(self.device)
        self.target_network = MLPQNetwork(observation_space.flat_dim, action_space.n, hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=float(learning_rate))
        self.replay_buffer = ReplayBuffer(
            capacity=int(replay_size),
            obs_shape=(len(observation_space.nvec),),
            obs_dtype=np.int64,
        )
        self.update_steps = 0

    def act(self, obs, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.action_space.sample(self.rng)
        with torch.no_grad():
            x = self._obs_batch_to_tensor(np.asarray([obs]))
            q_values = self.q_network(x)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, transition: Transition) -> dict:
        self.replay_buffer.add(
            transition.obs,
            transition.action,
            transition.reward,
            transition.next_obs,
            transition.terminated,
            transition.truncated,
        )
        if len(self.replay_buffer) < max(self.warmup_steps, self.batch_size):
            return {}

        batch = self.replay_buffer.sample(self.batch_size, self.rng)
        obs = self._obs_batch_to_tensor(batch["obs"])
        next_obs = self._obs_batch_to_tensor(batch["next_obs"])
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32, device=self.device)

        q_values = self.q_network(obs).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_obs).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - terminated) * next_q

        td_errors = targets - q_values
        loss = torch.mean(td_errors.pow(2))
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), float(self.grad_clip_norm))
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": float(loss.detach().cpu().item()),
            "td_error_mean": float(td_errors.detach().mean().cpu().item()),
            "q_mean": float(q_values.detach().mean().cpu().item()),
        }

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_steps": self.update_steps,
            },
            out,
        )

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(state["q_network"])
        self.target_network.load_state_dict(state["target_network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update_steps = int(state.get("update_steps", 0))

    def _obs_batch_to_tensor(self, obs_batch: np.ndarray) -> torch.Tensor:
        flat = np.stack([self.observation_space.flatten(obs) for obs in obs_batch]).astype(np.float32)
        return torch.as_tensor(flat, dtype=torch.float32, device=self.device)

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device is None or str(device).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

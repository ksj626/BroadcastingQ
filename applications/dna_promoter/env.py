from __future__ import annotations

import numpy as np

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class DNAPromoterEnv(BaseEnv):
    BASES = ("A", "C", "G", "T")
    BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASES)}
    DEFAULT_TARGETS = ("TATAAT", "TATAAA", "TATATT", "TAAAAT")

    def __init__(self, config: dict) -> None:
        self.config = config
        env_config = config.get("env", {})
        self.sequence_length = int(env_config.get("sequence_length", 6))
        self.max_steps = int(env_config.get("max_steps", 20))
        self.avoid_target_start = bool(env_config.get("avoid_target_start", True))
        self.start_sequence = env_config.get("start_sequence")
        self.rng = np.random.default_rng(int(config.get("seed", 0)))
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        # Build the target set before spaces so invalid biology config fails early.
        raw_targets = env_config.get("target_sequences", self.DEFAULT_TARGETS)
        self.target_sequences = tuple(self._normalize_sequence(seq) for seq in raw_targets)
        if not self.target_sequences:
            raise ValueError("target_sequences must contain at least one sequence")
        self.target_set = set(self.target_sequences)

        # The repository uses factorized categorical observations for tabular and DQN agents.
        self.observation_space = MultiDiscreteSpace([len(self.BASES)] * self.sequence_length)
        self.action_space = DiscreteActionSpace(self.sequence_length * (len(self.BASES) - 1))
        if self.avoid_target_start and len(self.target_set) >= self.observation_space.size:
            raise ValueError("avoid_target_start=True requires at least one non-target state")
        self.state = np.zeros(self.sequence_length, dtype=np.int64)
        self.steps = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0

        # Start from a configured sequence when supplied; otherwise sample a non-terminal state.
        if self.start_sequence is not None:
            self.state = self.sequence_to_obs(self.start_sequence)
        else:
            self.state = self._sample_initial_state()
        return self.state.copy(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # Decode action as one position plus a cyclic mutation to a different base.
        position, delta = self.decode_action(action)
        previous_base = int(self.state[position])
        self.state[position] = (previous_base + delta) % len(self.BASES)
        self.steps += 1

        # Reward only exact membership in the configured sparse target set.
        sequence = self.obs_to_sequence(self.state)
        terminated = sequence in self.target_set
        truncated = bool(not terminated and self.steps >= self.max_steps)
        reward = 1.0 if terminated else 0.0

        info = self._info()
        info.update(
            {
                "action_position": int(position),
                "previous_base": self.BASES[previous_base],
                "new_base": self.BASES[int(self.state[position])],
                "success": bool(terminated),
            }
        )
        return self.state.copy(), float(reward), bool(terminated), truncated, info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("DNAPromoterEnv currently supports only mode='rgb_array'")

        # Draw one colored tile per base; tests only require a valid RGB frame.
        cell = 56
        margin = 8
        height = cell + 2 * margin
        width = self.sequence_length * cell + 2 * margin
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        colors = np.array(
            [
                [68, 170, 95],
                [65, 125, 210],
                [235, 155, 55],
                [210, 75, 85],
            ],
            dtype=np.uint8,
        )
        for idx, base in enumerate(self.state):
            x0 = margin + idx * cell
            frame[margin : margin + cell - 4, x0 : x0 + cell - 4] = colors[int(base)]
        return frame

    def decode_action(self, action: int) -> tuple[int, int]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        return action // (len(self.BASES) - 1), action % (len(self.BASES) - 1) + 1

    def sequence_to_obs(self, sequence: str) -> np.ndarray:
        sequence = self._normalize_sequence(sequence)
        return np.array([self.BASE_TO_INDEX[base] for base in sequence], dtype=np.int64)

    def obs_to_sequence(self, obs) -> str:
        arr = np.asarray(obs, dtype=np.int64)
        if not self.observation_space.contains(arr):
            raise ValueError(f"{obs!r} is not a valid DNA observation")
        return "".join(self.BASES[int(value)] for value in arr)

    def hamming_distance(self, left: str, right: str) -> int:
        left = self._normalize_sequence(left)
        right = self._normalize_sequence(right)
        return int(sum(a != b for a, b in zip(left, right)))

    def _sample_initial_state(self) -> np.ndarray:
        while True:
            obs = self.observation_space.sample(self.rng)
            if not self.avoid_target_start or self.obs_to_sequence(obs) not in self.target_set:
                return obs

    def _info(self) -> dict:
        sequence = self.obs_to_sequence(self.state)
        nearest_target = min(self.target_sequences, key=lambda target: self.hamming_distance(sequence, target))
        distance = self.hamming_distance(sequence, nearest_target)
        target_hit = sequence in self.target_set
        return {
            "sequence": sequence,
            "target_hit": bool(target_hit),
            "nearest_target": nearest_target,
            "hamming_distance": int(distance),
            "step_count": int(self.steps),
            "success": bool(target_hit),
        }

    def _normalize_sequence(self, sequence: str) -> str:
        sequence = str(sequence).upper()
        if len(sequence) != self.sequence_length:
            raise ValueError(f"Expected DNA sequence of length {self.sequence_length}, got {sequence!r}")
        unknown = sorted(set(sequence) - set(self.BASES))
        if unknown:
            raise ValueError(f"Unknown DNA bases in {sequence!r}: {unknown}")
        return sequence

from __future__ import annotations

import numpy as np

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class LightsOutEnv(BaseEnv):
    def __init__(self, config: dict) -> None:
        self.config = config
        env_config = config.get("env", {})
        self.rows = int(env_config.get("rows", env_config.get("size", 5)))
        self.cols = int(env_config.get("cols", env_config.get("size", 5)))
        self.max_steps = int(env_config.get("max_steps", self.rows * self.cols * 4))
        self.action_slip_prob = float(env_config.get("action_slip_prob", 0.05))
        self.start_mode = str(env_config.get("start_mode", "solvable_scramble"))
        self.scramble_steps = int(env_config.get("scramble_steps", self.rows * self.cols))
        self.avoid_solved_start = bool(env_config.get("avoid_solved_start", True))
        self.start_board = env_config.get("start_board")
        self.rng = np.random.default_rng(int(config.get("seed", 0)))
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.scramble_steps < 0:
            raise ValueError("scramble_steps must be non-negative")
        if not 0.0 <= self.action_slip_prob <= 1.0:
            raise ValueError("action_slip_prob must be in [0, 1]")

        # Each light is a binary categorical factor in row-major order.
        self.observation_space = MultiDiscreteSpace([2] * (self.rows * self.cols))
        self.action_space = DiscreteActionSpace(self.rows * self.cols)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int64)
        self.steps = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0

        # Create a start board using the configured mode.
        if self.start_board is not None:
            self.board = self._parse_board(self.start_board)
        elif self.start_mode == "solvable_scramble":
            self.board = self._scrambled_board()
        elif self.start_mode == "random":
            self.board = self.rng.integers(0, 2, size=(self.rows, self.cols), dtype=np.int64)
        elif self.start_mode in {"off", "all_off", "zeros"}:
            self.board = np.zeros((self.rows, self.cols), dtype=np.int64)
        else:
            raise ValueError(f"Unknown Lights Out start_mode: {self.start_mode}")
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # A slip replaces the requested button with a different uniformly sampled button.
        requested_action = int(action)
        executed_action = requested_action
        action_slipped = False
        if self.action_space.n > 1 and self.rng.random() < self.action_slip_prob:
            offset = int(self.rng.integers(self.action_space.n - 1))
            executed_action = int((requested_action + 1 + offset) % self.action_space.n)
            action_slipped = True

        self._toggle(executed_action)
        self.steps += 1

        # Strict sparse reward: only solving the board gives reward.
        solved = self.is_solved()
        terminated = bool(solved)
        truncated = bool(not terminated and self.steps >= self.max_steps)
        reward = 1.0 if solved else 0.0

        info = self._info()
        row, col = divmod(executed_action, self.cols)
        info.update(
            {
                "requested_action": int(requested_action),
                "executed_action": int(executed_action),
                "action_slipped": bool(action_slipped),
                "executed_position": (int(row), int(col)),
                "success": bool(solved),
            }
        )
        return self._obs(), float(reward), terminated, truncated, info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("LightsOutEnv currently supports only mode='rgb_array'")

        # Draw the board as fixed-size RGB tiles.
        cell = 36
        gap = 3
        height = self.rows * cell + (self.rows + 1) * gap
        width = self.cols * cell + (self.cols + 1) * gap
        frame = np.full((height, width, 3), 32, dtype=np.uint8)
        off = np.array([35, 42, 58], dtype=np.uint8)
        on = np.array([247, 218, 90], dtype=np.uint8)
        for row in range(self.rows):
            for col in range(self.cols):
                y0 = gap + row * (cell + gap)
                x0 = gap + col * (cell + gap)
                frame[y0 : y0 + cell, x0 : x0 + cell] = on if self.board[row, col] else off
        return frame

    def is_solved(self) -> bool:
        return bool(np.all(self.board == 0))

    def _obs(self) -> np.ndarray:
        return self.board.reshape(-1).astype(np.int64).copy()

    def _toggle(self, action: int) -> None:
        self._toggle_board(self.board, action)

    def _toggle_board(self, board: np.ndarray, action: int) -> None:
        row, col = divmod(int(action), self.cols)
        for rr, cc in ((row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= rr < self.rows and 0 <= cc < self.cols:
                board[rr, cc] ^= 1

    def _scrambled_board(self) -> np.ndarray:
        for _ in range(100):
            board = np.zeros((self.rows, self.cols), dtype=np.int64)
            for _ in range(self.scramble_steps):
                action = int(self.rng.integers(self.action_space.n))
                self._toggle_board(board, action)
            if not self.avoid_solved_start or np.any(board):
                return board

        # Degenerate seeds can still be forced into a non-terminal solvable state.
        board = np.zeros((self.rows, self.cols), dtype=np.int64)
        board[0, 0] = 1
        if self.rows > 1:
            board[1, 0] = 1
        if self.cols > 1:
            board[0, 1] = 1
        return board

    def _parse_board(self, board) -> np.ndarray:
        arr = np.asarray(board, dtype=np.int64)
        if arr.shape == (self.rows * self.cols,):
            arr = arr.reshape(self.rows, self.cols)
        if arr.shape != (self.rows, self.cols):
            raise ValueError(f"start_board must have shape ({self.rows}, {self.cols}) or ({self.rows * self.cols},)")
        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError("start_board values must be 0 or 1")
        return arr.copy()

    def _info(self) -> dict:
        lights_on = int(np.sum(self.board))
        solved = lights_on == 0
        return {
            "board": self._obs(),
            "lights_on": lights_on,
            "step_count": int(self.steps),
            "success": bool(solved),
        }

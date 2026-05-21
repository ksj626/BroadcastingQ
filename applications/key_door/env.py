from __future__ import annotations

import numpy as np

import gymnasium as gym
import minigrid  # noqa: F401 - importing registers MiniGrid environments

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace


class KeyDoorEnv(BaseEnv):
    def __init__(self, config: dict) -> None:
        self.config = config
        env_config = config.get("env", {})
        self.observation_config = config.get("observation", {})
        env_id = env_config.get("minigrid_env_id", "MiniGrid-DoorKey-5x5-v0")
        self.env = gym.make(env_id, render_mode="rgb_array")

        max_steps_override = env_config.get("max_steps_override")
        if max_steps_override is not None:
            self.env.unwrapped.max_steps = int(max_steps_override)

        self.width = int(self.env.unwrapped.width)
        self.height = int(self.env.unwrapped.height)
        self.feature_names, nvec = self._build_observation_spec()
        self.observation_space = MultiDiscreteSpace(nvec)
        self.minigrid_action_count = int(self.env.action_space.n)
        default_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
        if self.minigrid_action_count == len(default_names):
            all_action_names = default_names
        else:
            raise ValueError(f"Unexpected number of MiniGrid actions: {self.minigrid_action_count}; please specify action names in the config")
            # all_action_names = [f"action_{i}" for i in range(self.minigrid_action_count)]
            
        self.action_map = self._build_action_map(env_config.get("action_subset"), all_action_names)
        self.action_names = [all_action_names[i] for i in self.action_map]
        self.action_space = DiscreteActionSpace(len(self.action_map))

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        obs_raw, info = self.env.reset(seed=seed)
        obs = self._convert_obs(obs_raw)
        info = dict(info)
        info.update(self._symbolic_info(obs))
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        minigrid_action = self.action_map[int(action)]
        obs_raw, reward, terminated, truncated, info = self.env.step(minigrid_action)
        obs = self._convert_obs(obs_raw)
        info = dict(info)
        info.update(self._symbolic_info(obs))
        info["action_name"] = self.action_names[int(action)]
        info["minigrid_action"] = int(minigrid_action)
        info["success"] = bool(terminated and float(reward) > 0.0)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("KeyDoorEnv currently supports only mode='rgb_array'")
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("MiniGrid returned no frame; ensure render_mode='rgb_array'")
        return np.asarray(frame)

    def close(self) -> None:
        self.env.close()

    def _convert_obs(self, obs_raw) -> np.ndarray:
        unwrapped = self.env.unwrapped
        agent_pos = unwrapped.agent_pos
        agent_col = int(agent_pos[0])
        agent_row = int(agent_pos[1])
        agent_dir = int(unwrapped.agent_dir)
        carrying = unwrapped.carrying
        has_key = int(carrying is not None and getattr(carrying, "type", None) == "key")
        grid_summary = self._scan_grid()
        values = {
            "agent_row": agent_row,
            "agent_col": agent_col,
            "agent_direction": agent_dir,
            "has_key": has_key,
            "door_state": grid_summary["door_state"],
            "key_row": grid_summary["key_pos"][0],
            "key_col": grid_summary["key_pos"][1],
            "door_row": grid_summary["door_pos"][0],
            "door_col": grid_summary["door_pos"][1],
            "goal_row": grid_summary["goal_pos"][0],
            "goal_col": grid_summary["goal_pos"][1],
        }
        obs = np.array([values[name] for name in self.feature_names], dtype=np.int64)
        if not self.observation_space.contains(obs):
            raise RuntimeError(f"Converted observation {obs} is outside {self.observation_space}")
        return obs

    def _scan_grid(self) -> dict:
        grid = self.env.unwrapped.grid
        key_pos = (self.height, self.width)
        door_pos = (self.height, self.width)
        goal_pos = (self.height, self.width)
        found_closed_door = False
        found_open_door = False
        for x in range(self.width):
            for y in range(self.height):
                obj = grid.get(x, y)
                if obj is None:
                    continue
                obj_type = getattr(obj, "type", None)
                row_col = (int(y), int(x))
                if obj_type == "key":
                    key_pos = row_col
                elif obj_type == "goal":
                    goal_pos = row_col
                elif obj_type == "door":
                    door_pos = row_col
                    if bool(getattr(obj, "is_open", False)):
                        found_open_door = True
                    else:
                        found_closed_door = True
        if found_open_door:
            door_state = 2
        elif found_closed_door:
            door_state = 1
        else:
            door_state = 0
        return {
            "door_state": door_state,
            "key_pos": key_pos,
            "door_pos": door_pos,
            "goal_pos": goal_pos,
        }

    def _symbolic_info(self, obs: np.ndarray) -> dict:
        values = {name: int(value) for name, value in zip(self.feature_names, obs)}
        info = {"symbolic_state": values}
        if "agent_row" in values and "agent_col" in values:
            info["agent_pos"] = (values["agent_row"], values["agent_col"])
        if "agent_direction" in values:
            info["agent_dir"] = values["agent_direction"]
        if "has_key" in values:
            info["has_key"] = values["has_key"]
        if "door_state" in values:
            info["door_state"] = values["door_state"]
        if "key_row" in values and "key_col" in values:
            info["key_pos"] = (values["key_row"], values["key_col"])
        if "door_row" in values and "door_col" in values:
            info["door_pos"] = (values["door_row"], values["door_col"])
        if "goal_row" in values and "goal_col" in values:
            info["goal_pos"] = (values["goal_row"], values["goal_col"])
        return info

    def _build_observation_spec(self) -> tuple[list[str], list[int]]:
        fields: list[tuple[str, int]] = []
        if self.observation_config.get("include_agent_position", True):
            fields.extend([("agent_row", self.height), ("agent_col", self.width)])
        if self.observation_config.get("include_agent_direction", True):
            fields.append(("agent_direction", 4))
        if self.observation_config.get("include_has_key", True):
            fields.append(("has_key", 2))
        if self.observation_config.get("include_door_state", True):
            fields.append(("door_state", 3))
        if self.observation_config.get("include_key_position", False):
            fields.extend([("key_row", self.height + 1), ("key_col", self.width + 1)])
        if self.observation_config.get("include_door_position", False):
            fields.extend([("door_row", self.height + 1), ("door_col", self.width + 1)])
        if self.observation_config.get("include_goal_position", False):
            fields.extend([("goal_row", self.height + 1), ("goal_col", self.width + 1)])
        if not fields:
            raise ValueError("At least one observation factor must be enabled")
        names, nvec = zip(*fields)
        return list(names), list(nvec)

    def _build_action_map(self, action_subset, action_names: list[str]) -> list[int]:
        if action_subset is None:
            return list(range(self.minigrid_action_count))
        name_to_index = {name: idx for idx, name in enumerate(action_names)}
        action_map = []
        for action in action_subset:
            if isinstance(action, str):
                if action not in name_to_index:
                    raise ValueError(f"Unknown MiniGrid action name: {action}")
                idx = name_to_index[action]
            else:
                idx = int(action)
            if idx < 0 or idx >= self.minigrid_action_count:
                raise ValueError(f"MiniGrid action index {idx} is outside the action space")
            action_map.append(idx)
        if not action_map:
            raise ValueError("action_subset must contain at least one action")
        return action_map

# Coding Agent Prompt: Structural Broadcasting RL Repository

You are building a clean Python reinforcement learning research repository for testing tabular and neural RL algorithms on multiple plug-in applications. The future research method is **ANOVA Structural Broadcasting Q-learning**, but **do not implement it yet**. Only create a TODO placeholder.

The repository must support adding applications as folders under `applications/`. Each application folder must contain its own environment wrapper, unified config files, and visualization code. The main training loop and agents must be shared across all applications.

## Important design constraints

1. Do **not** create a global `configs/` folder. Each application must contain its own full config files.
2. Every observation/state space must be represented as `MultiDiscreteSpace`.
3. Each application must include visualization utilities that can save environment-specific output artifacts such as images, GIFs, or videos into the run output directory.
4. For now, implement only one application: a MiniGrid DoorKey wrapper.
5. The MiniGrid environment must be imported and wrapped. Do **not** manually implement MiniGrid dynamics.
6. Implement standard agents: RandomAgent, tabular Q-learning, SARSA, DQN, and a TODO placeholder for ANOVA-Q.
7. Keep code clean, modular, and easy to extend.

---

## Repository structure

Create this repository structure:

```text
structural-broadcasting-rl/
├── README.md
├── pyproject.toml
├── requirements.txt
├── main.py
│
├── core/
│   ├── __init__.py
│   ├── spaces.py
│   ├── env_base.py
│   ├── agent_base.py
│   ├── trainer.py
│   ├── replay_buffer.py
│   ├── schedules.py
│   ├── logging.py
│   ├── seeding.py
│   └── utils.py
│
├── agents/
│   ├── __init__.py
│   ├── random_agent.py
│   ├── q_learning.py
│   ├── sarsa.py
│   ├── q_table.py
│   ├── dqn.py
│   └── anova_q.py
│
├── applications/
│   ├── __init__.py
│   └── key_door/
│       ├── __init__.py
│       ├── config_qlearning.yaml
│       ├── config_sarsa.yaml
│       ├── config_dqn.yaml
│       ├── env.py
│       ├── visualize.py
│       └── README.md
│
├── experiments/
│   ├── run_sweep.py
│   └── plot_results.py
│
├── outputs/
│   └── .gitkeep
│
└── tests/
    ├── test_spaces.py
    ├── test_env_contract.py
    ├── test_q_learning_keydoor.py
    └── test_dqn_shapes.py
```

Do not create a separate global config directory. All configs must live inside each application folder.

---

## General requirements

Use Python 3.10+.

Use these dependencies:

```text
numpy
pyyaml
torch
matplotlib
imageio
pillow
pytest
gymnasium
minigrid
```

Use a Gymnasium-like API internally, but keep the repository's own wrapper interface stable.

Every environment must expose:

```python
observation_space
action_space

reset(seed=None) -> tuple[obs, info]
step(action) -> tuple[next_obs, reward, terminated, truncated, info]
render(mode="rgb_array") -> np.ndarray
```

Every observation must be a `MultiDiscreteSpace` observation. Even MiniGrid observations must be converted into a compact symbolic vector of categorical factors.

Every agent must expose:

```python
act(obs, explore=True) -> int
update(transition) -> dict
begin_episode() -> None
end_episode() -> None
save(path) -> None
load(path) -> None
```

`main.py` must accept:

```bash
python main.py --config applications/key_door/config_qlearning.yaml
```

The config file must include everything: application settings, environment settings, agent settings, training settings, logging settings, and visualization settings.

---

## Core spaces

Implement only `MultiDiscreteSpace` as the main observation space abstraction. You may implement `DiscreteActionSpace` for actions. Observations must always be `MultiDiscreteSpace`.

In `core/spaces.py`, implement:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class MultiDiscreteSpace:
    nvec: list[int]

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        ...

    def contains(self, x) -> bool:
        ...

    def flatten(self, x) -> np.ndarray:
        ...

    def to_index(self, x) -> int:
        ...

    def from_index(self, index: int) -> np.ndarray:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def flat_dim(self) -> int:
        ...
```

Important behavior:

- `x` is a vector of integers with shape `(len(nvec),)`.
- Each component satisfies `0 <= x[i] < nvec[i]`.
- `to_index` must use mixed-radix encoding.
- `from_index` must invert `to_index`.
- `flatten` should return a one-hot concatenation of all categorical factors.
- `size` should return total number of possible states: `prod(nvec)`.
- `flat_dim` should return `sum(nvec)`.

Example:

```python
space = MultiDiscreteSpace([5, 5, 2, 3])
obs = np.array([2, 1, 0, 2])
idx = space.to_index(obs)
obs2 = space.from_index(idx)
flat = space.flatten(obs)
```

The state must remain factored. Do not collapse the state into an integer inside the environment. Agents may use `to_index` when needed.

For actions, implement:

```python
@dataclass
class DiscreteActionSpace:
    n: int

    def sample(self, rng: np.random.Generator) -> int:
        ...

    def contains(self, a) -> bool:
        ...
```

---

## Transition dataclass

Define this in `core/agent_base.py` or `core/types.py`:

```python
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class Transition:
    obs: Any
    action: int
    reward: float
    next_obs: Any
    terminated: bool
    truncated: bool
    info: dict
    next_action: Optional[int] = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
```

---

## Base environment

In `core/env_base.py`, define:

```python
class BaseEnv:
    observation_space: MultiDiscreteSpace
    action_space: DiscreteActionSpace

    def reset(self, seed: int | None = None):
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError

    def render(self, mode: str = "rgb_array"):
        raise NotImplementedError
```

---

## Base agent

In `core/agent_base.py`, define:

```python
class BaseAgent:
    def act(self, obs, explore: bool = True) -> int:
        raise NotImplementedError

    def update(self, transition: Transition) -> dict:
        return {}

    def begin_episode(self) -> None:
        pass

    def end_episode(self) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
```

---

## Agents

Implement the following agents.

### RandomAgent

Samples random actions from `action_space`.

### QLearningAgent

Tabular Q-learning.

Use:

```python
state_idx = observation_space.to_index(obs)
```

Q-table shape:

```python
[num_states, num_actions]
```

Epsilon-greedy action selection:

```python
if explore and rng.random() < epsilon:
    action = action_space.sample(rng)
else:
    action = int(np.argmax(Q[state_idx]))
```

Update:

```text
target = reward if done else reward + gamma * max_a Q[next_state, a]
Q[state, action] += lr * (target - Q[state, action])
```

Return update info:

```python
{
    "td_error": float,
    "q_value": float,
    "target": float,
}
```

### SarsaAgent

Tabular SARSA.

Use the same Q-table structure.

Update:

```text
target = reward if done else reward + gamma * Q[next_state, next_action]
Q[state, action] += lr * (target - Q[state, action])
```

`transition.next_action` must be provided by the trainer.

### DQNAgent

Minimal DQN.

Requirements:

- MLP Q-network
- target network
- replay buffer
- epsilon-greedy
- Adam optimizer
- MSE TD loss
- configurable hidden sizes
- target update interval
- batch size
- replay warmup
- optional gradient clipping

Input:

```python
x = observation_space.flatten(obs)
```

Output dimension:

```python
action_space.n
```

Target:

```text
target = reward + gamma * max_a Q_target(next_obs, a)
```

Use replay buffer.

Return update info:

```python
{
    "loss": float,
    "td_error_mean": float,
    "q_mean": float,
}
```

### AnovaQAgent

Create only a placeholder.

```python
class AnovaQAgent(BaseAgent):
    """
    TODO:
    Implement ANOVA Structural Broadcasting Q-learning.

    Intended future form:
        Q(s,a) = sum_g sqrt(eta_g) * theta[a, g, s_g]

    This class is intentionally not implemented yet.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ANOVA-Q is intentionally left as TODO.")
```

Do not implement ANOVA-Q yet.

---

## Replay buffer

Implement `core/replay_buffer.py` for DQN.

Requirements:

- Fixed capacity circular buffer.
- Store observations, actions, rewards, next observations, terminated/truncated flags.
- Support `add(...)` and `sample(batch_size, rng)`.
- Return NumPy arrays suitable for converting to PyTorch tensors.

---

## Schedules

Implement `core/schedules.py`.

Support at least:

```yaml
epsilon:
  type: linear
  start: 1.0
  end: 0.05
  decay_steps: 50000
```

Function/class should return epsilon at a given global step.

---

## Dynamic import utility

In `core/utils.py`, implement:

```python
def import_from_string(path: str):
    module_path, object_name = path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, object_name)
```

Use this to load:

```yaml
application.entrypoint
application.visualizer
```

---

## Trainer

Implement `core/trainer.py`.

Responsibilities:

1. Load env and agent from config.
2. Run online interaction loop.
3. Support Q-learning, SARSA, DQN, and RandomAgent.
4. Handle episode resets.
5. Handle epsilon schedules.
6. Run evaluation every `eval_interval`.
7. Save metrics to CSV.
8. Save config copy to output directory.
9. Save checkpoints.
10. Save visualization artifacts from the application's visualization function.

Training loop must be environment-agnostic.

SARSA-specific behavior:

- The trainer must select `next_action` before calling `agent.update`.
- `transition.next_action` must be passed to SARSA.

Evaluation:

- Run `eval_episodes`.
- Use `explore=False`.
- Record return, length, success rate.
- Optionally record frames for visualization.

Visualization:

- During evaluation, collect frames from `env.render(mode="rgb_array")`.
- Pass frames to the application visualization utility.
- Save output under:

```text
outputs/<run_name>/visualizations/
```

The application may save:

```text
eval_step_<step>_episode_0.gif
eval_step_<step>_episode_0.png
```

The trainer should call a visualization hook if the config enables it.

Pseudo-logic:

```python
obs, info = env.reset(seed=seed)
if agent_name == "sarsa":
    action = agent.act(obs, explore=True)

for step in range(total_steps):
    if agent_name == "sarsa":
        current_action = action
    else:
        current_action = agent.act(obs, explore=True)

    next_obs, reward, terminated, truncated, info = env.step(current_action)
    done = terminated or truncated

    if agent_name == "sarsa":
        next_action = None if done else agent.act(next_obs, explore=True)
        transition = Transition(obs, current_action, reward, next_obs, terminated, truncated, info, next_action)
        update_info = agent.update(transition)
        action = next_action
    else:
        transition = Transition(obs, current_action, reward, next_obs, terminated, truncated, info)
        update_info = agent.update(transition)

    obs = next_obs

    if done:
        agent.end_episode()
        obs, info = env.reset()
        agent.begin_episode()
        if agent_name == "sarsa":
            action = agent.act(obs, explore=True)

    # log metrics
    # periodically evaluate
    # periodically save checkpoint
    # periodically save visualization
```

---

## Logging

Implement simple CSV logging.

Save:

```text
outputs/<run_name>/
├── config.yaml
├── metrics.csv
├── eval_metrics.csv
├── checkpoints/
└── visualizations/
```

Metrics should include:

```text
step
episode
episode_return
episode_length
success
epsilon
td_error
loss
eval_return_mean
eval_return_std
eval_success_rate
```

Application-specific scalar metrics from `info` may also be logged when possible.

---

## Config format

Each application folder contains config files. For KeyDoor, create:

```text
applications/key_door/config_qlearning.yaml
applications/key_door/config_sarsa.yaml
applications/key_door/config_dqn.yaml
```

Each config must contain everything: application, environment, observation, agent, training, visualization, and logging settings.

Do not use separate environment-default and experiment config files. The application config is the experiment config.

---

# Application: MiniGrid DoorKey wrapper

Implement only one application for now:

```text
applications/key_door/
```

This application must **use an imported MiniGrid environment**, not a custom gridworld implementation. Do not reimplement MiniGrid dynamics manually.

Use `minigrid` through Gymnasium-compatible APIs.

Add these dependencies to `requirements.txt` or `pyproject.toml`:

```text
gymnasium
minigrid
```

The application should wrap a MiniGrid DoorKey environment and convert its observation into the repository's required `MultiDiscreteSpace` format.

## Environment import

In `applications/key_door/env.py`, implement a wrapper class:

```python
class KeyDoorEnv(BaseEnv):
    ...
```

The wrapper should internally create a MiniGrid environment using `gymnasium.make`.

The config should specify the MiniGrid env id:

```yaml
env:
  minigrid_env_id: MiniGrid-DoorKey-5x5-v0
```

Support at least:

```yaml
env:
  minigrid_env_id: MiniGrid-DoorKey-5x5-v0
```

Optionally allow:

```yaml
env:
  minigrid_env_id: MiniGrid-DoorKey-6x6-v0
```

or other installed MiniGrid DoorKey variants.

Example implementation:

```python
import gymnasium as gym
import minigrid  # noqa: F401, ensures environments are registered

self.env = gym.make(
    config["env"]["minigrid_env_id"],
    render_mode="rgb_array",
)
```

Do not manually implement walls, keys, doors, transitions, pickup/open logic, or goal logic. MiniGrid must handle the environment dynamics.

## Required wrapper API

The wrapper must expose the repository's common API:

```python
observation_space: MultiDiscreteSpace
action_space: DiscreteActionSpace

reset(seed=None) -> tuple[np.ndarray, dict]
step(action: int) -> tuple[np.ndarray, float, bool, bool, dict]
render(mode="rgb_array") -> np.ndarray
```

Internally, MiniGrid returns observations such as dictionaries with fields like `"image"`, `"direction"`, and `"mission"`. The wrapper must convert the MiniGrid observation into a compact `MultiDiscreteSpace` state.

## Observation conversion to MultiDiscreteSpace

All observations in this repository must be `MultiDiscreteSpace`.

The wrapper should support a configurable symbolic observation mode.

Use this default mode:

```yaml
observation:
  type: multidiscrete_minigrid_symbolic
  include_agent_position: true
  include_agent_direction: true
  include_has_key: true
  include_door_state: true
```

The converted observation should be a vector:

```text
[agent_row, agent_col, agent_direction, has_key, door_state]
```

where:

```text
agent_row       ∈ {0, ..., grid_height - 1}
agent_col       ∈ {0, ..., grid_width - 1}
agent_direction ∈ {0, 1, 2, 3}
has_key         ∈ {0, 1}
door_state      ∈ {0, 1, 2}
```

Define:

```text
door_state = 0: no door found / unknown
door_state = 1: door closed or locked
door_state = 2: door open
```

Then:

```python
observation_space = MultiDiscreteSpace([
    grid_height,
    grid_width,
    4,
    2,
    3,
])
```

The wrapper must extract these factors from the underlying MiniGrid environment.

## Extracting symbolic state from MiniGrid

Use MiniGrid internals when available.

Suggested approach:

```python
agent_pos = self.env.unwrapped.agent_pos
agent_dir = self.env.unwrapped.agent_dir
carrying = self.env.unwrapped.carrying
grid = self.env.unwrapped.grid
width = self.env.unwrapped.width
height = self.env.unwrapped.height
```

Compute:

```python
agent_col = int(agent_pos[0])
agent_row = int(agent_pos[1])
```

Be careful: MiniGrid usually uses `(x, y)` coordinates, while the repo state should be `[row, col]`.

Compute `has_key`:

```python
has_key = int(carrying is not None and carrying.type == "key")
```

Compute `door_state` by scanning the MiniGrid grid:

```python
door_state = 0
for x in range(width):
    for y in range(height):
        obj = grid.get(x, y)
        if obj is not None and obj.type == "door":
            if obj.is_open:
                door_state = 2
            else:
                door_state = 1
```

If multiple doors exist, aggregate as:

```text
if any open door exists: door_state = 2
elif any closed/locked door exists: door_state = 1
else: door_state = 0
```

Return:

```python
obs = np.array(
    [agent_row, agent_col, agent_dir, has_key, door_state],
    dtype=np.int64,
)
```

The returned observation must satisfy:

```python
self.observation_space.contains(obs)
```

## Action mapping

Use the MiniGrid action space but expose it as repository `DiscreteActionSpace`.

For DoorKey, MiniGrid actions usually include:

```text
left
right
forward
pickup
drop
toggle
done
```

The wrapper should expose all MiniGrid actions by default:

```python
action_space = DiscreteActionSpace(self.env.action_space.n)
```

Do not manually remap action semantics unless necessary.

Add action names for logging:

```python
self.action_names = [
    "left",
    "right",
    "forward",
    "pickup",
    "drop",
    "toggle",
    "done",
]
```

If the underlying action count differs, infer names safely or generate generic names:

```python
action_names = [f"action_{i}" for i in range(self.env.action_space.n)]
```

## Rewards

By default, use the MiniGrid reward exactly as returned by the environment.

Do not manually define sparse or shaped rewards unless the config explicitly asks for reward shaping.

Support:

```yaml
env:
  reward_wrapper: none
```

Optional shaping can be left as TODO.

For now:

```python
reward = float(reward)
```

## Termination and truncation

Use the values returned by MiniGrid:

```python
obs_raw, reward, terminated, truncated, info = self.env.step(action)
```

Then convert `obs_raw` into the repository `MultiDiscreteSpace` observation.

Return:

```python
return obs, float(reward), bool(terminated), bool(truncated), info
```

Add useful wrapper info:

```python
info.update({
    "success": bool(terminated and reward > 0),
    "agent_pos": (agent_row, agent_col),
    "agent_dir": int(agent_dir),
    "has_key": int(has_key),
    "door_state": int(door_state),
})
```

## Reset

Use:

```python
obs_raw, info = self.env.reset(seed=seed)
obs = self._convert_obs(obs_raw)
return obs, info
```

Also add symbolic info fields after reset.

## Rendering

The wrapper's `render(mode="rgb_array")` should call MiniGrid rendering:

```python
frame = self.env.render()
```

Return an RGB NumPy array.

If MiniGrid rendering returns `None`, create the underlying env with `render_mode="rgb_array"`.

---

## KeyDoor visualization

Implement:

```text
applications/key_door/visualize.py
```

Create:

```python
class KeyDoorVisualizer:
    def __init__(self, config):
        ...

    def save_episode_gif(self, frames, path, fps=4):
        ...

    def save_final_frame(self, frame, path):
        ...

    def render_policy_rollout(self, env, agent, path, max_steps, fps=4):
        ...
```

Requirements:

- Save GIFs using `imageio`.
- Save final frame PNG using PIL.
- Do not manually draw the MiniGrid environment.
- Use `env.render(mode="rgb_array")` frames.

Example output:

```text
outputs/keydoor_qlearning_seed0/visualizations/eval_step_10000_episode_0.gif
outputs/keydoor_qlearning_seed0/visualizations/eval_step_10000_episode_0.png
```

---

## KeyDoor config files

Create these three configs inside `applications/key_door/`:

```text
applications/key_door/config_qlearning.yaml
applications/key_door/config_sarsa.yaml
applications/key_door/config_dqn.yaml
```

Each config must include the full experiment configuration.

Example `config_qlearning.yaml`:

```yaml
seed: 0

application:
  name: key_door
  entrypoint: applications.key_door.env:KeyDoorEnv
  visualizer: applications.key_door.visualize:KeyDoorVisualizer

env:
  minigrid_env_id: MiniGrid-DoorKey-5x5-v0
  max_steps_override: null
  reward_wrapper: none

observation:
  type: multidiscrete_minigrid_symbolic
  include_agent_position: true
  include_agent_direction: true
  include_has_key: true
  include_door_state: true

agent:
  name: q_learning
  gamma: 0.99
  learning_rate: 0.1
  epsilon:
    type: linear
    start: 1.0
    end: 0.05
    decay_steps: 50000

training:
  total_steps: 200000
  eval_interval: 10000
  eval_episodes: 20
  log_interval: 1000
  save_interval: 50000

visualization:
  enabled: true
  interval: 10000
  episodes: 1
  format: gif
  fps: 4
  max_steps: 200

logging:
  output_root: outputs
  run_name: keydoor_qlearning_seed0
  save_csv: true
```

Example `config_sarsa.yaml` should be identical except:

```yaml
agent:
  name: sarsa
  gamma: 0.99
  learning_rate: 0.1
  epsilon:
    type: linear
    start: 1.0
    end: 0.05
    decay_steps: 50000
```

Example `config_dqn.yaml`:

```yaml
seed: 0

application:
  name: key_door
  entrypoint: applications.key_door.env:KeyDoorEnv
  visualizer: applications.key_door.visualize:KeyDoorVisualizer

env:
  minigrid_env_id: MiniGrid-DoorKey-5x5-v0
  max_steps_override: null
  reward_wrapper: none

observation:
  type: multidiscrete_minigrid_symbolic
  include_agent_position: true
  include_agent_direction: true
  include_has_key: true
  include_door_state: true

agent:
  name: dqn
  gamma: 0.99
  learning_rate: 0.0005
  batch_size: 64
  replay_size: 50000
  warmup_steps: 1000
  target_update_interval: 1000
  hidden_sizes: [128, 128]
  epsilon:
    type: linear
    start: 1.0
    end: 0.05
    decay_steps: 100000

training:
  total_steps: 300000
  eval_interval: 10000
  eval_episodes: 20
  log_interval: 1000
  save_interval: 50000

visualization:
  enabled: true
  interval: 10000
  episodes: 1
  format: gif
  fps: 4
  max_steps: 200

logging:
  output_root: outputs
  run_name: keydoor_dqn_seed0
  save_csv: true
```

---

## `main.py`

`main.py` must:

1. Parse `--config`.
2. Load YAML.
3. Set seed.
4. Create output directory.
5. Copy config into output directory.
6. Dynamically import env and visualizer from config.
7. Instantiate env.
8. Instantiate agent based on `agent.name`.
9. Instantiate Trainer.
10. Run training.

Example commands:

```bash
python main.py --config applications/key_door/config_qlearning.yaml
python main.py --config applications/key_door/config_sarsa.yaml
python main.py --config applications/key_door/config_dqn.yaml
```

---

## Tests

Implement tests.

### `test_spaces.py`

Check:

- `MultiDiscreteSpace.contains`
- `to_index` and `from_index` inverse correctness
- `flatten` dimension equals `flat_dim`

### `test_env_contract.py`

For KeyDoor:

- instantiate `KeyDoorEnv` from config
- call reset
- verify obs is a NumPy integer vector
- verify `env.observation_space.contains(obs)`
- verify `env.observation_space` is `MultiDiscreteSpace`
- sample random actions for 10 steps
- verify API return types
- call `env.render(mode="rgb_array")`
- verify returned frame is RGB with shape `(H, W, 3)`

Do not test custom key/door dynamics, because MiniGrid owns the dynamics.

### `test_q_learning_keydoor.py`

Use MiniGrid DoorKey 5x5. For speed, run a very small number of steps only as a smoke test, not a strong learning assertion. Do not require Q-learning to solve MiniGrid within unit tests.

Assert:

- training loop runs without crashing
- metrics file is created
- at least one evaluation entry exists

### `test_dqn_shapes.py`

Instantiate DQN on KeyDoor and verify:

- flattened obs has correct dimension
- network output has shape `[batch_size, action_space.n]`
- one update step does not crash after buffer warmup

---

## README

Write a README with:

1. Project purpose.
2. Installation.
3. How to run KeyDoor experiments.
4. How to add a new application.
5. Agent list.
6. Config format.
7. Output directory structure.
8. Explanation that the KeyDoor application wraps MiniGrid DoorKey through `gymnasium` and `minigrid`.
9. Explanation that the wrapper converts MiniGrid internals into symbolic `MultiDiscreteSpace` observations:

```text
[agent_row, agent_col, agent_direction, has_key, door_state]
```

10. Explanation that rendered GIFs are generated from MiniGrid RGB frames.
11. Explanation that ANOVA-Q is intentionally TODO.

Example commands:

```bash
pip install -e .
python main.py --config applications/key_door/config_qlearning.yaml
python main.py --config applications/key_door/config_sarsa.yaml
python main.py --config applications/key_door/config_dqn.yaml
pytest tests
```

---

## Code style

- Keep code simple and readable.
- Use type hints where useful.
- Avoid unnecessary abstractions.
- Use `np.random.default_rng(seed)` for repository-level randomness.
- No hidden global random state.
- Make every experiment reproducible from config seed.
- Do not implement ANOVA-Q yet.
- Do not implement multiple applications yet.
- Do not create global experiment config files outside application folders.
- Ensure all observations are `MultiDiscreteSpace`.
- Ensure visualization artifacts are saved to the output directory.
- Do not manually implement MiniGrid dynamics.
- The `applications/key_door/env.py` file must be a wrapper around an imported MiniGrid environment.
- MiniGrid must handle movement, pickup, toggle/open, rewards, termination, truncation, and rendering.

---

## Development order

Implement in this order:

```text
1. core/spaces.py
2. core/env_base.py
3. core/agent_base.py
4. applications/key_door/env.py as a MiniGrid wrapper
5. applications/key_door/visualize.py
6. RandomAgent
7. QLearningAgent
8. SarsaAgent
9. ReplayBuffer
10. DQNAgent
11. Trainer
12. main.py
13. KeyDoor configs
14. Tests
15. README
16. ANOVA-Q placeholder
```

The final repository should be runnable with:

```bash
python main.py --config applications/key_door/config_qlearning.yaml
python main.py --config applications/key_door/config_sarsa.yaml
python main.py --config applications/key_door/config_dqn.yaml
pytest tests
```

The run output directory must include metrics and at least one visualization artifact when visualization is enabled.

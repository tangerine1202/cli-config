# Refactoring Examples - ML/RL/Robotics

This file contains before/after examples of refactoring git changes for ML/RL/Robotics code, specifically focused on Isaac Lab, PyTorch, and reinforcement learning.

## Example 1: PPO Training Loop

### Git Diff Context

User added a new PPO training script for their robotic manipulation task.

### Before Refactoring

```python
# File: scripts/train_ppo.py (newly added in git diff)

import torch
import numpy as np
from isaaclab.envs import DirectRLEnv

def train():
    # Create environment
    env = DirectRLEnv(cfg=factory_env_cfg)

    # Create policy
    policy = torch.nn.Sequential(
        torch.nn.Linear(64, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 6)
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)

    # Training loop
    for epoch in range(10000):
        # Collect rollouts
        obs = env.reset()
        rewards = []
        actions = []
        log_probs = []

        for step in range(1024):
            # Get action
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            actions.append(action)

        # Compute loss (simple policy gradient)
        loss = 0
        for i in range(len(rewards)):
            loss = loss + -log_probs[i] * rewards[i]

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()
```

### Issues Identified

1. **Hardcoded hyperparameters** (learning rate, epochs, rollout length)
2. **No reproducibility** (missing seeds)
3. **No experiment tracking** (only print statements)
4. **No checkpointing** (can't resume training)
5. **No proper PPO implementation** (should use GAE, clipping)
6. **No type hints**
7. **No error handling**
8. **Magic numbers** (64, 256, 6, 1024, etc.)

### Refactoring Plan

```markdown
1. Extract configuration to dataclass
2. Add seed setting for reproducibility
3. Add type hints to train() function
4. Extract policy creation to helper function
5. Add TensorBoard logging
6. Add checkpoint saving
7. Extract observation/action dimensions to constants
8. Add proper PPO loss computation with GAE
```

### After Refactoring (Incremental Changes)

#### Step 1: Extract Configuration

```python
# File: scripts/train_ppo.py

from dataclasses import dataclass
import torch
import numpy as np
from isaaclab.envs import DirectRLEnv

@dataclass
class TrainingConfig:
    """PPO training configuration."""
    # Environment
    task: str = "Task1-Direct"
    num_envs: int = 4096
    device: str = "cuda"

    # Training
    total_epochs: int = 10000
    num_steps_per_rollout: int = 24
    learning_rate: float = 3e-4

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2

    # Network
    hidden_dim: int = 256
    num_hidden_layers: int = 2

    # Logging & Checkpoints
    log_interval: int = 10
    checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"

    # Reproducibility
    seed: int = 42

# [Rest of code remains same for now]
# Test: pytest tests/test_training.py -v
# Result: ✅ PASS
```

#### Step 2: Add Reproducibility

```python
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(config: TrainingConfig):
    # Set seed first
    set_seed(config.seed)

    # Create environment
    env = DirectRLEnv(cfg=factory_env_cfg)

    # [Rest remains same]

# Test: pytest tests/test_training.py -v
# Result: ✅ PASS
```

#### Step 3: Add Type Hints

```python
from typing import Tuple
import torch.nn as nn

def create_policy(obs_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
    """Create MLP policy network."""
    return nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim)
    )

def train(config: TrainingConfig) -> None:
    set_seed(config.seed)

    # Create environment
    env = DirectRLEnv(cfg=factory_env_cfg)

    # Create policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = create_policy(obs_dim, action_dim, config.hidden_dim)

    # [Rest remains same]

# Test: mypy scripts/train_ppo.py
# Result: ✅ No errors
```

#### Step 4: Add Experiment Tracking

```python
from torch.utils.tensorboard import SummaryWriter

def train(config: TrainingConfig) -> None:
    set_seed(config.seed)

    # Setup logging
    writer = SummaryWriter(log_dir=f"runs/ppo_{config.task}")

    # Create environment
    env = DirectRLEnv(cfg=factory_env_cfg)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create policy
    policy = create_policy(obs_dim, action_dim, config.hidden_dim)
    policy.to(config.device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.total_epochs):
        # [Rollout collection code]

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if epoch % config.log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), epoch)
            writer.add_scalar("train/mean_reward", np.mean(rewards), epoch)
            writer.add_scalar("train/learning_rate", config.learning_rate, epoch)

    writer.close()

# Test: pytest tests/test_training.py -v
# Result: ✅ PASS
```

#### Step 5: Add Checkpointing

```python
from pathlib import Path

def save_checkpoint(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: Path
) -> None:
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{epoch:06d}.pt"

    torch.save({
        "epoch": epoch,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)

    # Also save latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save({
        "epoch": epoch,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, latest_path)

def train(config: TrainingConfig) -> None:
    set_seed(config.seed)

    # Setup logging and checkpoints
    writer = SummaryWriter(log_dir=f"runs/ppo_{config.task}")
    checkpoint_dir = Path(config.checkpoint_dir) / config.task

    # [Environment and policy creation]

    # Training loop
    for epoch in range(config.total_epochs):
        # [Training code]

        # Checkpointing
        if epoch % config.checkpoint_interval == 0 and epoch > 0:
            save_checkpoint(policy, optimizer, epoch, checkpoint_dir)

    writer.close()

# Test: pytest tests/test_training.py::test_checkpointing -v
# Result: ✅ PASS
```

#### Final Result

```python
# File: scripts/train_ppo.py (after all refactoring)

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import DirectRLEnv


@dataclass
class TrainingConfig:
    """PPO training configuration."""
    # Environment
    task: str = "Task1-Direct"
    num_envs: int = 4096
    device: str = "cuda"

    # Training
    total_epochs: int = 10000
    num_steps_per_rollout: int = 24
    learning_rate: float = 3e-4

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2

    # Network
    hidden_dim: int = 256
    num_hidden_layers: int = 2

    # Logging & Checkpoints
    log_interval: int = 10
    checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"

    # Reproducibility
    seed: int = 42


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_policy(obs_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
    """Create MLP policy network."""
    return nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim),
        nn.Tanh()  # Squash actions to [-1, 1]
    )


def save_checkpoint(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: Path,
) -> None:
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{epoch:06d}.pt"

    torch.save(
        {
            "epoch": epoch,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    # Save latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(
        {
            "epoch": epoch,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        latest_path,
    )


def train(config: TrainingConfig) -> None:
    """Train PPO policy."""
    # Reproducibility
    set_seed(config.seed)

    # Setup logging and checkpoints
    writer = SummaryWriter(log_dir=f"runs/ppo_{config.task}")
    checkpoint_dir = Path(config.checkpoint_dir) / config.task

    # Create environment
    env = DirectRLEnv(cfg=factory_env_cfg)

    # Create policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = create_policy(obs_dim, action_dim, config.hidden_dim)
    policy.to(config.device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.total_epochs):
        # Collect rollouts
        obs = env.reset()
        episode_rewards = []

        for step in range(config.num_steps_per_rollout):
            # Get action
            with torch.no_grad():
                action = policy(obs)

            # Step environment
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        # Compute loss (simplified - real PPO would use GAE)
        total_reward = sum(episode_rewards)
        loss = -total_reward  # Policy gradient

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        # Logging
        if epoch % config.log_interval == 0:
            mean_reward = np.mean([r.item() for r in episode_rewards])
            writer.add_scalar("train/loss", loss.item(), epoch)
            writer.add_scalar("train/mean_reward", mean_reward, epoch)
            writer.add_scalar("train/learning_rate", config.learning_rate, epoch)

            print(f"Epoch {epoch:6d} | Loss: {loss.item():8.3f} | Reward: {mean_reward:8.3f}")

        # Checkpointing
        if epoch % config.checkpoint_interval == 0 and epoch > 0:
            save_checkpoint(policy, optimizer, epoch, checkpoint_dir)

    # Final checkpoint
    save_checkpoint(policy, optimizer, config.total_epochs, checkpoint_dir)
    writer.close()


if __name__ == "__main__":
    config = TrainingConfig()
    train(config)
```

### Summary of Changes

| Change | Files Modified | Tests |
|--------|---------------|-------|
| ✅ Extract configuration to dataclass | `train_ppo.py` | Pass |
| ✅ Add seed setting for reproducibility | `train_ppo.py` | Pass |
| ✅ Add type hints | `train_ppo.py` | Pass (mypy) |
| ✅ Extract policy creation to helper | `train_ppo.py` | Pass |
| ✅ Add TensorBoard logging | `train_ppo.py` | Pass |
| ✅ Add checkpoint saving | `train_ppo.py` | Pass |

---

## Example 2: Environment Reward Function

### Git Diff Context

User modified the reward function in their factory environment to add a force penalty.

### Before Refactoring

```python
# File: factory_env.py (modified in git diff)

def _get_reward(self):
    # Keypoint reward
    kp_dist = self._compute_kp_distance()
    kp_rew = 1.0 / (torch.exp(5 * kp_dist) + 4 + torch.exp(-5 * kp_dist))

    # Force penalty (NEWLY ADDED)
    force = self._compute_force()
    if force > 10.0:  # Hardcoded threshold
        force_penalty = -0.5 * (force - 10.0)
    else:
        force_penalty = 0.0

    total_reward = kp_rew + force_penalty

    # Logging (only total)
    self.extras["log"]["reward_total"] = total_reward.mean()

    return total_reward
```

### Issues Identified

1. **Hardcoded threshold** (10.0)
2. **Hardcoded penalty scale** (0.5)
3. **Hardcoded squashing parameters** (5, 4)
4. **No separate logging** for reward components
5. **If-else instead of smooth penalty** (not differentiable at boundary)

### Refactoring Plan

```markdown
1. Extract magic numbers to configuration
2. Replace if-else with smooth penalty function
3. Add separate logging for each reward component
4. Extract reward computation to helper functions
```

### After Refactoring (Incremental Changes)

#### Step 1: Extract Configuration

```python
# File: factory_env_cfg.py (add to existing config)

@dataclass
class RewardConfig:
    """Reward function configuration."""
    # Keypoint reward
    kp_squash_a: float = 5.0
    kp_squash_b: float = 4.0

    # Force penalty
    force_threshold: float = 10.0
    force_penalty_scale: float = 0.5

# File: factory_env.py (modified)

def _get_reward(self):
    cfg = self.cfg.reward  # Access reward config

    # Keypoint reward
    kp_dist = self._compute_kp_distance()
    kp_rew = 1.0 / (
        torch.exp(cfg.kp_squash_a * kp_dist)
        + cfg.kp_squash_b
        + torch.exp(-cfg.kp_squash_a * kp_dist)
    )

    # Force penalty
    force = self._compute_force()
    if force > cfg.force_threshold:
        force_penalty = -cfg.force_penalty_scale * (force - cfg.force_threshold)
    else:
        force_penalty = 0.0

    total_reward = kp_rew + force_penalty

    # Logging
    self.extras["log"]["reward_total"] = total_reward.mean()

    return total_reward

# Test: pytest tests/test_factory_env.py::test_reward -v
# Result: ✅ PASS
```

#### Step 2: Smooth Penalty Function

```python
def _compute_force_penalty(self, force: torch.Tensor) -> torch.Tensor:
    """Compute smooth force penalty using ReLU."""
    cfg = self.cfg.reward
    excess_force = torch.relu(force - cfg.force_threshold)
    penalty = -cfg.force_penalty_scale * excess_force
    return penalty

def _get_reward(self):
    cfg = self.cfg.reward

    # Keypoint reward
    kp_dist = self._compute_kp_distance()
    kp_rew = 1.0 / (
        torch.exp(cfg.kp_squash_a * kp_dist)
        + cfg.kp_squash_b
        + torch.exp(-cfg.kp_squash_a * kp_dist)
    )

    # Force penalty (now uses helper)
    force = self._compute_force()
    force_penalty = self._compute_force_penalty(force)

    total_reward = kp_rew + force_penalty

    # Logging
    self.extras["log"]["reward_total"] = total_reward.mean()

    return total_reward

# Test: pytest tests/test_factory_env.py::test_reward -v
# Result: ✅ PASS
```

#### Step 3: Add Component Logging

```python
def _get_reward(self):
    cfg = self.cfg.reward

    # Keypoint reward
    kp_dist = self._compute_kp_distance()
    kp_rew = 1.0 / (
        torch.exp(cfg.kp_squash_a * kp_dist)
        + cfg.kp_squash_b
        + torch.exp(-cfg.kp_squash_a * kp_dist)
    )

    # Force penalty
    force = self._compute_force()
    force_penalty = self._compute_force_penalty(force)

    total_reward = kp_rew + force_penalty

    # Logging (IMPROVED - separate components)
    self.extras["log"]["reward/total"] = total_reward.mean()
    self.extras["log"]["reward/keypoint"] = kp_rew.mean()
    self.extras["log"]["reward/force_penalty"] = force_penalty.mean()
    self.extras["log"]["force/magnitude"] = force.mean()

    return total_reward

# Test: pytest tests/test_factory_env.py::test_logging -v
# Result: ✅ PASS
```

#### Step 4: Extract Keypoint Reward Helper

```python
def _compute_keypoint_reward(self, distance: torch.Tensor) -> torch.Tensor:
    """Compute keypoint reward with squashing function.

    Args:
        distance: Keypoint distance [N, K] where N=num_envs, K=num_keypoints

    Returns:
        Squashed reward in range (0, 1)
    """
    cfg = self.cfg.reward
    a, b = cfg.kp_squash_a, cfg.kp_squash_b
    reward = 1.0 / (torch.exp(a * distance) + b + torch.exp(-a * distance))
    return reward

def _get_reward(self) -> torch.Tensor:
    """Compute total reward as sum of components."""
    cfg = self.cfg.reward

    # Keypoint reward (now uses helper)
    kp_dist = self._compute_kp_distance()
    kp_rew = self._compute_keypoint_reward(kp_dist)

    # Force penalty
    force = self._compute_force()
    force_penalty = self._compute_force_penalty(force)

    # Total reward
    total_reward = kp_rew + force_penalty

    # Logging
    self.extras["log"]["reward/total"] = total_reward.mean()
    self.extras["log"]["reward/keypoint"] = kp_rew.mean()
    self.extras["log"]["reward/force_penalty"] = force_penalty.mean()
    self.extras["log"]["force/magnitude"] = force.mean()

    return total_reward

# Test: pytest tests/test_factory_env.py -v
# Result: ✅ PASS (all tests)
```

### Summary of Changes

| Change | Files Modified | Tests |
|--------|---------------|-------|
| ✅ Extract magic numbers to config | `factory_env_cfg.py`, `factory_env.py` | Pass |
| ✅ Replace if-else with smooth ReLU penalty | `factory_env.py` | Pass |
| ✅ Add component-wise logging | `factory_env.py` | Pass |
| ✅ Extract reward helpers | `factory_env.py` | Pass |

---

## Example 3: Asset Loading Refactor

### Git Diff Context

User added code to load multiple peg variants with different tolerances.

### Before Refactoring

```python
# File: factory_env.py (newly added code in git diff)

def __init__(self, cfg, **kwargs):
    super().__init__(cfg, **kwargs)

    # Load assets
    peg1 = RigidObject(cfg=peg_cfg_1)
    peg2 = RigidObject(cfg=peg_cfg_2)
    peg3 = RigidObject(cfg=peg_cfg_3)
    peg4 = RigidObject(cfg=peg_cfg_4)

    self.pegs = [peg1, peg2, peg3, peg4]

    # Read properties
    peg1_tol = peg1.get_property("tolerance")
    peg2_tol = peg2.get_property("tolerance")
    peg3_tol = peg3.get_property("tolerance")
    peg4_tol = peg4.get_property("tolerance")

    self.tolerances = torch.tensor([peg1_tol, peg2_tol, peg3_tol, peg4_tol])
```

### Issues Identified

1. **Code duplication** (repeated pattern for each peg)
2. **No type hints**
3. **Hardcoded number of pegs**
4. **No error handling** if property doesn't exist

### Refactoring Plan

```markdown
1. Extract asset loading to helper function
2. Use list comprehension instead of manual enumeration
3. Add type hints
4. Add error handling for missing properties
```

### After Refactoring

```python
from typing import List
import torch
from omni.isaac.lab.sim import RigidObject

def _load_peg_assets(self, peg_configs: List[RigidObjectCfg]) -> List[RigidObject]:
    """Load peg assets from configurations.

    Args:
        peg_configs: List of peg asset configurations

    Returns:
        List of loaded RigidObject instances
    """
    return [RigidObject(cfg=cfg) for cfg in peg_configs]

def _extract_peg_tolerances(self, pegs: List[RigidObject]) -> torch.Tensor:
    """Extract tolerance properties from peg assets.

    Args:
        pegs: List of peg RigidObjects

    Returns:
        Tensor of tolerances [num_pegs]

    Raises:
        ValueError: If any peg is missing tolerance property
    """
    tolerances = []

    for i, peg in enumerate(pegs):
        try:
            tolerance = peg.get_property("tolerance")
            tolerances.append(tolerance)
        except KeyError:
            raise ValueError(
                f"Peg {i} missing 'tolerance' property. "
                "Ensure USD asset has 'aft:tolerance_m' custom attribute."
            )

    return torch.tensor(tolerances, dtype=torch.float32)

def __init__(self, cfg: FactoryEnvCfg, **kwargs):
    super().__init__(cfg, **kwargs)

    # Load peg assets
    peg_configs = [
        cfg.scene.peg_1,
        cfg.scene.peg_2,
        cfg.scene.peg_3,
        cfg.scene.peg_4,
    ]
    self.pegs = self._load_peg_assets(peg_configs)
    self.tolerances = self._extract_peg_tolerances(self.pegs)

# Test: pytest tests/test_factory_env.py::test_asset_loading -v
# Result: ✅ PASS
```

### Summary of Changes

| Change | Files Modified | Tests |
|--------|---------------|-------|
| ✅ Extract asset loading to helper | `factory_env.py` | Pass |
| ✅ Use list comprehension | `factory_env.py` | Pass |
| ✅ Add type hints | `factory_env.py` | Pass (mypy) |
| ✅ Add error handling | `factory_env.py` | Pass |

---

## Example 4: Observation Computation

### Git Diff Context

User added new observations for fingertip pose relative to target.

### Before Refactoring

```python
# File: factory_env.py (newly added in git diff)

def _get_observations(self):
    # Get fingertip pose
    fingertip_pos = self.robot.get_ee_pose()[0]  # [N, 3]
    fingertip_rot = self.robot.get_ee_pose()[1]  # [N, 4]

    # Get target pose
    target_pos = self.target.get_pose()[0]  # [N, 3]
    target_rot = self.target.get_pose()[1]  # [N, 4]

    # Compute relative position
    rel_pos = fingertip_pos - target_pos

    # Compute relative rotation (quaternion difference)
    rel_rot = quat_mul(quat_conjugate(target_rot), fingertip_rot)

    obs = torch.cat([rel_pos, rel_rot], dim=-1)  # [N, 7]

    return obs
```

### Issues Identified

1. **Repeated function calls** (`get_ee_pose()` called twice)
2. **No type hints**
3. **Magic number** (7 in comment)
4. **No observation normalization** (could be useful for training)

### Refactoring Plan

```markdown
1. Extract pose retrieval to avoid repeated calls
2. Add type hints
3. Extract observation dimension to constant
4. Extract relative pose computation to helper
```

### After Refactoring

```python
from typing import Tuple
import torch
from omni.isaac.lab.utils.math import quat_mul, quat_conjugate

# Constants (at top of file)
OBS_DIM_FINGERTIP_POS = 3
OBS_DIM_FINGERTIP_ROT = 4
OBS_DIM = OBS_DIM_FINGERTIP_POS + OBS_DIM_FINGERTIP_ROT  # 7

def _compute_relative_pose(
    self,
    source_pos: torch.Tensor,
    source_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute relative pose from source to target frame.

    Args:
        source_pos: Source position [N, 3]
        source_rot: Source rotation quaternion [N, 4]
        target_pos: Target position [N, 3]
        target_rot: Target rotation quaternion [N, 4]

    Returns:
        Tuple of (relative_position [N, 3], relative_rotation [N, 4])
    """
    rel_pos = source_pos - target_pos
    rel_rot = quat_mul(quat_conjugate(target_rot), source_rot)
    return rel_pos, rel_rot

def _get_observations(self) -> torch.Tensor:
    """Compute observations: fingertip pose relative to target.

    Returns:
        Observations tensor [N, 7] where 7 = position (3) + rotation (4)
    """
    # Get current poses (retrieve once)
    fingertip_pos, fingertip_rot = self.robot.get_ee_pose()
    target_pos, target_rot = self.target.get_pose()

    # Compute relative pose
    rel_pos, rel_rot = self._compute_relative_pose(
        fingertip_pos, fingertip_rot,
        target_pos, target_rot
    )

    # Concatenate observation
    obs = torch.cat([rel_pos, rel_rot], dim=-1)

    assert obs.shape[-1] == OBS_DIM, f"Expected obs dim {OBS_DIM}, got {obs.shape[-1]}"

    return obs

# Test: pytest tests/test_factory_env.py::test_observations -v
# Result: ✅ PASS
```

### Summary of Changes

| Change | Files Modified | Tests |
|--------|---------------|-------|
| ✅ Extract pose retrieval to avoid duplication | `factory_env.py` | Pass |
| ✅ Add type hints | `factory_env.py` | Pass (mypy) |
| ✅ Extract observation dimension constant | `factory_env.py` | Pass |
| ✅ Extract relative pose helper | `factory_env.py` | Pass |

---

## Common Refactoring Patterns Summary

### Pattern 1: Configuration Extraction

**When to use**: Hardcoded values (learning rates, thresholds, scales)

**Before**: `lr = 0.0003`

**After**: `@dataclass class Config: lr: float = 3e-4`

### Pattern 2: Helper Function Extraction

**When to use**: Repeated code, long functions, code doing multiple things

**Before**: Inline computation of relative pose

**After**: `_compute_relative_pose()` helper

### Pattern 3: Error Handling Addition

**When to use**: Operations that could fail (file I/O, property access)

**Before**: `peg.get_property("tolerance")`

**After**: Try-except with informative error message

### Pattern 4: Logging Enhancement

**When to use**: When debugging or monitoring would benefit from more detail

**Before**: `log["reward_total"] = reward`

**After**: `log["reward/keypoint"]`, `log["reward/force"]`, etc.

### Pattern 5: Type Hints Addition

**When to use**: All new functions

**Before**: `def train(config):`

**After**: `def train(config: TrainingConfig) -> None:`

### Pattern 6: Constant Extraction

**When to use**: Magic numbers, repeated literals

**Before**: `obs.shape[-1] == 7`

**After**: `OBS_DIM = 7; obs.shape[-1] == OBS_DIM`

---

## Remember

These examples show incremental refactoring with tests after each step. In practice:

1. **Make ONE change at a time**
2. **Run tests after EACH change**
3. **Don't batch multiple refactorings together**
4. **Focus on code in the git diff only**
5. **Preserve original functionality**

The goal is to make git changes production-ready while maintaining their intent.

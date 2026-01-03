# Refactoring Guidelines - Quick Reference

This document provides quick-reference guidelines for safe, effective refactoring of git changes.

## The Golden Rule

```
ONE CHANGE → RUN TESTS → VERIFY → NEXT CHANGE
```

Never batch multiple refactorings together. Each change should be independent, testable, and reversible.

## Safety Principles

### 1. Never Skip Tests

**ALWAYS run tests after every change**, no matter how small.

```bash
# After EVERY edit, run appropriate tests
pytest tests/test_modified_module.py -v

# Periodically run full suite
pytest tests/ -v

# Type checking
mypy source/

# Linting
ruff check .
```

**If tests fail:**
- ❌ **STOP** - Do not proceed to next change
- 🔍 **ANALYZE** - Understand why it failed
- 🔧 **FIX** - Correct the code or update the test
- ✅ **VERIFY** - Re-run until green
- ➡️ **CONTINUE** - Only then move to next change

### 2. Make Changes Incrementally

**Bad approach (big-bang refactor):**
```python
# Change 1: Extract config
# Change 2: Add type hints
# Change 3: Add logging
# Change 4: Extract helper functions
# Change 5: Add error handling
# → All in one edit → RUN TESTS
# ❌ If tests fail, hard to know which change broke it
```

**Good approach (incremental):**
```python
# Change 1: Extract config → RUN TESTS ✅
# Change 2: Add type hints → RUN TESTS ✅
# Change 3: Add logging → RUN TESTS ✅
# Change 4: Extract helper → RUN TESTS ✅
# Change 5: Add error handling → RUN TESTS ✅
# ✅ Each change verified independently
```

### 3. Keep Working Versions

Mentally note rollback points:

```python
# After each successful change, note:
# "Last working version: Added config dataclass, all tests pass"
#
# If next change breaks something, you know where to revert to
```

### 4. Ask Before Major Structural Changes

**When to ask user:**
- Changing class hierarchy or architecture
- Renaming public APIs
- Moving files or modules
- Removing functionality
- Adding new dependencies

**What to ask:**
- "I'd like to extract this into a separate module for better organization. Is that okay?"
- "This function is public API - should I maintain backward compatibility or is it okay to change the signature?"
- "I noticed unused code - can I remove it or should I leave it?"

## Maintain Functionality Principles

### 1. Refactoring ≠ Behavior Change

**Refactoring**: Improving code structure WITHOUT changing what it does

```python
# ✅ REFACTORING (behavior unchanged)
# Before
def compute(x):
    return x * 2 + 5

# After (more readable, same behavior)
MULTIPLIER = 2
OFFSET = 5
def compute(x):
    return x * MULTIPLIER + OFFSET
```

```python
# ❌ NOT REFACTORING (behavior changed)
# Before
def compute(x):
    return x * 2 + 5

# After (different result!)
def compute(x):
    return x * 3 + 5  # Changed multiplier
```

**Exception**: If user explicitly requests behavior change:
- "Fix the bug in the reward calculation" ✅
- "Change the learning rate schedule" ✅
- "Update the observation space" ✅

### 2. All Tests Should Pass

After refactoring, one of two outcomes:

**Option A: All tests pass (ideal)**
```bash
$ pytest tests/ -v
======================== 42 passed in 2.34s ========================
✅ Refactoring successful, behavior preserved
```

**Option B: Tests fail (needs attention)**

Determine why:

1. **Real bug** (your refactoring broke something):
   ```python
   # You introduced a bug
   def calculate_total(items):
       # BUG: Removed handling of empty list
       return sum(item.price for item in items)  # Crashes on empty!
   ```
   → **FIX THE CODE**

2. **Outdated test** (test expectations need updating):
   ```python
   # Old test expected specific error message
   def test_validation():
       with pytest.raises(ValueError, match="Invalid input"):
           validate(None)

   # You improved error message during refactoring
   # New error message: "Input cannot be None"
   ```
   → **UPDATE THE TEST**

3. **Missing coverage** (refactoring exposed untested code):
   ```python
   # Your refactoring revealed a case that wasn't tested
   # Now the test shows a bug that was always there
   ```
   → **ADD TEST** and **FIX BUG**

### 3. Don't Introduce New Features

**Refactoring scope:**
- ✅ Extract configuration to dataclass
- ✅ Add type hints
- ✅ Improve naming
- ✅ Add logging
- ✅ Extract long function into helpers
- ✅ Add error handling for existing operations

**Feature scope (NOT refactoring):**
- ❌ Add multi-GPU support (new feature)
- ❌ Implement new reward component (new feature)
- ❌ Add wandb integration (new feature)
- ❌ Implement data augmentation (new feature)

**If you identify feature opportunities:**
```markdown
## Refactoring Complete

...

### Future Improvement Opportunities
- Could add multi-GPU support for faster training
- Consider implementing prioritized experience replay
- W&B integration would improve experiment tracking

[Let user decide whether to implement these]
```

### 4. Don't Remove Functionality Without Approval

**Before removing code, ask:**

```python
# You found this in the git diff:
def old_reward_function(state):
    # This seems unused...
    return state.distance * 0.1

# ❌ DON'T automatically delete it
# ✅ DO ask the user:
```

*"I noticed `old_reward_function` appears unused in the current changes. Should I remove it, or is it still needed elsewhere?"*

## Focus and Scope Principles

### 1. Only Refactor Git Changes

**Your scope**: Code in `git diff`

```bash
$ git diff

# Shows changes in:
# - source/training/ppo.py
# - source/envs/factory_env.py

# ✅ Refactor ONLY these two files
# ❌ Don't refactor source/utils/logging.py (not in diff)
```

**Example:**

```python
# Git diff shows you added this function:
def train_policy(env, config):
    policy = create_policy(config)  # ← This line is NEW
    # ... training loop ...

# While reading create_policy(), you notice it's poorly written
# It's in utils/model.py (NOT in git diff)

# ❌ DON'T refactor create_policy() - out of scope
# ✅ DO focus on refactoring train_policy() - in scope
# 💡 OPTIONAL: Mention in final report that create_policy() could be improved
```

### 2. Don't Expand Scope Without Asking

**Scope creep example:**

```
Original task: Refactor PPO training code (in git diff)

❌ While refactoring, you also:
- Refactor the environment code (not in diff)
- Improve the data loader (not in diff)
- Optimize the model architecture (not in diff)
- Add new metrics (feature, not refactor)

✅ Correct approach:
- Refactor ONLY the PPO training code
- Mention other improvement opportunities in final report
- Ask user if they want you to expand scope
```

**How to ask:**

*"While refactoring the training code, I noticed the environment code could also benefit from similar improvements. Would you like me to refactor that as well, or should I stay focused on just the training code?"*

### 3. Don't Improve Unrelated Code

**Scenario**: You're refactoring reward calculation (in git diff)

```python
# File: factory_env.py (in git diff)

def _get_reward(self):
    # Git diff shows changes here ← REFACTOR THIS
    reward = self.kp_reward + self.force_penalty
    return reward

def _reset_scene(self):
    # Not changed in git diff ← DON'T TOUCH THIS
    self.robot.reset()  # You notice this could be improved
    # Resist the urge to refactor it!
```

**Guideline**: If it's not in the diff, leave it alone (unless user explicitly asks).

### 4. Don't Add Unnecessary Abstractions

**Over-engineering example:**

```python
# Git diff shows user added this simple function:
def compute_advantage(rewards, values):
    return rewards - values

# ❌ DON'T over-engineer:
class AdvantageComputationStrategy(ABC):
    @abstractmethod
    def compute(self, rewards, values):
        pass

class SimpleAdvantageStrategy(AdvantageComputationStrategy):
    def compute(self, rewards, values):
        return rewards - values

class GeneralizedAdvantageEstimation(AdvantageComputationStrategy):
    def __init__(self, gamma, lambda_):
        self.gamma = gamma
        self.lambda_ = lambda_
    # ... 50 more lines ...

# ❌ This is over-engineered for a simple subtraction!
```

```python
# ✅ DO keep it simple:
def compute_advantage(rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Compute advantage as rewards - values."""
    return rewards - values

# ✅ Simple, clear, sufficient for the use case
```

**When abstraction IS appropriate:**

```python
# If git diff shows THREE similar reward components:
def kp_reward_baseline(...):
    a, b = 5, 4
    return 1 / (exp(a*x) + b + exp(-a*x))

def kp_reward_coarse(...):
    a, b = 50, 2
    return 1 / (exp(a*x) + b + exp(-a*x))

def kp_reward_fine(...):
    a, b = 100, 0
    return 1 / (exp(a*x) + b + exp(-a*x))

# ✅ Abstraction is justified (DRY principle):
def kp_reward(x, a, b):
    """Keypoint reward with configurable squashing parameters."""
    return 1 / (exp(a*x) + b + exp(-a*x))

kp_baseline = kp_reward(x, a=5, b=4)
kp_coarse = kp_reward(x, a=50, b=2)
kp_fine = kp_reward(x, a=100, b=0)
```

## Code Style Principles

### 1. Follow Existing Conventions

**Match the project's style:**

```python
# Existing codebase uses this style:
class FactoryEnv:
    def _reset_scene(self):
        ...

    def _get_observations(self):
        ...

# ✅ Follow the same pattern:
def _compute_rewards(self):
    ...

# ❌ Don't introduce new style:
def computeRewards(self):  # Wrong: uses camelCase
    ...
```

### 2. Use Language Idioms

**Python idioms:**

```python
# ❌ Non-idiomatic
result = []
for item in items:
    if item.is_valid:
        result.append(item.value)

# ✅ Pythonic (list comprehension)
result = [item.value for item in items if item.is_valid]
```

**PyTorch idioms:**

```python
# ❌ Non-idiomatic (loops)
for i in range(len(rewards)):
    advantages[i] = rewards[i] - values[i]

# ✅ Idiomatic (vectorized)
advantages = rewards - values
```

### 3. Be Consistent

**Within the refactored code:**

```python
# ✅ Consistent naming
def compute_kp_baseline_reward(...)
def compute_kp_coarse_reward(...)
def compute_kp_fine_reward(...)

# ❌ Inconsistent naming
def compute_kp_baseline_reward(...)
def get_kp_coarse_rew(...)
def kp_fine(...)
```

### 4. Match Abstraction Level

**Bad (mixed abstraction levels):**

```python
def train_model(env):
    # High-level
    data = load_training_data()

    # Suddenly low-level (out of place)
    for i in range(len(data)):
        data[i] = (data[i] - mean) / std

    # High-level again
    model = create_model()
    train(model, data)
```

**Good (consistent abstraction):**

```python
def train_model(env):
    # All high-level
    data = load_training_data()
    normalized_data = normalize_data(data)
    model = create_model()
    train(model, normalized_data)

def normalize_data(data):
    # Low-level details hidden here
    return (data - data.mean()) / data.std()
```

## Common Refactoring Patterns

### Pattern 1: Extract Configuration

```python
# Before
def train(env):
    lr = 0.0003
    batch_size = 256
    num_epochs = 1000
    ...

# After
@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_epochs: int = 1000

def train(env, config: TrainingConfig):
    ...
```

### Pattern 2: Extract Helper Function

```python
# Before (long function)
def process_rollouts(rollouts):
    # 30 lines of reward computation
    ...
    # 30 lines of advantage computation
    ...
    # 30 lines of normalization
    ...

# After (extracted helpers)
def process_rollouts(rollouts):
    rewards = compute_rewards(rollouts)
    advantages = compute_advantages(rollouts, rewards)
    normalized_advantages = normalize_advantages(advantages)
    return normalized_advantages

def compute_rewards(rollouts):
    # 30 lines, focused on reward computation
    ...
```

### Pattern 3: Replace Magic Numbers

```python
# Before
if episode_step > 1000:
    terminate = True
if force > 50.0:
    penalty = 0.1

# After
MAX_EPISODE_LENGTH = 1000
FORCE_THRESHOLD = 50.0
FORCE_PENALTY = 0.1

if episode_step > MAX_EPISODE_LENGTH:
    terminate = True
if force > FORCE_THRESHOLD:
    penalty = FORCE_PENALTY
```

### Pattern 4: Add Type Hints

```python
# Before
def train_policy(env, config):
    ...

def compute_loss(rollouts):
    ...

# After
def train_policy(env: gym.Env, config: TrainingConfig) -> nn.Module:
    ...

def compute_loss(rollouts: RolloutBuffer) -> torch.Tensor:
    ...
```

### Pattern 5: Add Error Handling

```python
# Before
def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint

# After
def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    required_keys = ["model_state", "optimizer_state", "epoch"]
    missing_keys = [k for k in required_keys if k not in checkpoint]
    if missing_keys:
        raise ValueError(f"Checkpoint missing keys: {missing_keys}")

    return checkpoint
```

## Testing Guidelines

### When to Run Tests

**After EVERY change:**
```bash
# After adding type hints
mypy source/training.py

# After extracting function
pytest tests/test_training.py::test_train_policy -v

# After configuration refactor
pytest tests/test_training.py -v
```

**Periodically (every 2-3 changes):**
```bash
# Run full test suite
pytest tests/ -v

# Check coverage
pytest tests/ --cov=source --cov-report=term-missing

# Run linter
ruff check .
```

**Before final report:**
```bash
# Everything
pytest tests/ -v
mypy source/
ruff check .
```

### What to Test

**Unit tests** (test individual functions):
```python
def test_compute_advantage():
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 1.0, 1.5])
    expected = torch.tensor([0.5, 1.0, 1.5])

    result = compute_advantage(rewards, values)

    assert torch.allclose(result, expected)
```

**Integration tests** (test components working together):
```python
def test_training_loop():
    env = create_test_env()
    config = TrainingConfig(num_epochs=2)

    policy = train_policy(env, config)

    assert policy is not None
    assert policy.state_dict() is not None
```

**Smoke tests** (test basic functionality):
```python
def test_can_run_training():
    """Just verify training runs without crashing."""
    env = create_test_env()
    config = TrainingConfig(num_epochs=1)

    # Should not raise
    train_policy(env, config)
```

## Checklist Before Reporting Complete

Use this checklist before providing final summary:

### Code Quality
- [ ] All changes applied incrementally (one at a time)
- [ ] Tests run after each change
- [ ] All tests currently passing
- [ ] No new linter warnings
- [ ] Type hints added where applicable
- [ ] Code follows project conventions

### Functionality
- [ ] Core behavior unchanged (unless requested)
- [ ] No features added (only refactoring)
- [ ] No functionality removed (unless approved)
- [ ] Error handling improved (not weakened)

### Scope
- [ ] Only modified code in git diff
- [ ] No unrelated code refactored
- [ ] No unnecessary abstractions added
- [ ] Scope stayed focused

### Documentation
- [ ] Docstrings updated if function signatures changed
- [ ] Comments updated if logic changed
- [ ] Outdated comments removed
- [ ] README updated if public API changed

### Testing
- [ ] All unit tests pass
- [ ] Integration tests pass (if applicable)
- [ ] Manual testing performed (for critical paths)
- [ ] Performance verified (if relevant)

### Final Review
- [ ] `git diff` shows only intended changes
- [ ] No debug code left behind
- [ ] No commented-out code added
- [ ] Committed to report summary

## Remember

**You are refactoring git changes, not rewriting the codebase.**

Focus on making the specific changes in the git diff production-ready while:
- ✅ Maintaining their original intent
- ✅ Preserving functionality
- ✅ Following safety principles
- ✅ Testing continuously
- ✅ Keeping scope focused

**When in doubt**: Ask the user, run tests, make smaller changes.

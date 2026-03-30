# Common Patterns and Anti-Patterns for ML/RL/Robotics

Quick reference guide for recognizing code smells and their best-practice solutions in machine learning, reinforcement learning, and robotics research code.

## ML/RL-Specific Anti-Patterns

### 1. Non-Reproducible Experiments

**Anti-Pattern:**
```python
# No seed setting
model = NeuralNetwork()
train_loader = DataLoader(dataset, shuffle=True)

for epoch in range(100):
    train(model, train_loader)
# Results vary every run!
```

**Best Practice:**
```python
def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For Isaac Lab/Gym envs
    if 'GYM_SEED' in os.environ:
        os.environ['GYM_SEED'] = str(seed)

set_seed(42)
# Now reproducible!
```

### 2. Hardcoded Hyperparameters

**Anti-Pattern:**
```python
# train.py
model = NeuralNetwork(hidden_size=256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    if epoch == 50:
        # Magic number change mid-training
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
```

**Best Practice:**
```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model
    hidden_size: int = 256
    dropout: float = 0.2

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100

    # Scheduling
    scheduler: str = "step"
    step_size: int = 30
    gamma: float = 0.1

    # Reproducibility
    seed: int = 42

config = TrainingConfig()
# Easy to track, modify, and version
```

### 3. No Experiment Tracking

**Anti-Pattern:**
```python
for epoch in range(100):
    loss = train_epoch()
    print(f"Epoch {epoch}: Loss = {loss}")
    # No way to compare runs, visualize progress, or recover results!
```

**Best Practice:**
```python
from torch.utils.tensorboard import SummaryWriter
import wandb  # or MLflow, TensorBoard, etc.

# Initialize tracking
wandb.init(project="my-research", config=config.__dict__)

for epoch in range(100):
    loss = train_epoch()

    # Log to TensorBoard
    writer.add_scalar("train/loss", loss, epoch)

    # Log to W&B
    wandb.log({
        "epoch": epoch,
        "train/loss": loss,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

# Results saved, visualized, and comparable!
```

### 4. Missing Checkpointing

**Anti-Pattern:**
```python
# 10 hours of training...
for epoch in range(1000):
    train()

# Crash at epoch 995! All progress lost!
torch.save(model.state_dict(), "final_model.pt")
```

**Best Practice:**
```python
def save_checkpoint(model, optimizer, epoch, metrics, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.__dict__
    }
    torch.save(checkpoint, path)

for epoch in range(1000):
    metrics = train()

    # Save every N epochs
    if epoch % 10 == 0:
        save_checkpoint(
            model, optimizer, epoch, metrics,
            f"checkpoints/model_epoch_{epoch:04d}.pt"
        )

    # Save best model
    if metrics['val_loss'] < best_val_loss:
        save_checkpoint(
            model, optimizer, epoch, metrics,
            "checkpoints/best_model.pt"
        )
```

### 5. Inefficient Data Loading

**Anti-Pattern:**
```python
# Loading entire dataset into memory
data = np.load("huge_dataset.npy")  # 100GB file!
dataset = TensorDataset(torch.from_numpy(data))

# Single-threaded loading
train_loader = DataLoader(dataset, batch_size=32, num_workers=0)
```

**Best Practice:**
```python
# Lazy loading with memory mapping
class LazyDataset(Dataset):
    def __init__(self, data_path):
        # Memory-map for large files
        self.data = np.load(data_path, mmap_mode='r')

    def __getitem__(self, idx):
        # Load only what's needed
        return torch.from_numpy(self.data[idx].copy())

# Multi-worker loading with pinned memory
train_loader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=8,      # Parallel data loading
    pin_memory=True,    # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

### 6. No Validation Split

**Anti-Pattern:**
```python
# Training on entire dataset
train_loader = DataLoader(full_dataset, batch_size=32)

for epoch in range(100):
    train(model, train_loader)

# No idea if model generalizes or is overfitting!
```

**Best Practice:**
```python
from torch.utils.data import random_split

# Proper train/val/test split
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # Reproducible split
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train with validation
for epoch in range(100):
    train_loss = train(model, train_loader)
    val_loss = validate(model, val_loader)

    if val_loss < best_val_loss:
        save_checkpoint(model, "best.pt")
```

### 7. Ignoring GPU Efficiency

**Anti-Pattern:**
```python
# Moving data one sample at a time
for x, y in dataloader:
    x = x.cuda()  # Inefficient!
    y = y.cuda()

    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
```

**Best Practice:**
```python
# Efficient GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Use pinned memory for faster transfer
train_loader = DataLoader(dataset, pin_memory=True)

for x, y in train_loader:
    # non_blocking for async transfer
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    # Mixed precision training
    with torch.cuda.amp.autocast():
        output = model(x)
        loss = criterion(output, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## RL-Specific Anti-Patterns

### 8. Improper Advantage Estimation

**Anti-Pattern:**
```python
# Naive returns
returns = []
G = 0
for r in reversed(rewards):
    G = r + gamma * G
    returns.insert(0, G)

advantages = returns - values  # No GAE, no normalization
```

**Best Practice:**
```python
def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + values
    return advantages, returns
```

### 9. Not Using Vectorized Environments

**Anti-Pattern:**
```python
# Sequential environment stepping
env = gym.make("CartPole-v1")

for episode in range(1000):
    obs = env.reset()
    done = False

    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        # Extremely slow for RL!
```

**Best Practice:**
```python
# Isaac Lab vectorized environments
from omni.isaac.lab.envs import DirectRLEnv

# 4096 parallel environments!
env = DirectRLEnv(cfg=MyEnvCfg(), num_envs=4096)

obs = env.reset()  # Shape: (4096, obs_dim)

for step in range(1000):
    actions = policy(obs)  # Batched inference
    obs, rewards, dones, infos = env.step(actions)

    # 4096x faster data collection!
```

### 10. Ignoring Observation/Reward Scaling

**Anti-Pattern:**
```python
# Raw observations and rewards
obs = env.reset()  # Values in [-1000, 1000]
reward = env.step(action)[1]  # Values in [0, 10000]

# Network struggles to learn with unscaled inputs!
```

**Best Practice:**
```python
# Normalize observations
class ObservationNormalizer:
    def __init__(self, obs_dim, clip=10.0):
        self.mean = torch.zeros(obs_dim)
        self.var = torch.ones(obs_dim)
        self.count = 0
        self.clip = clip

    def update(self, obs):
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0)
        batch_count = obs.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        self.var = (
            self.var * self.count + batch_var * batch_count +
            delta**2 * self.count * batch_count / total_count
        ) / total_count
        self.count = total_count

    def normalize(self, obs):
        return torch.clamp(
            (obs - self.mean) / torch.sqrt(self.var + 1e-8),
            -self.clip, self.clip
        )

normalizer = ObservationNormalizer(obs_dim)
obs_normalized = normalizer.normalize(obs)
```

## Robotics Simulation Anti-Patterns

### 11. Poorly Designed Observations

**Anti-Pattern:**
```python
# High-dimensional, redundant observations
def get_obs(self):
    obs = torch.cat([
        self.robot.get_joint_positions(),      # 7
        self.robot.get_joint_velocities(),     # 7
        self.robot.get_link_positions(),       # 21 (all links!)
        self.robot.get_link_velocities(),      # 21
        self.target_absolute_position,         # 3
        # 59 dimensions with redundancy!
    ])
    return obs
```

**Best Practice:**
```python
def get_obs(self):
    """Compact, task-relevant observations."""
    # Joint state
    joint_pos = self.robot.data.joint_pos  # 7
    joint_vel = self.robot.data.joint_vel  # 7

    # End-effector (task-relevant)
    ee_pos = self.robot.data.body_pos_w[:, -1, :]  # 3

    # Relative target (better than absolute)
    target_rel = self.target_pos - ee_pos  # 3
    distance = torch.norm(target_rel, dim=1, keepdim=True)  # 1
    direction = target_rel / (distance + 1e-8)  # 3 (normalized)

    obs = torch.cat([
        joint_pos,     # 7
        joint_vel,     # 7
        direction,     # 3 (normalized direction to target)
        distance,      # 1
    ], dim=-1)  # Total: 18 (much better!)

    return obs
```

### 12. Sparse Rewards Only

**Anti-Pattern:**
```python
def compute_reward(self):
    distance = torch.norm(self.ee_pos - self.target_pos, dim=1)

    # Only reward at goal
    reward = (distance < 0.01).float()
    # Extremely hard to learn!
    return reward
```

**Best Practice:**
```python
def compute_reward(self):
    """Dense reward shaping for faster learning."""
    distance = torch.norm(self.ee_pos - self.target_pos, dim=1)

    # Dense distance reward (shaped)
    distance_reward = torch.exp(-5.0 * distance)

    # Sparse success bonus
    success = (distance < 0.01).float()
    success_reward = success * 10.0

    # Action penalty (encourage smooth motions)
    action_penalty = -0.01 * torch.norm(self.actions, dim=1)

    # Velocity penalty (reduce oscillations)
    vel_penalty = -0.001 * torch.norm(self.robot.data.joint_vel, dim=1)

    # Combine
    total_reward = (
        distance_reward +
        success_reward +
        action_penalty +
        vel_penalty
    )

    return total_reward
```

## Universal Anti-Patterns (Any Language)

### 1. Magic Numbers and Hardcoded Values

**Anti-Pattern:**
```python
if user.age > 18:
    if total > 100:
        discount = total * 0.15
```

**Best Practice:**
```python
MINIMUM_AGE = 18
DISCOUNT_THRESHOLD = 100
DISCOUNT_RATE = 0.15

if user.age > MINIMUM_AGE:
    if total > DISCOUNT_THRESHOLD:
        discount = total * DISCOUNT_RATE
```

### 2. God Objects/Functions

**Anti-Pattern:**
```python
class UserManager:
    def handle_user(self, data):
        # Validate
        # Transform
        # Save to DB
        # Send email
        # Log
        # Update cache
        # Trigger webhook
        # ... 500 lines later
```

**Best Practice:**
```python
class UserValidator:
    def validate(self, data): ...

class UserRepository:
    def save(self, user): ...

class UserNotifier:
    def send_welcome_email(self, user): ...

class UserService:
    def __init__(self, validator, repository, notifier):
        self.validator = validator
        self.repository = repository
        self.notifier = notifier

    def create_user(self, data):
        user = self.validator.validate(data)
        saved_user = self.repository.save(user)
        self.notifier.send_welcome_email(saved_user)
        return saved_user
```

### 3. Primitive Obsession

**Anti-Pattern:**
```python
def create_order(user_id: int, amount: float, currency: str):
    if currency not in ["USD", "EUR", "GBP"]:
        raise ValueError("Invalid currency")
    if amount <= 0:
        raise ValueError("Invalid amount")
    # ... logic
```

**Best Practice:**
```python
from dataclasses import dataclass
from enum import Enum

class Currency(Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"

@dataclass(frozen=True)
class Money:
    amount: float
    currency: Currency

    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError("Amount must be positive")

@dataclass(frozen=True)
class UserId:
    value: int

    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("Invalid user ID")

def create_order(user_id: UserId, price: Money):
    # Type system enforces validity
    # ... logic
```

### 4. Deep Nesting

**Anti-Pattern:**
```python
def process(data):
    if data:
        if data.is_valid():
            if data.has_permission():
                if data.is_active():
                    if data.balance > 0:
                        result = data.process()
                        if result:
                            return result.value
    return None
```

**Best Practice (Guard Clauses):**
```python
def process(data):
    if not data:
        return None
    if not data.is_valid():
        return None
    if not data.has_permission():
        return None
    if not data.is_active():
        return None
    if data.balance <= 0:
        return None

    result = data.process()
    return result.value if result else None
```

### 5. Error Swallowing

**Anti-Pattern:**
```python
try:
    user = get_user(user_id)
    user.update(data)
except:
    pass  # Silent failure
```

**Best Practice:**
```python
import logging

logger = logging.getLogger(__name__)

try:
    user = get_user(user_id)
    user.update(data)
except UserNotFoundError as e:
    logger.error(f"User {user_id} not found: {e}")
    raise
except ValidationError as e:
    logger.warning(f"Invalid data for user {user_id}: {e}")
    raise
except Exception as e:
    logger.exception(f"Unexpected error updating user {user_id}")
    raise
```

## Configuration Anti-Patterns

### 1. Hardcoded Configuration

**Anti-Pattern:**
```python
DATABASE_URL = "postgresql://localhost:5432/mydb"
API_KEY = "sk-1234567890abcdef"
DEBUG = True
```

**Best Practice:**
```python
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    api_key: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()
```

### 2. Configuration in Code

**Anti-Pattern:**
```python
if environment == "production":
    workers = 4
    timeout = 30
elif environment == "staging":
    workers = 2
    timeout = 60
else:
    workers = 1
    timeout = 120
```

**Best Practice:**
```yaml
# config/production.yaml
workers: 4
timeout: 30

# config/staging.yaml
workers: 2
timeout: 60

# config/development.yaml
workers: 1
timeout: 120
```

```python
import yaml

def load_config(env: str):
    with open(f"config/{env}.yaml") as f:
        return yaml.safe_load(f)

config = load_config(os.getenv("ENVIRONMENT", "development"))
```

## Testing Anti-Patterns

### 1. Testing Implementation Details

**Anti-Pattern:**
```python
def test_user_creation():
    manager = UserManager()
    # Testing internal implementation
    assert manager._validate_email("test@example.com") == True
    assert manager._hash_password("password") is not None
```

**Best Practice:**
```python
def test_user_creation():
    manager = UserManager()
    # Test public behavior
    user = manager.create_user("test@example.com", "password")
    assert user.email == "test@example.com"
    assert user.can_authenticate("password") == True
```

### 2. Fragile Tests (Over-Mocking)

**Anti-Pattern:**
```python
def test_process_order():
    mock_db = Mock()
    mock_cache = Mock()
    mock_logger = Mock()
    mock_notifier = Mock()
    mock_validator = Mock()

    # Test becomes brittle, tightly coupled to implementation
    service = OrderService(mock_db, mock_cache, mock_logger, mock_notifier, mock_validator)
    ...
```

**Best Practice:**
```python
def test_process_order():
    # Use real objects where possible, mock only external dependencies
    service = OrderService(
        db=in_memory_db(),  # Real in-memory implementation
        notifier=mock_notifier()  # Mock external service
    )
    ...
```

### 3. Non-Deterministic Tests

**Anti-Pattern:**
```python
def test_timestamp():
    result = create_record()
    assert result.created_at == datetime.now()  # Fails randomly
```

**Best Practice:**
```python
from freezegun import freeze_time

@freeze_time("2024-01-01 12:00:00")
def test_timestamp():
    result = create_record()
    assert result.created_at == datetime(2024, 1, 1, 12, 0, 0)
```

## Security Anti-Patterns

### 1. SQL Injection

**Anti-Pattern:**
```python
query = f"SELECT * FROM users WHERE email = '{user_input}'"
cursor.execute(query)
```

**Best Practice:**
```python
query = "SELECT * FROM users WHERE email = %s"
cursor.execute(query, (user_input,))
```

### 2. Missing Input Validation

**Anti-Pattern:**
```python
@app.post("/users")
def create_user(email: str, age: int):
    # No validation, trusts input
    user = User(email=email, age=age)
    db.save(user)
```

**Best Practice:**
```python
from pydantic import BaseModel, EmailStr, Field

class CreateUserRequest(BaseModel):
    email: EmailStr
    age: int = Field(gt=0, lt=150)

@app.post("/users")
def create_user(request: CreateUserRequest):
    user = User(email=request.email, age=request.age)
    db.save(user)
```

### 3. Exposing Internal Errors

**Anti-Pattern:**
```python
@app.get("/users/{id}")
def get_user(id: int):
    try:
        return db.query(User).filter(User.id == id).one()
    except Exception as e:
        return {"error": str(e)}  # Exposes DB schema and internals
```

**Best Practice:**
```python
@app.get("/users/{id}")
def get_user(id: int):
    try:
        return db.query(User).filter(User.id == id).one()
    except NoResultFound:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.exception(f"Error fetching user {id}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Performance Anti-Patterns

### 1. N+1 Queries

**Anti-Pattern:**
```python
users = db.query(User).all()
for user in users:
    # N+1: One query per user
    orders = db.query(Order).filter(Order.user_id == user.id).all()
    user.orders = orders
```

**Best Practice:**
```python
# Single query with join or eager loading
users = db.query(User).options(joinedload(User.orders)).all()
```

### 2. Unnecessary Computation in Loops

**Anti-Pattern:**
```python
results = []
for item in items:
    # Expensive operation called N times
    pattern = compile_regex(pattern_string)
    if pattern.match(item):
        results.append(item)
```

**Best Practice:**
```python
# Compute once, use many times
pattern = compile_regex(pattern_string)
results = [item for item in items if pattern.match(item)]
```

### 3. Loading Everything Into Memory

**Anti-Pattern:**
```python
# Loads 1M records into memory
all_records = db.query(Record).all()
for record in all_records:
    process(record)
```

**Best Practice:**
```python
# Stream/batch process
batch_size = 1000
offset = 0
while True:
    batch = db.query(Record).limit(batch_size).offset(offset).all()
    if not batch:
        break
    for record in batch:
        process(record)
    offset += batch_size
```

## Dependency Management Anti-Patterns

### 1. Tight Coupling

**Anti-Pattern:**
```python
class OrderService:
    def create_order(self, data):
        # Tightly coupled to concrete implementations
        db = PostgreSQLDatabase()
        email = SMTPEmailSender()
        logger = FileLogger()

        order = db.save(Order(data))
        email.send(order.user.email, "Order created")
        logger.log(f"Order {order.id} created")
```

**Best Practice:**
```python
from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    def save(self, entity): pass

class EmailSender(ABC):
    @abstractmethod
    def send(self, to: str, message: str): pass

class OrderService:
    def __init__(self, db: Database, email: EmailSender, logger):
        self.db = db
        self.email = email
        self.logger = logger

    def create_order(self, data):
        order = self.db.save(Order(data))
        self.email.send(order.user.email, "Order created")
        self.logger.info(f"Order {order.id} created")
```

### 2. Service Locator Anti-Pattern

**Anti-Pattern:**
```python
class ServiceLocator:
    _services = {}

    @classmethod
    def get(cls, name):
        return cls._services[name]

class OrderService:
    def create_order(self, data):
        # Hidden dependencies
        db = ServiceLocator.get("database")
        email = ServiceLocator.get("email")
```

**Best Practice:**
```python
# Explicit dependencies via constructor injection
class OrderService:
    def __init__(self, db: Database, email: EmailSender):
        self.db = db
        self.email = email

    def create_order(self, data):
        order = self.db.save(Order(data))
        self.email.send(order.user.email, "Order created")
```

## Logging Anti-Patterns

### 1. Print Debugging Left in Code

**Anti-Pattern:**
```python
def process_payment(amount):
    print(f"Processing {amount}")  # Debug print
    result = charge_card(amount)
    print(f"Result: {result}")  # Debug print
    return result
```

**Best Practice:**
```python
import logging

logger = logging.getLogger(__name__)

def process_payment(amount):
    logger.info(f"Processing payment", extra={"amount": amount})
    result = charge_card(amount)
    logger.info(f"Payment processed", extra={"amount": amount, "result": result})
    return result
```

### 2. Logging Sensitive Data

**Anti-Pattern:**
```python
logger.info(f"User login: {username}, password: {password}")
logger.debug(f"Credit card: {card_number}")
```

**Best Practice:**
```python
logger.info(f"User login attempt", extra={"username": username})
logger.debug(f"Payment processed", extra={"last_4": card_number[-4:]})
```

### 3. Wrong Log Levels

**Anti-Pattern:**
```python
logger.error("User clicked button")  # Not an error
logger.info("Database connection failed")  # Should be error
```

**Best Practice:**
```python
logger.debug("User clicked button")  # Debug/informational
logger.error("Database connection failed", exc_info=True)  # Error with stack trace
```

## Async/Concurrency Anti-Patterns

### 1. Blocking the Event Loop

**Anti-Pattern (Python asyncio):**
```python
async def fetch_data():
    # Blocks event loop
    result = requests.get("https://api.example.com")
    return result.json()
```

**Best Practice:**
```python
import httpx

async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
        return response.json()
```

### 2. Race Conditions

**Anti-Pattern:**
```python
if file_exists(path):
    # TOCTOU: File might be deleted here
    content = read_file(path)
```

**Best Practice:**
```python
try:
    content = read_file(path)
except FileNotFoundError:
    # Handle missing file
    content = default_content
```

## Documentation Anti-Patterns

### 1. Obvious Comments

**Anti-Pattern:**
```python
# Increment i by 1
i = i + 1

# Set user's name to name
user.name = name
```

**Best Practice:**
```python
# Comments explain WHY, not WHAT
# Retry with exponential backoff to handle rate limiting
for attempt in range(max_retries):
    ...
```

### 2. Outdated Documentation

**Anti-Pattern:**
```python
def calculate_discount(price, user):
    """
    Calculate discount based on user tier.

    Args:
        price: Item price
        user_tier: User tier (bronze/silver/gold)  # Parameter doesn't exist!
    """
    # Actually uses user.loyalty_points now
    return price * get_discount_rate(user.loyalty_points)
```

**Best Practice:**
```python
def calculate_discount(price: float, user: User) -> float:
    """
    Calculate discount based on user loyalty points.

    Discount tiers:
    - 0-99 points: 5%
    - 100-499 points: 10%
    - 500+ points: 15%

    Args:
        price: Item price before discount
        user: User object containing loyalty points

    Returns:
        Discounted price
    """
    return price * get_discount_rate(user.loyalty_points)
```

---

## Quick Recognition Checklist

When reviewing code, look for:

**Code Structure:**
- [ ] Functions >50 lines (candidate for extraction)
- [ ] Classes >300 lines (god object)
- [ ] Nesting depth >3 levels (use guard clauses)
- [ ] Repeated code blocks (extract to function)

**Configuration:**
- [ ] Hardcoded URLs, paths, credentials
- [ ] Magic numbers without constants
- [ ] Environment-specific logic in code

**Error Handling:**
- [ ] Bare `except:` or `catch (e) {}`
- [ ] Ignored exceptions
- [ ] No logging on errors
- [ ] Exposing internal errors to users

**Security:**
- [ ] String concatenation in SQL queries
- [ ] No input validation
- [ ] Hardcoded secrets
- [ ] Sensitive data in logs

**Performance:**
- [ ] Queries in loops (N+1)
- [ ] Inefficient algorithms (O(N²) where O(N) possible)
- [ ] Loading large datasets into memory

**Testing:**
- [ ] No tests
- [ ] Tests testing implementation details
- [ ] Tests that require specific execution order

Use this guide to quickly identify anti-patterns and apply appropriate best practices.

# tg-notify Usage Guide

## Basic Message Mode

Send a simple notification:
```bash
tg-notify "Experiment completed"
```

## Wrapper Mode (Execute & Notify)

Run a command and get notified on success/failure:

```bash
# Basic example
tg-notify "exp-1" python3 train.py --epochs 10

# With complex command
tg-notify "data-processing" python3 process.py --input data.csv --output results.csv

# With shell command
tg-notify "backup" tar -czf backup.tar.gz /data

# Sleep test
tg-notify "sleep-test" sleep 5
```

### What happens:
- **Success (exit code 0)**: Notification says "✓ Success: exp-1"
- **Failure (exit code ≠ 0)**: Notification says "✗ Failed (Exit Code: 42): exp-1"

## Python Integration

### Simple notification
```python
from tg_notify import notify

notify("Training complete")
```

### Wrapper mode
```python
from tg_notify import wrap

# Run command and notify on completion
exit_code = wrap("exp-1", ["python3", "train.py", "--epochs", "10"])

# Or with string
exit_code = wrap("exp-1", "python3 train.py --epochs 10")
```

### Decorator mode
```python
from tg_notify import notify_decorator

@notify_decorator
def train_model():
    # your code
    return accuracy

result = train_model()  # Auto-notifies on completion
```

## Examples

### Long-running experiments
```bash
# Train model
tg-notify "bert-training" python3 train.py --model bert --epochs 100

# Data processing
tg-notify "etl-job" python3 etl.py --date 2024-01-01

# Backup
tg-notify "db-backup" pg_dump mydb > backup.sql
```

### From cron jobs
```cron
0 2 * * * tg-notify "nightly-backup" /scripts/backup.sh
```

### Chain multiple tasks
```bash
tg-notify "task-1" ./task1.sh && \
tg-notify "task-2" ./task2.sh && \
tg-notify "task-3" ./task3.sh
```

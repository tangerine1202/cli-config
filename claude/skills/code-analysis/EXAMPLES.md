# Code Analysis Examples

This file contains scenario-specific examples for different types of codebases. The main SKILL.md remains agnostic - use these as references for common patterns in specific domains.


## Machine Learning / Data Science (Python)

### Common Issues to Check

**Structure:**
- Notebook code converted to scripts without refactoring
- No separation between data loading, preprocessing, training
- Hardcoded hyperparameters
- Missing configuration management

**Performance:**
- Not using vectorized operations (NumPy/Pandas)
- Loading entire dataset into memory
- No GPU acceleration where applicable
- Inefficient data pipelines

**Maintainability:**
- No experiment tracking
- Hardcoded file paths
- No reproducibility (missing random seeds)
- No model versioning

**Example Analysis:**

```python
# BAD: Performance issues, not reproducible
def train_model(data_file):
    # Issue 1: Hardcoded path
    df = pd.read_csv('/home/user/data.csv')

    # Issue 2: Inefficient iteration (should be vectorized)
    for i in range(len(df)):
        df.loc[i, 'normalized'] = (df.loc[i, 'value'] - df['value'].mean()) / df['value'].std()

    # Issue 3: No random seed (not reproducible)
    # Issue 4: Hyperparameters hardcoded
    model = RandomForest(n_estimators=100)
    model.fit(df[['feature']], df['target'])

    # Issue 5: No model versioning or tracking
    return model

# GOOD: Vectorized, reproducible, configurable
def train_model(config: ModelConfig, experiment_tracker: ExperimentTracker):
    # Set seed for reproducibility
    np.random.seed(config.random_seed)

    # Use data loader
    df = DataLoader.load(config.data_path)

    # Vectorized operations
    df['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Configuration-driven
    model = RandomForest(**config.model_params)
    model.fit(df[config.features], df[config.target])

    # Track experiment
    experiment_tracker.log_model(model, config, metrics)

    return model
```



## Common Patterns Across All Scenarios

### Magic Numbers
```
BAD:  if (retry_count > 3) ...
GOOD: const MAX_RETRIES = 3; if (retry_count > MAX_RETRIES) ...
```

### Error Handling
```
BAD:  result = risky_operation()  # No error handling
GOOD: try { result = risky_operation() } catch (e) { handle(e) }
```

### Code Duplication
```
BAD:  Same logic copy-pasted in 5 places
GOOD: Extracted to shared function/method
```

### Long Functions
```
BAD:  500-line function doing everything
GOOD: Broken into focused 20-50 line functions
```

### Poor Naming
```
BAD:  def f(x, y): ...
GOOD: def calculate_distance(point_a, point_b): ...
```

---

Use these examples as references when analyzing code, but adapt the principles to the specific language and domain.

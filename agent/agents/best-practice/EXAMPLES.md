# Best Practice Agent - Implementation Examples

Scenario-specific examples showing how to transform MVP code to production-ready code for different technology stacks.


## Machine Learning Pipeline (Python)

### MVP Code

```python
# train.py (MVP)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('/home/user/data.csv')
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained!")
```

### Production-Ready Code

```python
# config/config.py
from pydantic import BaseSettings, FilePath
from pathlib import Path

class Config(BaseSettings):
    # Data
    data_path: FilePath
    train_test_split: float = 0.8

    # Model
    model_type: str = "random_forest"
    n_estimators: int = 100
    max_depth: int = 10
    random_seed: int = 42

    # Features
    feature_columns: list[str] = ["feature1", "feature2", "feature3"]
    target_column: str = "target"

    # Paths
    model_output_dir: Path = Path("models")
    metrics_output_dir: Path = Path("metrics")

    class Config:
        env_file = ".env"

# data/data_loader.py
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path: Path):
        self.data_path = data_path

    def load(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} rows")

        return df

    def validate(self, df: pd.DataFrame, required_columns: list[str]) -> None:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

# preprocessing/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, feature_columns: list[str], target_column: str):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.scaler = StandardScaler()

    def preprocess(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        logger.info("Preprocessing data")

        # Handle missing values
        df = self._handle_missing(df)

        # Extract features and target
        X = df[self.feature_columns].values
        y = df[self.target_column].values

        # Normalize features
        X = self.scaler.fit_transform(X)

        logger.info(f"Preprocessed data shape: X={X.shape}, y={y.shape}")
        return X, y

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fill missing values with median
        for col in self.feature_columns:
            if df[col].isnull().any():
                median = df[col].median()
                df[col].fillna(median, inplace=True)
                logger.warning(f"Filled {df[col].isnull().sum()} missing values in {col}")

        return df

# models/model_factory.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator

class ModelFactory:
    @staticmethod
    def create(model_type: str, **kwargs) -> BaseEstimator:
        models = {
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
        }

        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")

        return models[model_type](**kwargs)

# training/trainer.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: BaseEstimator,
        test_size: float = 0.2,
        random_seed: int = 42
    ):
        self.model = model
        self.test_size = test_size
        self.random_seed = random_seed

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict[str, float]:
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed
        )

        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)}")

        # Train
        self.model.fit(X_train, y_train)

        # Evaluate
        metrics = self._evaluate(X_test, y_test)

        logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.3f}")

        return metrics

    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def save_model(self, output_dir: Path, version: str = None) -> Path:
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"model_{version}.joblib"

        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        return model_path

    def save_metrics(self, metrics: dict, output_dir: Path, version: str = None) -> Path:
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / f"metrics_{version}.json"

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")

        return metrics_path

# train.py
import logging
from config.config import Config
from data.data_loader import DataLoader
from preprocessing.preprocessor import Preprocessor
from models.model_factory import ModelFactory
from training.trainer import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    config = Config()

    logger.info("Starting training pipeline")

    # Load data
    data_loader = DataLoader(config.data_path)
    df = data_loader.load()
    data_loader.validate(df, config.feature_columns + [config.target_column])

    # Preprocess
    preprocessor = Preprocessor(config.feature_columns, config.target_column)
    X, y = preprocessor.preprocess(df)

    # Create model
    model = ModelFactory.create(
        config.model_type,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_seed
    )

    # Train
    trainer = Trainer(
        model,
        test_size=1 - config.train_test_split,
        random_seed=config.random_seed
    )
    metrics = trainer.train(X, y)

    # Save
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save_model(config.model_output_dir, version)
    trainer.save_metrics(metrics, config.metrics_output_dir, version)

    logger.info("Training pipeline complete")

if __name__ == "__main__":
    main()

# tests/test_trainer.py
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from training.trainer import Trainer

def test_trainer_trains_model():
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    trainer = Trainer(model, test_size=0.2, random_seed=42)

    metrics = trainer.train(X, y)

    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1

def test_trainer_reproducible():
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 2, 100)

    model1 = RandomForestClassifier(n_estimators=10, random_state=42)
    trainer1 = Trainer(model1, random_seed=42)
    metrics1 = trainer1.train(X, y)

    model2 = RandomForestClassifier(n_estimators=10, random_state=42)
    trainer2 = Trainer(model2, random_seed=42)
    metrics2 = trainer2.train(X, y)

    assert metrics1["accuracy"] == metrics2["accuracy"]
```

### Improvements Applied

✅ **Configuration Management**
- Pydantic settings
- Environment variables
- No hardcoded values

✅ **Reproducibility**
- Random seeds set
- Version tracking
- Metrics logging

✅ **Architecture**
- Separated concerns (load/preprocess/train)
- Factory pattern for models
- Dependency injection

✅ **Error Handling**
- Validation
- Logging
- Graceful failures

✅ **Testing**
- Unit tests
- Reproducibility tests

---

These examples show the transformation from MVP to production-ready code across different domains. Apply similar patterns for your specific technology stack.

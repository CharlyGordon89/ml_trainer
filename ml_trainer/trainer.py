from abc import ABC, abstractmethod
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from ml_trainer.utils import get_logger

try:
    from ml_evaluator import evaluate_model  # pluggable evaluator module
except ImportError:
    from ml_trainer.evaluator import evaluate_model  # fallback internal version


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        pass

    @abstractmethod
    def save(self, path: str):
        pass


class SklearnTrainer(BaseTrainer):
    
    def __init__(self, model, test_size=0.2, random_state=42, task="classification", logger=None):
        """
        Initialize the trainer with a model and training parameters.

        Args:
        model: Any model with fit/predict methods (e.g., scikit-learn).
        test_size (float): Fraction of data to use for testing.
        random_state (int): Seed for train/test splitting.
        task (str): Type of task ('classification' or 'regression').
        logger (logging.Logger): Optional custom logger.
        """
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.task = task
        self.logger = logger or get_logger(__name__)
        self._is_fitted = False


    def train(self, X, y):
        self.logger.info("Starting training...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        self.logger.info("Training completed.")
        return X_test, y_test

    def evaluate(self, X, y, metrics=None):
        if not self._is_fitted:
            raise ValueError("Model not trained yet")
        self.logger.info("Evaluating model...")
        return evaluate_model(self.model, X, y, task=self.task, logger=self.logger)

    def save(self, path: str):
        self.logger.info(f"Saving model to {path}...")
        joblib.dump(self.model, path)

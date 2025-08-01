# ml_trainer/trainer.py
from datetime import datetime
from ml_logger import get_logger
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class ModelTrainer:
    def __init__(self, model=None, test_size=0.2, random_state=42, save_path="artifacts/models"):
        self.model = model or LogisticRegression()
        self.test_size = test_size
        self.random_state = random_state
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.logger = get_logger("ml_trainer")
        
    def train(self, df: pd.DataFrame, target_col: str):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        self.logger.info(f"Model accuracy: {acc:.4f}")

        model_path = os.path.join(
           self.save_path, 
          f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )
        
        joblib.dump(self.model, model_path)
        
        self.logger.info(f"Model saved to {model_path}")
        self._save_metrics(acc, model_path)

        return acc, model_path


    def _save_metrics(self, accuracy, path="artifacts/metrics"):
       """Persists metrics as JSON for tracking"""
        os.makedirs(path, exist_ok=True)
        metrics = {
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
            "model_path": path
        }
        with open(f"{path}/training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

# ml_trainer/trainer.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class ModelTrainer:
    def __init__(self, model=None, test_size=0.2, random_state=42, save_path="models"):
        self.model = model or LogisticRegression()
        self.test_size = test_size
        self.random_state = random_state
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def train(self, df: pd.DataFrame, target_col: str):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")

        model_path = os.path.join(self.save_path, "model.pkl")
        joblib.dump(self.model, model_path)
        print(f"📦 Model saved to: {model_path}")

        return acc, model_path

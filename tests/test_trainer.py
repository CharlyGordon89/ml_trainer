import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml_trainer.trainer import SklearnTrainer

def test_trainer_trains_and_evaluates():
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6],
        "target": [0, 1, 0, 1, 0, 1]
    })

    X = df[["feature"]]
    y = df["target"]

    model = RandomForestClassifier(random_state=42)
    trainer = SklearnTrainer(model)

    X_test, y_test = trainer.train(X, y)
    results = trainer.evaluate(X_test, y_test)

    assert "accuracy" in results
    assert results["accuracy"] >= 0

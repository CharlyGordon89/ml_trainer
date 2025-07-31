from ml_trainer.trainer import ModelTrainer
import pandas as pd

def test_trainer_works():
    df = pd.DataFrame({
        "age": [22, 25, 47, 52],
        "income": [50000, 60000, 150000, 200000],
        "target": [0, 0, 1, 1]
    })

    trainer = ModelTrainer()
    acc, path = trainer.train(df, target_col="target")

    assert acc >= 0  # Just check it returns a valid score

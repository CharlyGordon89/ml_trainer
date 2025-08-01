import os
import pandas as pd
from ml_trainer.trainer import ModelTrainer

# Sample data
df = pd.DataFrame({
    "feature": [1, 2, 3, 4, 5, 6],
    "target": [0, 1, 0, 1, 0, 1]  # Binary classification
})

# Test the trainer
trainer = ModelTrainer()
acc, model_path = trainer.train(df, "target")

# Verify results
print(f"\n🔍 Verification:")
print(f"1. Model saved to: {model_path}")
print(f"2. Metrics exist: {os.path.exists('artifacts/metrics/training_metrics.json')}")
print(f"3. Directory structure:")
os.system("tree artifacts")  # Show folder structure

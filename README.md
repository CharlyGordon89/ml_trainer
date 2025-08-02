# 🧠 `ml_trainer`

`ml_trainer` is a modular, task-aware training engine designed for scalable machine learning pipelines.  
It supports clean integration with pluggable evaluators, custom loggers, and future trainers beyond scikit-learn — enabling frictionless experimentation and robust production-ready training workflows.

---

## ✅ Key Features

- 🧠 Trains any scikit-learn-compatible model via `SklearnTrainer`
- 🔌 Extensible design for adding new trainers (e.g., XGBoost, PyTorch)
- 🧪 Automatic train/test splitting with configurable random state
- 🎯 Built-in evaluation with task-aware metric selection (classification/regression)
- 📊 Seamless integration with `ml_evaluator` for advanced reporting
- 💾 Model serialization using `joblib`
- 📝 Evaluation metrics saved as JSON with timestamped model path
- 📂 Clean folder structure (`artifacts/models`, `artifacts/metrics`)
- ⚙️ Integrated with `ml_logger` (fallback to stdlib logging if unavailable)
- 🧱 Fully testable and production-grade architecture

---

## 🏗️ Architecture Overview

```text
ml_trainer/
├── trainer.py              # SklearnTrainer + BaseTrainer interface
├── utils.py                # Logger fallback
├── __init__.py             # Public API
├── evaluator.py            # Optional fallback evaluator (can remove if not needed)
tests/
└── test_trainer.py         # Unit test using sklearn model
```

---

## 🧪 Example Usage

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml_trainer import SklearnTrainer

# Sample dataset
df = pd.DataFrame({
    "feature": [1, 2, 3, 4, 5, 6],
    "target": [0, 1, 0, 1, 0, 1]
})
X = df[["feature"]]
y = df["target"]

trainer = SklearnTrainer(
    model=RandomForestClassifier(n_estimators=10),
    test_size=0.2,
    task="classification"
)

X_test, y_test = trainer.train(X, y)
metrics = trainer.evaluate(X_test, y_test)
trainer.save("artifacts/models/my_model.pkl")
```

---

## 🔧 Installation

### 🧑‍💻 Development Mode

```bash
git clone https://github.com/CharlyGordon89/ml_trainer.git
cd ml_trainer
pip install -e .
```

---

## 📦 Requirements

- Python 3.8+
- `scikit-learn`
- `pandas`
- `joblib`
- [`ml_logger`](https://github.com/CharlyGordon89/ml_logger) (optional)
- [`ml_evaluator`](https://github.com/CharlyGordon89/ml_evaluator) (optional)

---

## 🔮 Roadmap

- ✅ Support for classification/regression
- 🟡 Add `XGBoostTrainer`, `LightGBMTrainer`, `TorchTrainer`
- 🟡 CLI interface for training via YAML/JSON config
- 🟡 MLflow or DVC integration for experiment tracking
- 🟡 Distributed training support

---

## 🤝 Contributing

Pull requests, issues, and ideas welcome.  
Let’s make model training modular, professional, and enjoyable.

---

## 🧑‍💻 Authors

Originally by [Charly Gordon](https://github.com/CharlyGordon89)  


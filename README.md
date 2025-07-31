# ml_trainer

`ml_trainer` is a modular and reusable training engine for machine learning pipelines.  
It handles everything from model fitting and evaluation to artifact serialization and saving — allowing you to plug in your model training process with minimal friction.

---

## ✅ Features

- 🧠 Trains any scikit-learn-compatible model
- 🧪 Automatic train/test splitting
- 🎯 Built-in evaluation (accuracy for now)
- 💾 Saves trained models using `joblib`
- ⚙️ Fully testable and easily extensible
- 🧱 Clean module structure, ready to scale

---

## 📦 Installation

```bash
git clone https://github.com/CharlyGordon89/ml_trainer.git
cd ml_trainer
pip install -e .

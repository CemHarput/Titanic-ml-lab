"""
Load a trained model and generate Kaggle-style submission.csv.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from preprocess import load_data, prepare_test_features


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
SUBMISSION_PATH = ROOT_DIR / "submission.csv"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run src/train.py first."
        )

    _, test_df = load_data()
    artifacts = joblib.load(MODEL_PATH)
    preprocessor = artifacts["preprocessor"]
    model = artifacts["model"]

    x_test = prepare_test_features(test_df)
    x_test_processed = preprocessor.transform(x_test)
    test_pred = model.predict(x_test_processed)

    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": test_pred.astype(int),
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved predictions to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()

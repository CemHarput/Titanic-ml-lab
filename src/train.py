"""
Train a Titanic RandomForest model and report validation accuracy.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocess import build_preprocessor, load_data, prepare_train_features


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"


def main() -> None:
    train_df, _ = load_data()
    x_data, y_data = prepare_train_features(train_df)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=42,
        stratify=y_data,
    )

    preprocessor = build_preprocessor()
    x_train_processed = preprocessor.fit_transform(x_train)
    x_valid_processed = preprocessor.transform(x_valid)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=2,
    )
    model.fit(x_train_processed, y_train)

    valid_pred = model.predict(x_valid_processed)
    accuracy = accuracy_score(y_valid, valid_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": preprocessor, "model": model}, MODEL_PATH)
    print(f"Saved trained model to: {MODEL_PATH}")

    class_ratio = np.mean(y_train)
    print(f"Training survival ratio: {class_ratio:.4f}")


if __name__ == "__main__":
    main()

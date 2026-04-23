"""
Data loading and preprocessing helpers for the Titanic project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read train.csv and test.csv from the data folder.
    """
    source_dir = data_dir if data_dir is not None else DATA_DIR
    train_df = pd.read_csv(source_dir / "train.csv")
    test_df = pd.read_csv(source_dir / "test.csv")
    return train_df, test_df


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create FamilySize and IsAlone features.
    """
    updated = df.copy()
    updated["FamilySize"] = updated["SibSp"] + updated["Parch"] + 1
    updated["IsAlone"] = (updated["FamilySize"] == 1).astype(int)
    return updated


def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing steps:
    - fill missing Age/Fare with median
    - fill missing Embarked with most frequent value
    - one-hot encode categorical columns
    """
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def prepare_train_features(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features (X) and target (y) for training.
    """
    if "Survived" not in train_df.columns:
        raise ValueError("train.csv must include the 'Survived' column.")

    processed = add_family_features(train_df)
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"]

    x_data = processed[feature_cols]
    y_data = processed["Survived"]
    return x_data, y_data


def prepare_test_features(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for test-time prediction.
    """
    processed = add_family_features(test_df)
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"]
    return processed[feature_cols]

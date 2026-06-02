from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_COLUMNS = ["location"]
NUMERIC_COLUMNS = ["sqft", "bedrooms", "bathrooms", "age", "garage", "pool", "basement"]


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def detect_and_remove_outliers(df: pd.DataFrame, target_column: str = "price", threshold: float = 3.0) -> pd.DataFrame:
    """Remove outliers using IQR and Z-score methods."""
    df = df.copy()

    Q1 = df[target_column].quantile(0.25)
    Q3 = df[target_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[target_column] >= lower_bound) & (df[target_column] <= upper_bound)]

    z_scores = np.abs((df[target_column] - df[target_column].mean()) / df[target_column].std())
    df = df[z_scores < threshold]

    return df.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values."""
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)

    return df


def scale_numeric_features(df: pd.DataFrame, scaler: StandardScaler | None = None, fit: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric features."""
    df = df.copy()

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[NUMERIC_COLUMNS] = scaler.fit_transform(df[NUMERIC_COLUMNS])
    else:
        df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])

    return df, scaler


def encode_categorical_features(df: pd.DataFrame, encoder: OneHotEncoder | None = None, fit: bool = True) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Encode categorical features."""
    df = df.copy()

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    if fit:
        encoded = encoder.fit_transform(df[CATEGORICAL_COLUMNS])
    else:
        encoded = encoder.transform(df[CATEGORICAL_COLUMNS])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
    df = df.drop(columns=CATEGORICAL_COLUMNS)
    df = pd.concat([df, encoded_df], axis=1)

    return df, encoder


def preprocess(df: pd.DataFrame, target_column: str = "price", fit_scalers: bool = True) -> tuple[pd.DataFrame, pd.Series, StandardScaler, OneHotEncoder]:
    """Complete preprocessing pipeline."""
    df = load_data(df) if isinstance(df, (str, Path)) else df.copy()

    df = detect_and_remove_outliers(df, target_column)
    df = handle_missing_values(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    y = df[target_column].copy()
    X = df.drop(columns=[target_column])

    X, scaler = scale_numeric_features(X, fit=fit_scalers)
    X, encoder = encode_categorical_features(X, fit=fit_scalers)

    return X, y, scaler, encoder

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .config import DEFAULT_INPUT_FILE, DEFAULT_MODEL_FILE, DEFAULT_SCALER_FILE, DEFAULT_ENCODER_FILE, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from .preprocessing import preprocess
from .sample_data import generate_sample_properties


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train property price predictor model")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE, help="Path to property data CSV")
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL_FILE, help="Path to save model")
    parser.add_argument("--output-scaler", type=Path, default=DEFAULT_SCALER_FILE, help="Path to save scaler")
    parser.add_argument("--output-encoder", type=Path, default=DEFAULT_ENCODER_FILE, help="Path to save encoder")
    parser.add_argument("--generate-sample", action="store_true", help="Generate synthetic data if input is missing")
    return parser.parse_args()


def ensure_directories() -> None:
    DEFAULT_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_directories()

    input_path = args.input
    if not input_path.exists():
        if args.generate_sample:
            print(f"Input file not found. Generating sample dataset at {input_path}.")
            generate_sample_properties(input_path)
        else:
            print(f"Input file not found. Generating demo dataset.")
            generate_sample_properties(input_path)

    print("Loading and preprocessing data...")
    X, y, scaler, encoder = preprocess(input_path, target_column=TARGET_COLUMN, fit_scalers=True)

    print(f"Data shape: {X.shape}")
    print(f"Features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("\nEvaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)

    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\nTraining Metrics:")
    print(f"  MAE:  ${train_mae:,.2f}")
    print(f"  RMSE: ${train_rmse:,.2f}")
    print(f"  R²:   {train_r2:.4f}")

    print(f"\nTest Metrics:")
    print(f"  MAE:  ${test_mae:,.2f}")
    print(f"  RMSE: ${test_rmse:,.2f}")
    print(f"  R²:   {test_r2:.4f}")

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"\n5-Fold Cross-Validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\nSaving model and preprocessors...")
    dump(model, args.output_model)
    dump(scaler, args.output_scaler)
    dump(encoder, args.output_encoder)

    print(f"Model saved to: {args.output_model}")
    print(f"Scaler saved to: {args.output_scaler}")
    print(f"Encoder saved to: {args.output_encoder}")

    print("\nTop 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    print(feature_importance.to_string(index=False))


if __name__ == "__main__":
    main()

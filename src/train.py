import argparse
import logging
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train property price predictor model")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--output-scaler", type=Path, default=DEFAULT_SCALER_FILE)
    parser.add_argument("--output-encoder", type=Path, default=DEFAULT_ENCODER_FILE)
    parser.add_argument("--generate-sample", action="store_true")
    return parser.parse_args()


def ensure_directories() -> None:
    DEFAULT_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)


def train(
    input_path: Path = DEFAULT_INPUT_FILE,
    output_model: Path = DEFAULT_MODEL_FILE,
    output_scaler: Path = DEFAULT_SCALER_FILE,
    output_encoder: Path = DEFAULT_ENCODER_FILE,
    generate_sample: bool = False,
):
    ensure_directories()

    if not input_path.exists():
        if generate_sample:
            logger.info(f"Input file not found. Generating sample dataset at {input_path}.")
        else:
            logger.info(f"Input file not found. Generating demo dataset.")
        generate_sample_properties(input_path)

    logger.info("Loading and preprocessing data...")
    X, y, scaler, encoder = preprocess(input_path, target_column=TARGET_COLUMN, fit_scalers=True)

    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info("Training XGBoost model...")
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

    logger.info("Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)

    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    logger.info(f"Training Metrics:  MAE=₹{train_mae:,.2f}, RMSE=₹{train_rmse:,.2f}, R²={train_r2:.4f}")
    logger.info(f"Test Metrics:      MAE=₹{test_mae:,.2f}, RMSE=₹{test_rmse:,.2f}, R²={test_r2:.4f}")

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    logger.info(f"5-Fold CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    logger.info("Saving model and preprocessors...")
    dump(model, output_model)
    dump(scaler, output_scaler)
    dump(encoder, output_encoder)

    logger.info(f"Model saved to {output_model}")
    logger.info(f"Scaler saved to {output_scaler}")
    logger.info(f"Encoder saved to {output_encoder}")

    logger.info("Top 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    logger.info("\n" + feature_importance.to_string(index=False))

    return model, scaler, encoder


def main() -> None:
    args = parse_args()
    train(
        input_path=args.input,
        output_model=args.output_model,
        output_scaler=args.output_scaler,
        output_encoder=args.output_encoder,
        generate_sample=args.generate_sample,
    )


if __name__ == "__main__":
    main()

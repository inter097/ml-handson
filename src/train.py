"""
Fase 3 — Entrenamiento y tracking de experimentos
Referencia: Géron cap. 2 "Select and Train a Model"

Cada ejecución crea un run en MLflow con:
  - parámetros del modelo
  - métricas (RMSE, MAE, R²)
  - el artefacto del modelo serializado

Uso:
    python src/train.py --model linear_regression
    python src/train.py --model random_forest
    python src/train.py --model gradient_boosting
    # o simplemente: make train-all
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = Path("data/processed")

MODELS: dict = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}


def load_data() -> tuple:
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet").values
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet").values
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").values.ravel()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").values.ravel()
    return X_train, X_test, y_train, y_test


def train(model_name: str) -> None:
    if model_name not in MODELS:
        raise ValueError(f"Modelo desconocido: {model_name}. Opciones: {list(MODELS)}")

    X_train, X_test, y_train, y_test = load_data()
    model = MODELS[model_name]

    mlflow.set_experiment("california-housing")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model_type": model_name, **model.get_params()})

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"[{model_name}] RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f} | run_id={run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar un modelo y loggear en MLflow.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help="Nombre del modelo a entrenar",
    )
    args = parser.parse_args()
    train(args.model)

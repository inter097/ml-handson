"""
Fase 3 — Entrenamiento y tracking de experimentos
Referencia: Géron cap. 2 "Select and Train a Model"

Cada modelo se envuelve en un sklearn Pipeline (imputación + escalado +
one-hot + estimador, ver src/preprocessing.py) para que el preprocesamiento
viaje dentro del artefacto serializado: quien cargue el modelo le pasa datos
crudos y el Pipeline los transforma solo.

Cada ejecución crea un run en MLflow con:
  - parámetros del modelo
  - métricas (RMSE, MAE, R²)
  - el artefacto del Pipeline completo serializado

Uso:
    python src/train.py --model linear_regression
    python src/train.py --model random_forest
    python src/train.py --model xgboost
    # o: make train-all
"""
import argparse

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from preprocessing import SKOPS_TRUSTED, build_pipeline, load_data

MODELS: dict = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "decision_tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "extra_trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "xgboost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0),
    "svr": SVR(kernel="rbf", C=10, epsilon=0.1),
    "mlp": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
}


def train(model_name: str) -> None:
    if model_name not in MODELS:
        raise ValueError(f"Modelo desconocido: {model_name}. Opciones: {list(MODELS)}")

    X_train, X_test, y_train, y_test = load_data()
    estimator = MODELS[model_name]
    pipeline = build_pipeline(estimator, X_train)

    mlflow.set_experiment("california-housing")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "model_type": model_name,
            # Distingue corridas con y sin ocean_proximity al comparar en la UI
            "n_features_in": X_train.shape[1],
            "has_ocean_proximity": "ocean_proximity" in X_train.columns,
            **estimator.get_params(),
        })

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.sklearn.log_model(pipeline, artifact_path="model", skops_trusted_types=SKOPS_TRUSTED)

        run_id = mlflow.active_run().info.run_id
        print(f"[{model_name:20s}] RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f} | run_id={run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()
    train(args.model)

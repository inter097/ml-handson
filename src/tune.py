"""
Fase 3b — Búsqueda de hiperparámetros
Referencia: Géron cap. 2 "Fine-Tune Your Model"

RandomizedSearchCV explora el espacio de hiperparámetros con CV=5 para
evitar sobreajuste en la evaluación. Cada combinación probada se loggea
como un run separado en MLflow para comparar todo en la UI.

Por qué RandomizedSearchCV y no GridSearchCV:
  - Grid busca exhaustivamente: con 5 params × 5 valores = 3125 combinaciones
  - Randomized samplea n_iter combinaciones aleatorias: igual de efectivo
    en práctica pero mucho más rápido (Bergstra & Bengio, 2012)

Uso:
    python src/tune.py --model random_forest --n-iter 20
    python src/tune.py --model gradient_boosting --n-iter 20
    # o: make tune-random_forest
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold

PROCESSED_DIR = Path("data/processed")

SEARCH_SPACES: dict = {
    "random_forest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "max_features": ["sqrt", "log2", 0.5, 0.8],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    },
    "gradient_boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "min_samples_leaf": [1, 2, 4],
        },
    },
}


def load_data() -> tuple:
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet").values
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet").values
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").values.ravel()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").values.ravel()
    return X_train, X_test, y_train, y_test


def tune(model_name: str, n_iter: int = 20) -> None:
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"Modelo no soportado para tuning: {model_name}")

    X_train, X_test, y_train, y_test = load_data()
    config = SEARCH_SPACES[model_name]

    mlflow.set_experiment("california-housing")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=config["model"],
        param_distributions=config["params"],
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,  # re-entrena con los mejores params en todo el train set
    )

    print(f"[tune] Explorando {n_iter} combinaciones para {model_name} con CV=5...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    cv_rmse = float(-search.best_score_)

    with mlflow.start_run(run_name=f"{model_name}_tuned"):
        mlflow.log_params({
            "model_type": f"{model_name}_tuned",
            "n_iter": n_iter,
            "cv_folds": 5,
            **search.best_params_,
        })
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "cv_rmse": cv_rmse,   # RMSE promedio en validación cruzada
        })
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        run_id = mlflow.active_run().info.run_id

    print(f"\n[{model_name}_tuned]")
    print(f"  CV RMSE  : {cv_rmse:.4f}  (promedio 5-fold, estimado de generalización)")
    print(f"  Test RMSE: {rmse:.4f}  (resultado real en datos no vistos)")
    print(f"  MAE      : {mae:.4f}")
    print(f"  R²       : {r2:.4f}")
    print(f"  Mejores params: {search.best_params_}")
    print(f"  run_id   : {run_id}")
    print(f"\n  → Compara vs baseline con: make ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Búsqueda de hiperparámetros con RandomizedSearchCV.")
    parser.add_argument("--model", required=True, choices=list(SEARCH_SPACES.keys()))
    parser.add_argument("--n-iter", type=int, default=20, help="Combinaciones a probar (default: 20)")
    args = parser.parse_args()
    tune(args.model, args.n_iter)

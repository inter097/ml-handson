"""
Fase 3b — Búsqueda de hiperparámetros
Referencia: Géron cap. 2 "Fine-Tune Your Model"

RandomizedSearchCV explora el espacio de hiperparámetros con CV=5 para
evitar sobreajuste en la evaluación.

Cada combinación probada se loggea como un run anidado en MLflow (no solo
la ganadora), así que en la UI puedes abrir el run padre y ver las N
combinaciones con su cv_rmse para entender qué movió la aguja y qué no.

Por qué RandomizedSearchCV y no GridSearchCV:
  - Grid busca exhaustivamente: con 5 params × 5 valores = 3125 combinaciones
  - Randomized samplea n_iter combinaciones aleatorias: igual de efectivo
    en práctica pero mucho más rápido (Bergstra & Bengio, 2012)

Por qué el modelo va dentro de un Pipeline:
  el preprocesamiento (imputación, escalado, one-hot) se reajusta dentro de
  cada fold de la CV. Si se aplicara antes de la búsqueda, habría visto los
  folds de validación y el cv_rmse saldría optimista — leakage entre folds.

Uso:
    python src/tune.py --model xgboost --n-iter 30
    python src/tune.py --model random_forest --n-iter 20
    # o: make tune-all
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from preprocessing import SKOPS_TRUSTED, build_pipeline, load_data

SEARCH_SPACES: dict = {
    "ridge": {
        "model": Ridge(),
        "params": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0],
        },
    },
    "decision_tree": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", None],
        },
    },
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
    "extra_trees": {
        "model": ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "max_features": ["sqrt", "log2", 0.5, 0.8],
            "max_depth": [None, 10, 20, 30],
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
    "xgboost": {
        "model": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        "params": {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6, 7],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 0.5, 1.0],
            "reg_lambda": [1, 1.5, 2.0],
        },
    },
    "svr": {
        "model": SVR(),
        "params": {
            "kernel": ["rbf", "poly"],
            "C": [0.1, 1, 10, 100],
            "epsilon": [0.01, 0.1, 0.5],
            "gamma": ["scale", "auto"],
        },
    },
    "mlp": {
        "model": MLPRegressor(max_iter=500, random_state=42),
        "params": {
            "hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128), (128, 64, 32)],
            "activation": ["relu", "tanh"],
            "learning_rate_init": [0.0001, 0.001, 0.01],
            "alpha": [0.0001, 0.001, 0.01],
        },
    },
}


def _log_all_candidates(search: RandomizedSearchCV) -> None:
    """Guarda cada combinación probada como run anidado del run activo.

    RandomizedSearchCV deja todo en cv_results_ y luego lo tira; sin esto
    solo sobreviviría la ganadora y se perdería el resto de la búsqueda.
    """
    results = search.cv_results_
    for i in range(len(results["params"])):
        params = {k.removeprefix("model__"): v for k, v in results["params"][i].items()}
        with mlflow.start_run(run_name=f"candidato_{i:02d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({
                "cv_rmse": float(-results["mean_test_score"][i]),
                "cv_rmse_std": float(results["std_test_score"][i]),
                "rank": int(results["rank_test_score"][i]),
            })


def tune(model_name: str, n_iter: int = 20) -> None:
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"Modelo no soportado para tuning: {model_name}. Opciones: {list(SEARCH_SPACES)}")

    X_train, X_test, y_train, y_test = load_data()
    config = SEARCH_SPACES[model_name]

    # El preprocesamiento se reajusta en cada fold; los params van al paso "model"
    pipeline = build_pipeline(config["model"], X_train)
    param_dist = {f"model__{k}": v for k, v in config["params"].items()}

    mlflow.set_experiment("california-housing")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print(f"[tune] {model_name}: probando {n_iter} combinaciones con CV=5...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    cv_rmse = float(-search.best_score_)

    # Quitar el prefijo "model__" para que la UI muestre nombres limpios
    best_params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}

    with mlflow.start_run(run_name=f"{model_name}_tuned"):
        mlflow.log_params({
            "model_type": f"{model_name}_tuned",
            "n_iter": n_iter,
            "cv_folds": 5,
            "n_features_in": X_train.shape[1],
            "has_ocean_proximity": "ocean_proximity" in X_train.columns,
            **best_params,
        })
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2, "cv_rmse": cv_rmse})
        mlflow.sklearn.log_model(best_model, artifact_path="model", skops_trusted_types=SKOPS_TRUSTED)
        run_id = mlflow.active_run().info.run_id
        _log_all_candidates(search)

    print(f"[{model_name}_tuned] CV={cv_rmse:.4f} | Test RMSE={rmse:.4f} | R²={r2:.4f} | run_id={run_id}")
    print(f"  Mejores params: {best_params}")
    print(f"  {len(search.cv_results_['params'])} candidatos guardados como runs anidados")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(SEARCH_SPACES.keys()))
    parser.add_argument("--n-iter", type=int, default=20)
    args = parser.parse_args()
    tune(args.model, args.n_iter)

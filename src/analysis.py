"""
Análisis de modelos entrenados
Referencia: Géron cap. 2 + cap. 4 "Training Models"

Genera dos tipos de análisis y los guarda en reports/:

1. Importancia de features — qué variables influyen más en las predicciones
   - Modelos con feature_importances_ (árboles): uso directo
   - Modelos con coef_ (lineales): valor absoluto del coeficiente
   - SVR y MLP: permutation importance (más lento pero universal)

2. Curvas de aprendizaje — ¿el modelo mejora con más datos?
   - Si train >> val → overfitting (necesitas más datos o regularización)
   - Si ambas curvas son bajas → underfitting (modelo muy simple)
   - Si convergen → el modelo está bien calibrado

Uso:
    python src/analysis.py
    # o: make analysis
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import learning_curve, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

# Modelos para análisis (subconjunto representativo para no tardar horas)
MODELS_FOR_ANALYSIS: dict = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0),
    "SVR": SVR(kernel="rbf", C=10, epsilon=0.1),
    "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
}

# Solo estos para curvas de aprendizaje (SVR es muy lento con n grande)
MODELS_FOR_CURVES: dict = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0),
    "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
}


def load_data() -> tuple:
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").values.ravel()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").values.ravel()
    feature_names = list(X_train.columns)
    return X_train.values, X_test.values, y_train, y_test, feature_names


def plot_feature_importance() -> None:
    print("\n[analysis] Calculando importancia de features...")
    X_train, X_test, y_train, y_test, feature_names = load_data()

    n_models = len(MODELS_FOR_ANALYSIS)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(MODELS_FOR_ANALYSIS.items()):
        print(f"  [{idx+1}/{n_models}] {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            # Permutation importance para SVR y MLP
            result = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
            )
            importances = result.importances_mean

        order = np.argsort(importances)
        ax = axes[idx]
        bars = ax.barh(
            [feature_names[i] for i in order],
            importances[order],
            color=plt.cm.viridis(importances[order] / importances.max()),
        )
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Importancia")
        ax.tick_params(labelsize=8)
        print("✓")

    # Ocultar subplots vacíos
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Importancia de Features por Modelo", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = REPORTS_DIR / "feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Guardado → {out}")


def plot_learning_curves() -> None:
    print("\n[analysis] Calculando curvas de aprendizaje (puede tardar ~2 min)...")
    X_train, _, y_train, _, _ = load_data()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 8)

    n_models = len(MODELS_FOR_CURVES)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=False)

    for ax, (name, model) in zip(axes, MODELS_FOR_CURVES.items()):
        print(f"  {name}...", end=" ", flush=True)
        sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        # Convertir a RMSE positivo
        train_rmse = -train_scores.mean(axis=1)
        val_rmse = -val_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_std = val_scores.std(axis=1)

        ax.plot(sizes, train_rmse, "o-", color="#2196F3", label="Train RMSE")
        ax.plot(sizes, val_rmse, "s--", color="#F44336", label="Val RMSE")
        ax.fill_between(sizes, train_rmse - train_std, train_rmse + train_std, alpha=0.1, color="#2196F3")
        ax.fill_between(sizes, val_rmse - val_std, val_rmse + val_std, alpha=0.1, color="#F44336")

        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Tamaño del train set")
        ax.set_ylabel("RMSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Diagnóstico automático
        gap = val_rmse[-1] - train_rmse[-1]
        if gap > 0.05:
            diagnostico = "⚠ Overfitting"
        elif val_rmse[-1] > 0.55:
            diagnostico = "⚠ Underfitting"
        else:
            diagnostico = "✓ Bien calibrado"
        ax.set_title(f"{name}\n{diagnostico}", fontsize=10, fontweight="bold")
        print("✓")

    fig.suptitle("Curvas de Aprendizaje — Train vs Validación", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = REPORTS_DIR / "learning_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analysis] Guardado → {out}")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_importance()
    plot_learning_curves()
    print("\n✓ Análisis completo. Archivos en reports/:")
    print("  - feature_importance.png")
    print("  - learning_curves.png")


if __name__ == "__main__":
    main()

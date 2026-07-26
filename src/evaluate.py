"""
Fase 4 — Evaluación final del mejor modelo
Referencia: Géron cap. 2 "Fine-Tune Your Model" + "Evaluate on Test Set"

Flujo:
  1. Correr `make train-all`
  2. Comparar runs en `make ui` (mlflow ui → http://localhost:5000)
  3. Copiar el run_id del mejor modelo
  4. Correr `make evaluate RUN_ID=<id>`
  5. El reporte se guarda en reports/ → commitear con `git add reports/`

Uso:
    python src/evaluate.py --run-id <mlflow_run_id>
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


def evaluate(run_id: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet").values
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").values.ravel()

    preds = model.predict(X_test)
    residuals = y_test - preds

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    # ── Reporte en markdown ──────────────────────────────────────────────────
    model_name = type(model).__name__
    report = f"""# Evaluación Final — {model_name}

**run_id:** `{run_id}`

| Métrica | Valor |
|---------|-------|
| RMSE    | {rmse:.4f} |
| MAE     | {mae:.4f} |
| R²      | {r2:.4f} |

> RMSE en unidades originales: USD {rmse * 100_000:,.0f} promedio de error por casa.

![Residuos](evaluation.png)
"""
    (REPORTS_DIR / "evaluation.md").write_text(report)

    # ── Gráficas ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} — run {run_id[:8]}", fontsize=12)

    axes[0].scatter(preds, residuals, alpha=0.2, s=5)
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Predicciones")
    axes[0].set_ylabel("Residuos")
    axes[0].set_title("Residuos vs Predicciones")

    axes[1].hist(residuals, bins=60, edgecolor="none")
    axes[1].set_xlabel("Residuo")
    axes[1].set_title("Distribución de Residuos")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "evaluation.png", dpi=150)
    plt.close()

    print(f"\n=== {model_name} ===")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"\nReporte guardado en reports/")
    print("Commitea con: git add reports/ && git commit -m 'eval: <modelo> RMSE={:.4f}'".format(rmse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar el mejor modelo de un run de MLflow.")
    parser.add_argument("--run-id", required=True, help="MLflow run ID (cópialo de `make ui`)")
    args = parser.parse_args()
    evaluate(args.run_id)

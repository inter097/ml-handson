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
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
CONFIDENCE = 0.95


def rmse_confidence_interval(y_true, preds, confidence: float = CONFIDENCE) -> tuple:
    """Intervalo de confianza del RMSE (Géron §2 "Evaluate on the Test Set").

    El RMSE del test set sale de una muestra concreta de casas; con otra
    muestra habría dado distinto. El intervalo acota cuánto puede moverse.

    Se calcula sobre los errores CUADRADOS (no sobre el RMSE directo) porque
    el RMSE es una raíz de una media: se saca el intervalo de esa media y al
    final se le aplica la raíz a los dos extremos.
    """
    squared_errors = (y_true - preds) ** 2
    n = len(squared_errors)
    low, high = stats.t.interval(
        confidence,
        df=n - 1,
        loc=squared_errors.mean(),
        scale=stats.sem(squared_errors),
    )
    return float(np.sqrt(low)), float(np.sqrt(high))


def evaluate(run_id: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    import os, mlflow.artifacts, joblib, tempfile
    # Descarga el artefacto y carga con joblib (compatible con todos los modelos)
    with tempfile.TemporaryDirectory() as tmp:
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="model", dst_path=tmp
        )
        model_file = next(Path(tmp).rglob("model.pkl"), None) or \
                     next(Path(tmp).rglob("*.pkl"), None) or \
                     next(Path(tmp).rglob("*.joblib"), None)
        if model_file is None:
            # fallback: usar mlflow.pyfunc
            import mlflow.pyfunc
            pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            class _Wrapper:
                def predict(self, X): return pyfunc_model.predict(X)
            model = _Wrapper()
        else:
            model = joblib.load(model_file)

    # DataFrame, no array: el Pipeline necesita los nombres de columna para
    # separar las numéricas de ocean_proximity.
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").values.ravel()

    preds = model.predict(X_test)
    residuals = y_test - preds

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    ci_low, ci_high = rmse_confidence_interval(y_test, preds)

    # ── Nombre del modelo ────────────────────────────────────────────────────
    # El artefacto puede cargarse via pyfunc (envuelto en _Wrapper) y perder su
    # tipo real, así que se toma de los params del run.
    run = mlflow.get_run(run_id)
    model_name = (
        run.data.params.get("model_type")
        or run.info.run_name
        or type(model).__name__
    )

    # ── Reporte en markdown ──────────────────────────────────────────────────
    report = f"""# Evaluación Final — {model_name}

**run_id:** `{run_id}`

| Métrica | Valor |
|---------|-------|
| RMSE    | {rmse:.4f} |
| MAE     | {mae:.4f} |
| R²      | {r2:.4f} |
| IC {CONFIDENCE:.0%} del RMSE | [{ci_low:.4f}, {ci_high:.4f}] |

> RMSE en unidades originales: USD {rmse * 100_000:,.0f} promedio de error por casa.
> Con {CONFIDENCE:.0%} de confianza, el error real está entre USD {ci_low * 100_000:,.0f}
> y USD {ci_high * 100_000:,.0f}.

**Cómo leer el intervalo:** otro modelo solo es mejor de verdad si su RMSE queda
fuera de este rango. Si dos modelos tienen intervalos que se traslapan, la
diferencia entre ellos cabe dentro del ruido del muestreo y no se puede afirmar
que uno gane.

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
    print(f"  RMSE : {rmse:.4f}  (IC {CONFIDENCE:.0%}: [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"\nReporte guardado en reports/")
    print("Commitea con: git add reports/ && git commit -m 'eval: <modelo> RMSE={:.4f}'".format(rmse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar el mejor modelo de un run de MLflow.")
    parser.add_argument("--run-id", required=True, help="MLflow run ID (cópialo de `make ui`)")
    args = parser.parse_args()
    evaluate(args.run_id)

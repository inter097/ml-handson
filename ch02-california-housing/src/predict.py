"""
Fase 5 — Inferencia
Referencia: Géron cap. 2 "Launch, Monitor, and Maintain Your System"

Carga un modelo entrenado desde MLflow y predice sobre datos crudos.

Este script es la prueba de que el refactor del Pipeline sirvió: recibe un CSV
con las columnas originales — sin escalar, sin imputar, sin one-hot, con NaN si
los hay — y el artefacto se encarga de todo. No hay que replicar aquí ningún
paso de preprocesamiento, que es justo donde se rompen los despliegues reales
(el famoso training/serving skew).

Uso:
    # Predecir sobre un CSV propio
    python src/predict.py --run-id <id> --input casas.csv

    # Demo rápida con 5 filas del test set
    python src/predict.py --run-id <id> --demo

Las columnas requeridas son las mismas que produce `make features`:
    MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
    Latitude, Longitude, ocean_proximity,
    rooms_per_household, bedrooms_ratio, population_per_household
"""
import argparse
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.artifacts
import mlflow.pyfunc
import pandas as pd

PROCESSED_DIR = Path("data/processed")


def load_model(run_id: str):
    """Descarga el artefacto del run y lo carga como objeto de sklearn."""
    with tempfile.TemporaryDirectory() as tmp:
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="model", dst_path=tmp
        )
        model_file = (
            next(Path(tmp).rglob("model.pkl"), None)
            or next(Path(tmp).rglob("*.pkl"), None)
            or next(Path(tmp).rglob("*.joblib"), None)
        )
        if model_file is not None:
            return joblib.load(model_file)

    return mlflow.pyfunc.load_model(f"runs:/{run_id}/model")


def predict(run_id: str, input_path: str | None, demo: bool, output: str | None) -> None:
    model = load_model(run_id)

    if demo:
        X = pd.read_parquet(PROCESSED_DIR / "X_test.parquet").head(5)
        y_true = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").head(5).values.ravel()
    else:
        X = pd.read_csv(input_path)
        y_true = None

    preds = model.predict(X)

    result = X.copy()
    result["prediccion_usd"] = preds * 100_000
    if y_true is not None:
        result["real_usd"] = y_true * 100_000
        result["error_usd"] = result["prediccion_usd"] - result["real_usd"]

    cols = ["MedInc", "Latitude", "Longitude", "ocean_proximity", "prediccion_usd"]
    cols = [c for c in cols if c in result.columns]
    if y_true is not None:
        cols += ["real_usd", "error_usd"]

    print(f"\n=== Predicciones (run {run_id[:8]}) ===")
    print(result[cols].to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    if output:
        result.to_csv(output, index=False)
        print(f"\nGuardado en {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir con un modelo de MLflow.")
    parser.add_argument("--run-id", required=True, help="MLflow run ID (cópialo de `make ui`)")
    parser.add_argument("--input", help="CSV con datos crudos a predecir")
    parser.add_argument("--demo", action="store_true", help="Usar 5 filas del test set")
    parser.add_argument("--output", help="Guardar las predicciones en este CSV")
    args = parser.parse_args()

    if not args.demo and not args.input:
        parser.error("Se requiere --input <csv> o --demo")

    predict(args.run_id, args.input, args.demo, args.output)

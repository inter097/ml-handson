"""
Generador de datos para el sitio

Toma un run de MLflow, predice sobre el conjunto de prueba y deja esas
predicciones como JSON en web/public/data/, donde el proyecto Astro las
consume para el mapa interactivo del capítulo 2.

Antes este script producía un HTML completo por su cuenta. Ahora solo produce
los datos: la página la construye Astro, que es lo que corresponde cuando el
sitio tiene varias páginas compartiendo diseño.

Por qué las predicciones van precalculadas y no vía API:
  el artefacto de XGBoost pesa 2.4 MB pero sus dependencias (numpy, scipy,
  scikit-learn, pandas) suman 262 MB, así que no caben en un runtime
  serverless. Precalcular las 4,128 predicciones del test set cuesta 127 KB
  y elimina el backend por completo.

Uso:
    python src/build_site.py --run-id <mlflow_run_id>
    # o: make site RUN_ID=<id>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predict import load_model
from preprocessing import load_data

# El sitio vive en la raíz del repo, un nivel arriba del capítulo.
RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "web" / "public" / "data" / "ch02.json"

OCEAN_ORDER = ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
INCOME_BINS = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf]


def build(run_id: str) -> None:
    _, X_test, _, y_test = load_data()
    model = load_model(run_id)
    preds = model.predict(X_test)

    rmse = float(np.sqrt(((y_test - preds) ** 2).mean()))

    ocean_idx = {name: i for i, name in enumerate(OCEAN_ORDER)}
    income_cat = pd.cut(X_test["MedInc"], bins=INCOME_BINS, labels=[1, 2, 3, 4, 5]).astype(int)

    # Arrays en vez de objetos: mismo contenido, ~60% menos bytes.
    # 3 decimales bastan — el mapa no distingue más y el precio va en cientos de miles.
    rows = [
        [round(float(lat), 3), round(float(lon), 3), round(float(real), 3),
         round(float(pred), 3), ocean_idx[ocean], int(cat)]
        for lat, lon, real, pred, ocean, cat in zip(
            X_test["Latitude"], X_test["Longitude"], y_test, preds,
            X_test["ocean_proximity"], income_cat,
        )
    ]

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(
        {"ocean": OCEAN_ORDER, "cols": ["lat", "lon", "real", "pred", "ocean", "inc"],
         "rows": rows, "run_id": run_id, "rmse": round(rmse, 4)},
        separators=(",", ":"),
    ), encoding="utf-8")

    kb = SALIDA.stat().st_size / 1024
    print(f"[site] modelo del run {run_id[:8]} · RMSE {rmse:.4f}")
    print(f"[site] {len(rows):,} predicciones → {SALIDA.relative_to(RAIZ)} ({kb:.0f} KB)")
    print("[site] construye el sitio con: cd web && npm run build")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar los datos del sitio.")
    parser.add_argument("--run-id", required=True, help="MLflow run ID del modelo a mostrar")
    args = parser.parse_args()
    build(args.run_id)

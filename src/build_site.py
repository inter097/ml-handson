"""
Generador del caso de estudio estático

Toma un run de MLflow, predice sobre el conjunto de prueba e inyecta esas
predicciones dentro de site/template.html para producir site/index.html.

El resultado es un solo archivo sin dependencias externas: se sirve desde
cualquier hosting estático (Vercel, GitHub Pages, nginx en un VPS) sin build,
sin backend y sin variables de entorno.

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

SITE_DIR = Path("site")
TEMPLATE = SITE_DIR / "template.html"
OUTPUT = SITE_DIR / "index.html"
PLACEHOLDER = "__DATA__"

# El template guarda solo el contenido; aquí se envuelve en un documento válido.
# Sin <!doctype html> el navegador entra en quirks mode y box-sizing deja de
# comportarse como dice el CSS.
DOC_OPEN = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
"""
DOC_MID = """</head>
<body>
"""
DOC_CLOSE = """</body>
</html>
"""

# El orden fija los índices que usa el JavaScript de la página.
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

    payload = json.dumps(
        {"ocean": OCEAN_ORDER, "cols": ["lat", "lon", "real", "pred", "ocean", "inc"], "rows": rows},
        separators=(",", ":"),
    )

    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"{TEMPLATE} no contiene el marcador {PLACEHOLDER}")
    body = template.replace(PLACEHOLDER, payload)

    # Las etiquetas de <head> viven al inicio del template; se cortan ahí para
    # colocarlas donde corresponde en el documento final.
    split = body.index("<style>")
    head, rest = body[:split], body[split:]
    OUTPUT.write_text(DOC_OPEN + head + rest.replace("</style>", "</style>" + DOC_MID, 1) + DOC_CLOSE,
                      encoding="utf-8")

    kb = OUTPUT.stat().st_size / 1024
    print(f"[site] modelo del run {run_id[:8]} · RMSE {rmse:.4f}")
    print(f"[site] {len(rows):,} predicciones embebidas ({len(payload)/1024:.0f} KB)")
    print(f"[site] {OUTPUT} · {kb:.0f} KB en total")
    print(f"[site] pruébalo con: python -m http.server -d {SITE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar el caso de estudio estático.")
    parser.add_argument("--run-id", required=True, help="MLflow run ID del modelo a mostrar")
    args = parser.parse_args()
    build(args.run_id)

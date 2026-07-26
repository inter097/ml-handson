"""
Fase 1 — Obtener datos
Referencia: Géron cap. 2 "Get the Data"

Descarga el dataset California Housing de scikit-learn y lo guarda en
data/raw/housing.parquet para que el resto del pipeline lo consuma.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

RAW_DIR = Path("data/raw")


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    housing = fetch_california_housing(as_frame=True)
    df: pd.DataFrame = housing.frame
    out = RAW_DIR / "housing.parquet"
    df.to_parquet(out, index=False)
    print(f"[data] {len(df):,} filas → {out}")
    print(f"[data] columnas: {list(df.columns)}")


if __name__ == "__main__":
    download()

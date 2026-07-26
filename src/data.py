"""
Fase 1 — Obtener datos
Referencia: Géron cap. 2 "Get the Data"

Descarga el dataset California Housing de scikit-learn y lo guarda en
data/raw/housing.parquet para que el resto del pipeline lo consuma.

Fallback: si el entorno no tiene acceso a internet, genera un dataset
sintético con la misma estructura para que el pipeline pueda correr.
En tu máquina local siempre usará el dataset real.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
COLUMNS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
           "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"]


def _synthetic_fallback(n: int = 20_640, seed: int = 42) -> pd.DataFrame:
    """Genera datos con la misma estructura y correlaciones aproximadas."""
    rng = np.random.default_rng(seed)
    med_inc = np.abs(rng.normal(3.8, 1.9, n))
    house_age = rng.uniform(1, 52, n)
    ave_rooms = np.abs(rng.normal(5.4, 2.5, n))
    ave_bedrms = np.clip(ave_rooms / rng.uniform(4, 7, n), 0.5, 3.0)
    population = np.abs(rng.normal(1425, 1132, n))
    ave_occup = np.abs(rng.normal(3.1, 10, n)).clip(1, 20)
    lat = rng.uniform(32.5, 42.0, n)
    lon = rng.uniform(-124.4, -114.3, n)
    # Target correlacionado con ingresos, latitud y habitaciones
    target = (
        0.45 * med_inc
        + 0.01 * house_age
        + 0.05 * ave_rooms
        - 0.003 * population / ave_occup
        + rng.normal(0, 0.5, n)
    ).clip(0.15, 5.0)
    return pd.DataFrame({
        "MedInc": med_inc, "HouseAge": house_age, "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms, "Population": population, "AveOccup": ave_occup,
        "Latitude": lat, "Longitude": lon, "MedHouseVal": target,
    })


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "housing.parquet"

    try:
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing(as_frame=True)
        df: pd.DataFrame = housing.frame
        source = "California Housing (real)"
    except Exception:
        print("[data] Sin acceso a internet — usando dataset sintético")
        df = _synthetic_fallback()
        source = "sintético (misma estructura)"

    df.to_parquet(out, index=False)
    print(f"[data] {source}: {len(df):,} filas → {out}")
    print(f"[data] columnas: {list(df.columns)}")


if __name__ == "__main__":
    download()

"""
Fase 1 — Obtener datos
Referencia: Géron cap. 2 "Get the Data"

Descarga el CSV original del libro (ageron/data) y lo guarda en
data/raw/housing.parquet para que el resto del pipeline lo consuma.

Por qué el CSV del libro y no `fetch_california_housing` de scikit-learn:
sklearn entrega una versión recortada — descarta la columna categórica
`ocean_proximity` e imputa silenciosamente los 207 valores faltantes de
`total_bedrooms`. El libro enseña justamente a tratar esas dos cosas
(OneHotEncoder e imputación), así que necesitamos el CSV crudo.

Las columnas se renombran al estilo de sklearn (MedInc, HouseAge, ...) y el
target se divide entre 100,000 para que las métricas sean comparables con
las corridas anteriores del proyecto.
"""
import io
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
BOOK_URL = "https://github.com/ageron/data/raw/main/housing.tgz"


def _load_book_csv() -> pd.DataFrame:
    """Descarga housing.tgz del repo del libro y extrae housing.csv."""
    csv_cache = RAW_DIR / "housing_book.csv"

    if csv_cache.exists():
        print(f"[data] usando CSV cacheado: {csv_cache}")
        return pd.read_csv(csv_cache)

    print(f"[data] descargando {BOOK_URL} ...")
    with urllib.request.urlopen(BOOK_URL, timeout=60) as resp:
        payload = resp.read()

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("housing.csv"))
        df = pd.read_csv(tar.extractfile(member))

    csv_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_cache, index=False)
    return df


def _to_sklearn_schema(book: pd.DataFrame) -> pd.DataFrame:
    """Convierte los conteos crudos del CSV a los promedios que usa sklearn.

    Se conserva `ocean_proximity`, que sklearn descarta, y los NaN de
    `total_bedrooms` se dejan intactos — los imputa el Pipeline del modelo.
    """
    return pd.DataFrame({
        "MedInc": book["median_income"],
        "HouseAge": book["housing_median_age"],
        "AveRooms": book["total_rooms"] / book["households"],
        "AveBedrms": book["total_bedrooms"] / book["households"],
        "Population": book["population"],
        "AveOccup": book["population"] / book["households"],
        "Latitude": book["latitude"],
        "Longitude": book["longitude"],
        "ocean_proximity": book["ocean_proximity"],
        "MedHouseVal": book["median_house_value"] / 100_000,
    })


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "housing.parquet"

    try:
        df = _to_sklearn_schema(_load_book_csv())
        source = "California Housing — CSV del libro (con ocean_proximity)"
    except Exception as exc:
        # Fallback a sklearn: son los mismos datos pero SIN ocean_proximity.
        # Se avisa fuerte porque cambia qué features ve el modelo.
        print(f"[data] ⚠ No se pudo obtener el CSV del libro ({exc})")
        print("[data] ⚠ Usando fetch_california_housing — SIN ocean_proximity")
        from sklearn.datasets import fetch_california_housing
        df = fetch_california_housing(as_frame=True).frame
        source = "California Housing vía sklearn (sin ocean_proximity)"

    df.to_parquet(out, index=False)

    n_missing = int(df.isna().sum().sum())
    print(f"[data] {source}: {len(df):,} filas → {out}")
    print(f"[data] columnas: {list(df.columns)}")
    print(f"[data] valores faltantes: {n_missing} (los imputa el Pipeline)")
    if "ocean_proximity" in df.columns:
        print(f"[data] ocean_proximity: {df['ocean_proximity'].value_counts().to_dict()}")


if __name__ == "__main__":
    download()

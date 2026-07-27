"""
Fase 2 — Limpieza y Feature Engineering
Referencia: Géron cap. 2 "Prepare the Data for ML Algorithms"

Lee data/raw/housing.parquet, construye features derivadas, divide en
train/test y guarda los splits SIN ESCALAR en data/processed/.

Por qué sin escalar: el escalado vive dentro del sklearn Pipeline de cada
modelo (src/train.py, src/tune.py). Si se escalara aquí, el StandardScaler
vería el train set completo antes de la validación cruzada y filtraría
información entre folds — el propio leakage que el libro enseña a evitar.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/housing.parquet")
PROCESSED_DIR = Path("data/processed")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def build() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(RAW_PATH)

    # ── Feature engineering (Géron §2) ──────────────────────────────────────
    # Ratios más informativos que los valores absolutos
    df["rooms_per_household"] = df["AveRooms"] / df["AveOccup"]
    df["bedrooms_ratio"] = df["AveBedrms"] / df["AveRooms"]
    df["population_per_household"] = df["Population"] / df["AveOccup"]

    # ── Split ────────────────────────────────────────────────────────────────
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ── Guardar sin escalar (el scaler vive en el Pipeline del modelo) ───────
    X_train.to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(PROCESSED_DIR / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet", index=False)

    n_missing = int(X_train.isna().sum().sum() + X_test.isna().sum().sum())
    print(f"[features] train={len(X_train):,} | test={len(X_test):,}")
    print(f"[features] features={list(X.columns)}")
    print(f"[features] faltantes={n_missing} (los imputa el Pipeline)")


if __name__ == "__main__":
    build()

"""
Fase 2 — Limpieza y Feature Engineering
Referencia: Géron cap. 2 "Prepare the Data for ML Algorithms"

Lee data/raw/housing.parquet, construye features derivadas, escala,
divide en train/test y guarda los splits en data/processed/.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    # ── Escalado (fit solo en train para evitar data leakage) ────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns
    )

    # ── Guardar ──────────────────────────────────────────────────────────────
    X_train_scaled.to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
    X_test_scaled.to_parquet(PROCESSED_DIR / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet", index=False)
    joblib.dump(scaler, PROCESSED_DIR / "scaler.joblib")

    print(f"[features] train={len(X_train):,} | test={len(X_test):,}")
    print(f"[features] features={list(X.columns)}")


if __name__ == "__main__":
    build()

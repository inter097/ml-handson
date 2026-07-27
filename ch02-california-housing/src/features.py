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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/housing.parquet")
PROCESSED_DIR = Path("data/processed")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cortes de ingreso para estratificar (Géron §2 "Create a Test Set").
# MedInc es el predictor más fuerte del precio, así que el test set tiene que
# respetar su distribución o las métricas miden una California que no existe.
INCOME_BINS = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf]
INCOME_LABELS = [1, 2, 3, 4, 5]


def _report_stratification(income_cat: pd.Series, X_train, X_test) -> None:
    """Muestra el sesgo que la estratificación evita.

    Compara la proporción de cada categoría de ingreso en la población contra
    la del test set, y contra la que habría dado un split aleatorio puro.
    """
    full = income_cat.value_counts(normalize=True).sort_index()
    strat = income_cat.loc[X_test.index].value_counts(normalize=True).sort_index()

    _, X_test_rnd = train_test_split(
        X_train.index.append(X_test.index).to_frame(),
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    rnd = income_cat.loc[X_test_rnd.index].value_counts(normalize=True).sort_index()

    print("[features] distribución de MedInc por categoría (% del total)")
    print("           cat   población   estratif.   aleatorio   sesgo aleat.")
    for cat in INCOME_LABELS:
        sesgo = (rnd[cat] - full[cat]) / full[cat] * 100
        print(f"             {cat}     {full[cat]:7.2%}     {strat[cat]:7.2%}"
              f"     {rnd[cat]:7.2%}      {sesgo:+6.2f}%")


def build() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(RAW_PATH)

    # ── Feature engineering (Géron §2) ──────────────────────────────────────
    # Ratios más informativos que los valores absolutos
    df["rooms_per_household"] = df["AveRooms"] / df["AveOccup"]
    df["bedrooms_ratio"] = df["AveBedrms"] / df["AveRooms"]
    df["population_per_household"] = df["Population"] / df["AveOccup"]

    # ── Split estratificado por categoría de ingreso ─────────────────────────
    # Con un split aleatorio puro, el test set puede quedar con más casas ricas
    # que la población real y el RMSE mediría un país que no existe. Estratificar
    # fuerza que train y test tengan la misma proporción de cada categoría.
    income_cat = pd.cut(df["MedInc"], bins=INCOME_BINS, labels=INCOME_LABELS)

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=income_cat
    )

    _report_stratification(income_cat, X_train, X_test)

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

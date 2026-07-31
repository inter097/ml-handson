"""
La selección de variables del libro, sobre el modelo del libro

Ejercicio 3 de Géron: añadir un `SelectFromModel` al pipeline de preparación.
Aquí ya estaba resuelto, pero sobre otro modelo y otros datos: un
`RandomForest` con las 16,512 filas y umbral `"median"`. El enunciado lo monta
sobre el SVR de 5,000 filas con umbral absoluto 0.005, así que la cifra
publicada respondía a una pregunta parecida y distinta.

Este script mide la versión del libro, con el SVR que ganó la rejilla del
ejercicio 1 (rbf, C=3, gamma=0.1), y de paso repite la medición sobre
random_forest para que las dos sean comparables entre sí.

Uso:
    python src/select_study.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

from preprocessing import PROCESSED_DIR, build_pipeline

REPORTS_DIR = Path("reports")
OUT = REPORTS_DIR / "select_study.json"

N_TRAIN = 5_000     # las del enunciado
N_SPLITS = 3
N_JOBS = 2          # la búsqueda del ejercicio 2 sigue ocupando los otros núcleos
UMBRAL_LIBRO = 0.005
MEJOR_SVR = dict(kernel="rbf", C=3.0, gamma=0.1)


def medir(nombre, estimador, X, y, cv, **kwargs) -> dict:
    pipe = build_pipeline(estimador, X, **kwargs)
    t0 = time.perf_counter()
    scores = cross_val_score(pipe, X, y, cv=cv, n_jobs=N_JOBS,
                             scoring="neg_root_mean_squared_error")
    segundos = time.perf_counter() - t0
    fila = {"caso": nombre, "cv_rmse": float(-scores.mean()),
            "cv_std": float(scores.std()), "seconds": segundos, **kwargs}
    print(f"  {nombre:44s} RMSE {fila['cv_rmse']:.4f} ± {fila['cv_std']:.4f}"
          f"  {segundos:6.1f}s", flush=True)
    return fila


def columnas_que_sobreviven(estimador, X, y, umbral) -> tuple:
    """Cuántas columnas pasan el umbral, que es la mitad del resultado."""
    pipe = build_pipeline(estimador, X, select_features=True, select_threshold=umbral)
    pipe.fit(X, y)
    selector = pipe.named_steps["select"]
    return int(selector.get_support().sum()), int(len(selector.get_support()))


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").values.ravel()
    Xs, ys = X_train.iloc[:N_TRAIN], y_train[:N_TRAIN]
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    filas = []
    print(f"[select] La versión del libro: SVR sobre {N_TRAIN:,} filas, umbral {UMBRAL_LIBRO}")
    filas.append(medir("SVR sin selección", SVR(**MEJOR_SVR), Xs, ys, cv))
    filas.append(medir(f"SVR con SelectFromModel({UMBRAL_LIBRO})", SVR(**MEJOR_SVR), Xs, ys, cv,
                       select_features=True, select_threshold=UMBRAL_LIBRO))
    filas.append(medir('SVR con SelectFromModel("median")', SVR(**MEJOR_SVR), Xs, ys, cv,
                       select_features=True, select_threshold="median"))

    print("\n[select] El mismo umbral sobre random_forest, para comparar")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    filas.append(medir("random_forest sin selección", rf, Xs, ys, cv))
    filas.append(medir(f"random_forest con SelectFromModel({UMBRAL_LIBRO})", rf, Xs, ys, cv,
                       select_features=True, select_threshold=UMBRAL_LIBRO))

    vivas_libro, total = columnas_que_sobreviven(SVR(**MEJOR_SVR), Xs, ys, UMBRAL_LIBRO)
    vivas_median, _ = columnas_que_sobreviven(SVR(**MEJOR_SVR), Xs, ys, "median")
    print(f"\n[select] Columnas que sobreviven: {vivas_libro} de {total} con umbral "
          f"{UMBRAL_LIBRO}, {vivas_median} de {total} con \"median\"")

    OUT.write_text(json.dumps({
        "n_train": N_TRAIN, "cv": N_SPLITS, "umbral_libro": UMBRAL_LIBRO,
        "svr": MEJOR_SVR, "filas": filas,
        "columnas": {"total": total, "umbral_libro": vivas_libro, "median": vivas_median},
    }, indent=2))
    print(f"[select] Guardado → {OUT}")


if __name__ == "__main__":
    main()

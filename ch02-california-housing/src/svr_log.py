"""
Cuánto pesa el logaritmo que el pipeline del capítulo no tenía

El libro aplica `np.log` a las columnas de cola larga antes de estandarizarlas.
El pipeline de aquí no lo hacía, y la razón era implícita: el capítulo se armó
alrededor de modelos de árbol, que parten por umbrales y no se enteran de una
transformación monótona. Un SVR sí: mide distancias, y una columna con
asimetría 94 (AveOccup) domina el kernel.

Este script mide la diferencia en lugar de suponerla. Recorre dos veces la
rama rbf de la rejilla oficial del ejercicio 1, idéntica salvo por el
logaritmo, y compara candidato a candidato.

Solo la rama rbf: son 42 candidatos y termina en minutos, mientras que los
ocho del kernel lineal costarían dos horas por el valor alto de C. La pregunta
es si el logaritmo mueve el resultado, y eso ya se ve aquí.

Uso:
    python src/svr_log.py
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

from preprocessing import build_pipeline, columnas_sesgadas, load_data

REPORTS_DIR = Path("reports")
OUT = REPORTS_DIR / "svr_log.json"

N_TRAIN = 5_000
N_SPLITS = 3
# Un solo proceso a propósito: la rejilla oficial puede estar corriendo en
# paralelo, y dos trabajos peleándose los núcleos falsean los tiempos de ambos.
# Aquí lo que importa es el RMSE, no el reloj.
N_JOBS = 1

GRID_C = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
GRID_GAMMA = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, _, y_train, _ = load_data()
    Xs, ys = X_train.iloc[:N_TRAIN], y_train[:N_TRAIN]

    log_cols = columnas_sesgadas(Xs)
    print("[log] Columnas que reciben el logaritmo (asimetría > 1 y mínimo positivo):")
    for c in log_cols:
        print(f"  {c:26s} asimetría {Xs[c].skew():7.2f}")

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    filas = []
    t0 = time.perf_counter()
    for C in GRID_C:
        for gamma in GRID_GAMMA:
            fila = {"C": C, "gamma": gamma}
            for etq, use_log in (("sin_log", False), ("con_log", True)):
                pipe = build_pipeline(SVR(kernel="rbf", C=C, gamma=gamma),
                                      Xs, use_log=use_log)
                scores = cross_val_score(pipe, Xs, ys, cv=cv, n_jobs=N_JOBS,
                                         scoring="neg_root_mean_squared_error")
                fila[etq] = float(-scores.mean())
            fila["delta"] = fila["con_log"] - fila["sin_log"]
            filas.append(fila)
            print(f"  C={C:7g} gamma={gamma:5g}  sin={fila['sin_log']:.4f}  "
                  f"con={fila['con_log']:.4f}  Δ={fila['delta']:+.4f}", flush=True)

    mejor_sin = min(filas, key=lambda f: f["sin_log"])
    mejor_con = min(filas, key=lambda f: f["con_log"])
    ganancias = sum(1 for f in filas if f["delta"] < 0)

    print(f"\n[log] Mejor sin log: C={mejor_sin['C']:g} gamma={mejor_sin['gamma']:g} "
          f"→ {mejor_sin['sin_log']:.4f}")
    print(f"[log] Mejor con log: C={mejor_con['C']:g} gamma={mejor_con['gamma']:g} "
          f"→ {mejor_con['con_log']:.4f}")
    print(f"[log] Mejora en {ganancias} de {len(filas)} candidatos")
    print(f"[log] Diferencia entre los dos mejores: "
          f"{mejor_con['con_log'] - mejor_sin['sin_log']:+.4f} RMSE")

    OUT.write_text(json.dumps({
        "n_train": N_TRAIN, "cv": N_SPLITS,
        "log_cols": {c: float(Xs[c].skew()) for c in log_cols},
        "filas": filas,
        "mejor_sin_log": mejor_sin, "mejor_con_log": mejor_con,
        "candidatos_que_mejoran": ganancias,
        "seconds": time.perf_counter() - t0,
    }, indent=2))
    print(f"[log] Guardado → {OUT}")


if __name__ == "__main__":
    main()

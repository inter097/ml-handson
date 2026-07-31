"""
Buscar el preprocesamiento junto con el modelo — ejercicio 5 de Géron, cap. 2

El enunciado es una línea: «Automatically explore some preparation options
using RandomizedSearchCV». La solución oficial mete en el mismo espacio de
búsqueda los parámetros del transformador de k-vecinos del ejercicio 4 y los
del SVR:

    preprocessing__geo__estimator__n_neighbors  ∈ range(1, 30)
    preprocessing__geo__estimator__weights      ∈ {distance, uniform}
    svr__C                                      ~ loguniform(20, 200_000)
    svr__gamma                                  ~ expon(scale=1.0)

Aquí se busca lo mismo con el `KNNGeoFeature` del capítulo, que es el
transformador equivalente, y con el mismo tope y la misma proyección previa que
el ejercicio 2: la mitad de los sorteos cae en el kernel lineal, que ya está
medido como meseta.

La diferencia importante con el ejercicio 2 no es el espacio, es que ahora **el
preprocesamiento se reajusta en cada pliegue**. Solo se puede hacer porque vive
dentro del `Pipeline`; ajustar el k-vecinos una vez fuera contaminaría la
validación con las filas que luego hacen de validación.

Uso:
    python src/prep_search.py --check --tope 1800
    python src/prep_search.py --tope 1800
    python src/prep_search.py --resume --tope 1800
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import expon, loguniform
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, ParameterSampler, cross_val_score
from sklearn.svm import SVR

from preprocessing import PROCESSED_DIR, build_pipeline

REPORTS_DIR = Path("reports")
OUT = REPORTS_DIR / "prep_search.json"
MEDIDO = REPORTS_DIR / "svr_official.json"

N_TRAIN = 5_000
N_SPLITS = 3
N_JOBS = 3
N_ITER = 50
SEED = 42

DISTRIBS = {
    "knn__n_neighbors": list(range(1, 30)),
    "knn__weights": ["distance", "uniform"],
    "kernel": ["linear", "rbf"],
    "C": loguniform(20, 200_000),
    "gamma": expon(scale=1.0),
}


def sortear() -> list:
    muestras = list(ParameterSampler(DISTRIBS, n_iter=N_ITER, random_state=SEED))
    return [{k: v for k, v in m.items()
             if not (k == "gamma" and m["kernel"] == "linear")} for m in muestras]


def modelo_de_costo() -> tuple:
    """Mismo ajuste que el ejercicio 2, sobre los tiempos ya medidos.

    El k-vecinos añade su parte, pero es despreciable al lado del SVR: ajustar
    un KNeighborsRegressor sobre 3,333 coordenadas cuesta milisegundos.
    """
    datos = json.loads(MEDIDO.read_text())["resultados"]
    lin = [r for r in datos if r["kernel"] == "linear"]
    rbf = [r for r in datos if r["kernel"] == "rbf"]
    pl = np.polyfit(np.log([r["C"] for r in lin]), np.log([r["seconds"] for r in lin]), 1)
    A = np.column_stack([np.log([r["C"] for r in rbf]),
                         np.log([r["gamma"] for r in rbf]), np.ones(len(rbf))])
    pr, *_ = np.linalg.lstsq(A, np.log([r["seconds"] for r in rbf]), rcond=None)
    return pl, pr


def proyectar(cand: dict, pl, pr) -> float:
    if cand["kernel"] == "linear":
        return float(np.exp(np.polyval(pl, np.log(cand["C"]))))
    return float(np.exp(pr[0] * np.log(cand["C"]) + pr[1] * np.log(cand["gamma"]) + pr[2]))


def clave(cand: dict) -> str:
    g = cand.get("gamma")
    return (f"{cand['kernel']}|C={cand['C']:.4g}|gamma={g:.4g}" if g else
            f"{cand['kernel']}|C={cand['C']:.4g}|gamma=-") + \
           f"|k={cand['knn__n_neighbors']}|{cand['knn__weights'][:4]}"


def pipeline_de(cand: dict, X):
    svr = SVR(**{k: v for k, v in cand.items() if k in ("kernel", "C", "gamma")})
    pipe = build_pipeline(svr, X, use_knn_geo=True)
    pipe.set_params(prep__knn__n_neighbors=cand["knn__n_neighbors"],
                    prep__knn__weights=cand["knn__weights"])
    return pipe


def check(tope: float | None) -> None:
    pl, pr = modelo_de_costo()
    cands = sorted(sortear(), key=lambda c: proyectar(c, pl, pr))
    total = sum(proyectar(c, pl, pr) for c in cands)
    lineales = sum(1 for c in cands if c["kernel"] == "linear")
    print(f"{N_ITER} candidatos con semilla {SEED}: {lineales} lineales, "
          f"{N_ITER - lineales} rbf")
    print(f"Reloj proyectado con {N_JOBS} procesos: {total / 3600:.1f} h")
    if tope:
        fuera = [c for c in cands if proyectar(c, pl, pr) > tope]
        dentro = total - sum(proyectar(c, pl, pr) for c in fuera)
        print(f"Con tope de {tope / 60:.0f} min por candidato: {len(fuera)} fuera, "
              f"reloj {dentro / 3600:.1f} h")
    print("\nProyección con los tiempos del ejercicio 1. El k-vecinos no cuenta:")
    print("ajustarlo sobre 3,333 coordenadas cuesta milisegundos.")


def guardar(estado: dict) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(estado, indent=2))
    os.replace(tmp, OUT)


def main(resume: bool, tope: float | None) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").values.ravel()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").values.ravel()
    Xs, ys = X_train.iloc[:N_TRAIN], y_train[:N_TRAIN]

    pl, pr = modelo_de_costo()
    cands = sorted(sortear(), key=lambda c: proyectar(c, pl, pr))
    estado = json.loads(OUT.read_text()) if (resume and OUT.exists()) else {
        "n_train": N_TRAIN, "cv": N_SPLITS, "n_iter": N_ITER, "seed": SEED,
        "tope_segundos": tope, "resultados": [], "saltados": [], "mejor": None,
    }
    hechos = {r["clave"] for r in estado["resultados"]}
    pendientes = [c for c in cands if clave(c) not in hechos]
    print(f"[prep] {len(pendientes)} candidatos por medir", flush=True)

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for i, cand in enumerate(pendientes, 1):
        etq, prev = clave(cand), proyectar(cand, pl, pr)
        if tope and prev > tope:
            estado["saltados"].append({"clave": etq, **cand, "proyectado": prev})
            guardar(estado)
            print(f"[{i:2d}/{len(pendientes)}] {etq:52s} SALTADO ({prev/60:.0f} min)",
                  flush=True)
            continue
        print(f"[{i:2d}/{len(pendientes)}] {etq:52s} ", end="", flush=True)
        t0 = time.perf_counter()
        scores = cross_val_score(pipeline_de(cand, Xs), Xs, ys, cv=cv, n_jobs=N_JOBS,
                                 scoring="neg_root_mean_squared_error")
        secs = time.perf_counter() - t0
        estado["resultados"].append({"clave": etq, **cand,
                                     "cv_rmse": float(-scores.mean()),
                                     "seconds": secs, "proyectado": prev})
        guardar(estado)
        print(f"RMSE={-scores.mean():.4f}  {secs:7.1f}s", flush=True)

    mejor = min(estado["resultados"], key=lambda r: r["cv_rmse"])
    final = pipeline_de(mejor, Xs).fit(Xs, ys)
    rmse = float(np.sqrt(mean_squared_error(y_test, final.predict(X_test))))
    estado["mejor"] = {**mejor, "test_rmse": rmse}
    guardar(estado)
    print(f"\n[prep] Mejor: {mejor['clave']} → CV {mejor['cv_rmse']:.4f} | Test {rmse:.4f}",
          flush=True)
    if estado["saltados"]:
        print(f"[prep] {len(estado['saltados'])} sin ejecutar por el tope", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="La búsqueda del ejercicio 5.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tope", type=float, default=None)
    args = parser.parse_args()
    if args.check:
        check(args.tope)
    else:
        main(args.resume, args.tope)

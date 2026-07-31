"""
La rejilla del enunciado, entera — ejercicio 1 de Géron, cap. 2

`svr_study.py` mide de dónde viene el costo de un SVR y afina sobre un rango
corto de C. Este script hace lo otro: recorre la rejilla exacta que propone la
solución oficial del libro, hasta C = 30,000 en el kernel lineal.

    linear : C ∈ {10, 30, 100, 300, 1000, 3000, 10000, 30000}          8
    rbf    : C ∈ {1, 3, 10, 30, 100, 300, 1000} × gamma ∈ {0.01 … 3}  42

Son 50 candidatos con cv=3 sobre las primeras 5,000 filas, como pide el
enunciado. La proyección a partir de lo medido en `svr_study.py` da entre 6 y 8
horas de CPU, casi todas en los dos valores más altos de C del kernel lineal.

De ahí las tres decisiones de diseño:

  1. **Los candidatos van de barato a caro.** El archivo de resultados tiene
     valor desde el primer minuto, y si el más caro nunca termina, lo demás ya
     está medido y se puede publicar diciendo qué faltó.
  2. **Se escribe después de cada candidato**, con reemplazo atómico. Un corte
     de luz cuesta un candidato, no la noche entera.
  3. **`--resume` salta lo ya registrado**, así que relanzarlo continúa.

El paralelismo va dentro de la validación cruzada (tres pliegues, tres
procesos) y no sobre los candidatos: así el pico de memoria es de tres ajustes
y el progreso queda ordenado. Un SVR sobre 5,000 × 24 ocupa unos pocos MB, el
límite de esta máquina no se toca por ningún lado.

Uso:
    python src/svr_official.py --check     # proyecta y no ejecuta nada
    python src/svr_official.py             # la rejilla entera
    python src/svr_official.py --resume    # continúa una ejecución cortada
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

from preprocessing import build_pipeline, load_data

REPORTS_DIR = Path("reports")
OUT = REPORTS_DIR / "svr_official.json"

N_TRAIN = 5_000      # las primeras 5,000 filas, como pide el enunciado
N_SPLITS = 3         # cv=3, también del enunciado
N_JOBS = 3           # un proceso por pliegue, en los núcleos de rendimiento

# La rejilla de la solución oficial, verbatim:
# https://github.com/ageron/handson-ml3 → 02_end_to_end_machine_learning_project.ipynb
GRID_LINEAR = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]
GRID_RBF_C = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
GRID_RBF_GAMMA = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def candidates() -> list:
    """Los 50 candidatos, de barato a caro.

    El orden lo pone C, que es lo que domina el tiempo: entre C=10 y C=100 el
    kernel lineal se multiplica por 7.6. Con la lista ordenada, una ejecución
    cortada a la mitad deja medida la mitad barata, que es la que tiene más
    candidatos.
    """
    out = []
    for C in GRID_RBF_C:
        for gamma in GRID_RBF_GAMMA:
            out.append({"kernel": "rbf", "C": C, "gamma": gamma})
    for C in GRID_LINEAR:
        out.append({"kernel": "linear", "C": C})
    return sorted(out, key=lambda c: (c["C"], c["kernel"]))


def clave(cand: dict) -> str:
    """Identificador estable de un candidato, para poder reanudar."""
    g = cand.get("gamma", "-")
    return f"{cand['kernel']}|C={cand['C']:g}|gamma={g}"


def proyeccion() -> None:
    """Proyecta el reloj sin ajustar nada, y dice de dónde sale cada cifra.

    Las cifras hasta C=100 están medidas en reports/svr_study.json. Por encima
    son extrapolación del factor por década (7.6 en lineal, 7.0 en rbf), y ahí
    la palabra que corresponde es «proyectado», no «medido».
    """
    medido = {"linear": {10.0: 5.25, 100.0: 39.93}, "rbf": {10.0: 0.83, 100.0: 5.77}}
    factor = {"linear": 7.6, "rbf": 7.0}
    escala_n = (N_TRAIN * (N_SPLITS - 1) / N_SPLITS / 4000) ** 1.33  # de n=4000 al pliegue

    print(f"Rejilla oficial: {len(candidates())} candidatos × {N_SPLITS} pliegues "
          f"= {len(candidates()) * N_SPLITS} ajustes de {int(N_TRAIN * (N_SPLITS-1)/N_SPLITS):,} filas")
    total = 0.0
    peor = 0.0
    for cand in candidates():
        k, C = cand["kernel"], cand["C"]
        base = medido[k][100.0] * factor[k] ** np.log10(C / 100.0)
        segundos = base * escala_n
        total += segundos * N_SPLITS
        peor = max(peor, segundos)
    print(f"CPU acumulado proyectado : {total/3600:.1f} h")
    print(f"Reloj con {N_JOBS} procesos    : {total/N_JOBS/3600:.1f} h")
    print(f"Ajuste más lento         : {peor/60:.0f} min  (piso: no baja repartiendo)")
    print(f"Memoria                  : ~{N_JOBS * 3} MB. No es el límite de esta máquina.")
    print("\nLas cifras por encima de C=100 son extrapolación, no medición.")


def guardar(estado: dict) -> None:
    """Escritura atómica: se escribe al lado y se reemplaza de golpe.

    Sin esto, un corte durante el write deja un JSON truncado y se pierde todo
    lo medido, no solo el candidato en curso.
    """
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(estado, indent=2))
    os.replace(tmp, OUT)


def main(resume: bool) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    Xs, ys = X_train.iloc[:N_TRAIN], y_train[:N_TRAIN]

    estado = json.loads(OUT.read_text()) if (resume and OUT.exists()) else {
        "n_train": N_TRAIN, "cv": N_SPLITS, "resultados": [], "mejor": None,
    }
    hechos = {r["clave"] for r in estado["resultados"]}

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    pendientes = [c for c in candidates() if clave(c) not in hechos]
    print(f"[svr-oficial] {len(pendientes)} candidatos por medir "
          f"({len(hechos)} ya en {OUT})", flush=True)

    for i, cand in enumerate(pendientes, 1):
        etq = clave(cand)
        print(f"[{i:2d}/{len(pendientes)}] {etq:34s} ", end="", flush=True)
        pipe = build_pipeline(SVR(**cand), Xs)
        t0 = time.perf_counter()
        scores = cross_val_score(pipe, Xs, ys, cv=cv, n_jobs=N_JOBS,
                                 scoring="neg_root_mean_squared_error")
        secs = time.perf_counter() - t0
        estado["resultados"].append({
            "clave": etq, **cand,
            "cv_rmse": float(-scores.mean()), "cv_std": float(scores.std()),
            "seconds": secs,
        })
        guardar(estado)
        print(f"RMSE={-scores.mean():.4f}  {secs:7.1f}s", flush=True)

    mejor = min(estado["resultados"], key=lambda r: r["cv_rmse"])
    print(f"\n[svr-oficial] Mejor en validación: {mejor['clave']} → {mejor['cv_rmse']:.4f}",
          flush=True)

    # El enunciado pregunta cuánto rinde el mejor SVR, así que el ganador se
    # reajusta sobre las 5,000 filas y se mide contra el test set entero.
    params = {k: v for k, v in mejor.items()
              if k in ("kernel", "C", "gamma")}
    final = build_pipeline(SVR(**params), Xs).fit(Xs, ys)
    rmse = float(np.sqrt(mean_squared_error(y_test, final.predict(X_test))))
    estado["mejor"] = {**mejor, "test_rmse": rmse}
    guardar(estado)
    print(f"[svr-oficial] Test RMSE: {rmse:.4f}", flush=True)
    print(f"[svr-oficial] Guardado → {OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="La rejilla oficial del ejercicio 1, entera.")
    parser.add_argument("--check", action="store_true", help="Proyectar el costo y salir")
    parser.add_argument("--resume", action="store_true", help="Continuar una ejecución cortada")
    args = parser.parse_args()
    if args.check:
        proyeccion()
    else:
        main(args.resume)

"""
La búsqueda aleatoria del enunciado — ejercicio 2 de Géron, cap. 2

El enunciado es una línea: «Try replacing the GridSearchCV with a
RandomizedSearchCV». Lo que enseña no es el cambio de clase, es de dónde salen
los valores. La rejilla del ejercicio 1 los lleva escritos a mano; aquí se
muestrean de distribuciones continuas:

    C     ~ loguniform(20, 200_000)   cuatro órdenes de magnitud
    gamma ~ expon(scale=1.0)
    kernel ∈ {linear, rbf}

Sigue siendo el mismo SVR sobre las mismas 5,000 filas con cv=3, así que su
cifra se compara directamente con la de `svr_official.py`.

El costo hay que proyectarlo antes, y por una razón que la rejilla ya dejó
medida: el kernel lineal se estanca en 0.6452 desde C=10 y a la vez su tiempo
crece como C^0.84. La aleatoria sortea C hasta 200,000, o sea justo el tramo
caro que no compra nada.

`--check` usa los tiempos ya medidos en reports/svr_official.json para
proyectar candidato a candidato. Los candidatos son los mismos que sorteará la
búsqueda, porque `ParameterSampler` con la misma semilla da la misma lista.

Uso:
    python src/svr_random.py --check     # proyecta y no ejecuta nada
    python src/svr_random.py             # la búsqueda entera
    python src/svr_random.py --resume    # continúa una ejecución cortada
    python src/svr_random.py --tope 900  # salta los candidatos proyectados por
                                         # encima de 900 s y los deja anotados
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import expon, loguniform
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, ParameterSampler, cross_val_score
from sklearn.svm import SVR

from preprocessing import build_pipeline, load_data

REPORTS_DIR = Path("reports")
OUT = REPORTS_DIR / "svr_random.json"
MEDIDO = REPORTS_DIR / "svr_official.json"

N_TRAIN = 5_000
N_SPLITS = 3
N_JOBS = 3
N_ITER = 50
SEED = 42

# Las distribuciones de la solución oficial, verbatim.
DISTRIBS = {
    "kernel": ["linear", "rbf"],
    "C": loguniform(20, 200_000),
    "gamma": expon(scale=1.0),
}


def sortear() -> list:
    """Los 50 candidatos de la búsqueda, con la semilla del notebook.

    `RandomizedSearchCV` usa `ParameterSampler` por dentro, así que con la
    misma semilla la lista es la misma y se puede proyectar el costo de cada
    uno antes de ajustar nada.
    """
    muestras = list(ParameterSampler(DISTRIBS, n_iter=N_ITER, random_state=SEED))
    # gamma no existe en el kernel lineal. Dejarlo puesto no cambia el ajuste,
    # pero ensucia la clave con la que se identifica el candidato.
    return [{k: v for k, v in m.items() if not (k == "gamma" and m["kernel"] == "linear")}
            for m in muestras]


def modelo_de_costo() -> tuple:
    """Ajusta el tiempo por candidato contra C y gamma, sobre lo ya medido.

    Los 50 tiempos de la rejilla oficial son la base. El lineal solo depende de
    C, así que sale una recta en log-log; el rbf depende de C y gamma, y sale
    un plano. Es una proyección, no una medición, y así hay que publicarla.
    """
    datos = json.loads(MEDIDO.read_text())["resultados"]
    lin = [r for r in datos if r["kernel"] == "linear"]
    rbf = [r for r in datos if r["kernel"] == "rbf"]

    pl = np.polyfit(np.log([r["C"] for r in lin]), np.log([r["seconds"] for r in lin]), 1)

    A = np.column_stack([np.log([r["C"] for r in rbf]),
                         np.log([r["gamma"] for r in rbf]),
                         np.ones(len(rbf))])
    pr, *_ = np.linalg.lstsq(A, np.log([r["seconds"] for r in rbf]), rcond=None)
    return pl, pr


def proyectar(cand: dict, pl, pr) -> float:
    if cand["kernel"] == "linear":
        return float(np.exp(np.polyval(pl, np.log(cand["C"]))))
    return float(np.exp(pr[0] * np.log(cand["C"]) + pr[1] * np.log(cand["gamma"]) + pr[2]))


def clave(cand: dict) -> str:
    g = cand.get("gamma")
    return f"{cand['kernel']}|C={cand['C']:.4g}|gamma={g:.4g}" if g else \
           f"{cand['kernel']}|C={cand['C']:.4g}|gamma=-"


def check(tope: float | None) -> None:
    pl, pr = modelo_de_costo()
    cands = sorted(sortear(), key=lambda c: proyectar(c, pl, pr))
    total = sum(proyectar(c, pl, pr) for c in cands)
    lineales = sum(1 for c in cands if c["kernel"] == "linear")

    print(f"{N_ITER} candidatos sorteados con semilla {SEED}: "
          f"{lineales} lineales, {N_ITER - lineales} rbf")
    print(f"C mediana {np.median([c['C'] for c in cands]):,.0f} · "
          f"máxima {max(c['C'] for c in cands):,.0f}")
    print(f"\nReloj proyectado con {N_JOBS} procesos: {total/3600:.1f} h")
    print("\nLos cinco más caros:")
    for c in cands[-5:]:
        print(f"  {clave(c):40s} {proyectar(c, pl, pr)/60:8.1f} min")
    if tope:
        fuera = [c for c in cands if proyectar(c, pl, pr) > tope]
        dentro = total - sum(proyectar(c, pl, pr) for c in fuera)
        print(f"\nCon tope de {tope:.0f}s por candidato: {len(fuera)} quedan fuera, "
              f"reloj {dentro/3600:.1f} h")
    print("\nProyección a partir de los tiempos de svr_official.json, no medición.")


def guardar(estado: dict) -> None:
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(estado, indent=2))
    os.replace(tmp, OUT)


def main(resume: bool, tope: float | None) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    Xs, ys = X_train.iloc[:N_TRAIN], y_train[:N_TRAIN]

    pl, pr = modelo_de_costo()
    # De barato a caro: la lista de candidatos es la misma que sortea la
    # búsqueda, solo cambia el orden en que se evalúan, y así una ejecución
    # cortada deja medido lo que más candidatos cubre.
    cands = sorted(sortear(), key=lambda c: proyectar(c, pl, pr))

    estado = json.loads(OUT.read_text()) if (resume and OUT.exists()) else {
        "n_train": N_TRAIN, "cv": N_SPLITS, "n_iter": N_ITER, "seed": SEED,
        "tope_segundos": tope, "resultados": [], "saltados": [], "mejor": None,
    }
    hechos = {r["clave"] for r in estado["resultados"]}
    pendientes = [c for c in cands if clave(c) not in hechos]
    print(f"[svr-random] {len(pendientes)} candidatos por medir", flush=True)

    for i, cand in enumerate(pendientes, 1):
        etq, prev = clave(cand), proyectar(cand, pl, pr)
        if tope and prev > tope:
            # No se ejecuta, pero queda anotado con su proyección: un recorte
            # declarado es parte del resultado.
            estado["saltados"].append({"clave": etq, **cand, "proyectado": prev})
            guardar(estado)
            print(f"[{i:2d}/{len(pendientes)}] {etq:40s} SALTADO ({prev/60:.0f} min)", flush=True)
            continue
        print(f"[{i:2d}/{len(pendientes)}] {etq:40s} ", end="", flush=True)
        pipe = build_pipeline(SVR(**cand), Xs)
        t0 = time.perf_counter()
        scores = cross_val_score(pipe, Xs, ys, cv=KFold(N_SPLITS, shuffle=True, random_state=42),
                                 n_jobs=N_JOBS, scoring="neg_root_mean_squared_error")
        secs = time.perf_counter() - t0
        estado["resultados"].append({
            "clave": etq, **cand, "cv_rmse": float(-scores.mean()),
            "seconds": secs, "proyectado": prev,
        })
        guardar(estado)
        print(f"RMSE={-scores.mean():.4f}  {secs:7.1f}s (proy. {prev:.0f}s)", flush=True)

    mejor = min(estado["resultados"], key=lambda r: r["cv_rmse"])
    params = {k: v for k, v in mejor.items() if k in ("kernel", "C", "gamma")}
    final = build_pipeline(SVR(**params), Xs).fit(Xs, ys)
    rmse = float(np.sqrt(mean_squared_error(y_test, final.predict(X_test))))
    estado["mejor"] = {**mejor, "test_rmse": rmse}
    guardar(estado)
    print(f"\n[svr-random] Mejor: {mejor['clave']} → CV {mejor['cv_rmse']:.4f} | "
          f"Test {rmse:.4f}", flush=True)
    if estado["saltados"]:
        print(f"[svr-random] {len(estado['saltados'])} candidatos sin ejecutar por el tope",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="La búsqueda aleatoria del ejercicio 2.")
    parser.add_argument("--check", action="store_true", help="Proyectar el costo y salir")
    parser.add_argument("--resume", action="store_true", help="Continuar una ejecución cortada")
    parser.add_argument("--tope", type=float, default=None,
                        help="Segundos proyectados por candidato a partir de los cuales se salta")
    args = parser.parse_args()
    if args.check:
        check(args.tope)
    else:
        main(args.resume, args.tope)

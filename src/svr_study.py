"""
Estudio de SVR — por qué queda fuera de `make tune-all`
Referencia: Géron cap. 2, ejercicio 1 ("try a Support Vector Machine regressor
with various hyperparameters")

Una búsqueda de SVR sobre las 16,512 filas del train set tardaba más de 20
minutos. La explicación fácil era "SVR escala O(n²)", pero al medirlo resultó
falsa: el exponente real ronda n^1.3 y un ajuste completo toma segundos.

El culpable es el valor de C, no el tamaño del dataset. C alto significa menos
tolerancia al error, lo que multiplica los vectores de soporte y las
iteraciones de libsvm — con kernel polinómico llega a ser dos órdenes de
magnitud más lento que la configuración barata.

Este script mide las tres cosas:

  1. Tiempo de ajuste vs tamaño del train set, y el exponente de escalado.
  2. Tiempo de ajuste vs kernel y C, a tamaño fijo — donde está el problema real.
  3. Tuning sobre una submuestra, comparado contra el mejor modelo del proyecto.

Uso:
    python src/svr_study.py            # completo
    python src/svr_study.py --quick    # menos tamaños y menos combinaciones
    # o: make svr-study
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR

from preprocessing import build_pipeline, load_data

REPORTS_DIR = Path("reports")
SIZES = [500, 1000, 2000, 4000, 8000]
SIZES_QUICK = [500, 1000, 2000]

# El espacio del ejercicio: kernels y C, más gamma para el RBF.
GRID = {
    "model__kernel": ["linear", "rbf"],
    "model__C": [1.0, 10.0, 100.0],
    "model__gamma": ["scale", 0.1],
}
GRID_QUICK = {
    "model__kernel": ["linear", "rbf"],
    "model__C": [1.0, 10.0],
}


def measure_scaling(X_train, y_train, sizes: list) -> list:
    """Tiempo de ajuste de SVR contra el tamaño del train set."""
    print("\n[svr] Escalado: tiempo de ajuste vs tamaño")
    print(f"  {'n':>7s} {'segundos':>10s} {'s/n²·1e6':>10s}")
    rows = []
    for n in sizes:
        Xs, ys = X_train.iloc[:n], y_train[:n]
        pipe = build_pipeline(SVR(kernel="rbf", C=10), Xs)
        t0 = time.perf_counter()
        pipe.fit(Xs, ys)
        secs = time.perf_counter() - t0
        rows.append({"n": n, "seconds": secs})
        print(f"  {n:7,d} {secs:10.2f} {secs / n**2 * 1e6:10.3f}")
    return rows


def measure_hyperparams(X_train, y_train, n: int = 4000) -> list:
    """Tiempo de ajuste por kernel y C, a tamaño fijo.

    Aquí se ve que el costo lo domina C, no n.
    """
    print(f"\n[svr] Costo por hiperparámetro (n={n:,})")
    print(f"  {'kernel':8s} {'C':>7s} {'segundos':>10s}")
    Xs, ys = X_train.iloc[:n], y_train[:n]
    rows = []
    for kernel in ["rbf", "linear", "poly"]:
        for C in [1.0, 10.0, 100.0]:
            pipe = build_pipeline(SVR(kernel=kernel, C=C), Xs)
            t0 = time.perf_counter()
            pipe.fit(Xs, ys)
            secs = time.perf_counter() - t0
            rows.append({"kernel": kernel, "C": C, "seconds": secs})
            print(f"  {kernel:8s} {C:7.0f} {secs:10.1f}")
    cheap = min(rows, key=lambda r: r["seconds"])
    dear = max(rows, key=lambda r: r["seconds"])
    print(f"  → {dear['kernel']} C={dear['C']:.0f} es {dear['seconds']/cheap['seconds']:.0f}× "
          f"más lento que {cheap['kernel']} C={cheap['C']:.0f}")
    return rows


def fit_exponent(rows: list) -> float:
    """Ajusta t = k·n^p en escala log-log; p es el exponente de escalado."""
    n = np.log(np.array([r["n"] for r in rows], dtype=float))
    t = np.log(np.array([r["seconds"] for r in rows], dtype=float))
    p, _ = np.polyfit(n, t, 1)
    return float(p)


def tune_subsample(X_train, y_train, X_test, y_test, n: int, grid: dict) -> dict:
    """GridSearchCV sobre una submuestra, evaluado en el test set completo."""
    print(f"\n[svr] Tuning sobre submuestra de {n:,} filas")
    Xs, ys = X_train.iloc[:n], y_train[:n]
    search = GridSearchCV(
        build_pipeline(SVR(), Xs),
        grid,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.perf_counter()
    search.fit(Xs, ys)
    secs = time.perf_counter() - t0

    preds = search.best_estimator_.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    print(f"  mejor: {best}")
    print(f"  CV RMSE={-search.best_score_:.4f} | Test RMSE={rmse:.4f} | {secs:.0f}s")
    return {"n": n, "params": best, "cv_rmse": float(-search.best_score_),
            "test_rmse": rmse, "seconds": secs}


def main(quick: bool) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    sizes = SIZES_QUICK if quick else SIZES
    scaling = measure_scaling(X_train, y_train, sizes)
    p = fit_exponent(scaling)

    biggest = scaling[-1]
    full_n = len(X_train)
    projected = biggest["seconds"] * (full_n / biggest["n"]) ** p

    print(f"\n[svr] Exponente medido: t ∝ n^{p:.2f}")
    print(f"[svr] Proyección a {full_n:,} filas: {projected:.0f}s ({projected/60:.1f} min) por ajuste")

    hyper = measure_hyperparams(X_train, y_train, n=2000 if quick else 4000)

    tuned = tune_subsample(X_train, y_train, X_test, y_test,
                           n=sizes[-1], grid=GRID_QUICK if quick else GRID)

    out = {"scaling": scaling, "exponent": p, "projected_full_fit_seconds": projected,
           "hyperparams": hyper, "tuned": tuned, "train_size": full_n}
    (REPORTS_DIR / "svr_study.json").write_text(json.dumps(out, indent=2))
    print(f"\n[svr] Guardado → {REPORTS_DIR / 'svr_study.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estudio de escalado y tuning de SVR.")
    parser.add_argument("--quick", action="store_true", help="Menos tamaños y menos combinaciones")
    args = parser.parse_args()
    main(args.quick)

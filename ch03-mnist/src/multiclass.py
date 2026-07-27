"""
Fase 5 — Clasificación multiclase
Referencia: Géron cap. 3, "Multiclass Classification"

De «¿es un 5?» a «¿cuál de los diez?».

Muchos algoritmos son binarios por naturaleza. Para distinguir diez clases,
scikit-learn los envuelve automáticamente con una de dos estrategias:

  Uno contra todos (OvR)
    Entrena 10 clasificadores — «¿es un 0?», «¿es un 1?»... — y se queda con
    el de puntuación más alta. 10 modelos sobre 60,000 imágenes cada uno.

  Uno contra uno (OvO)
    Entrena un clasificador por cada par de dígitos: 0-vs-1, 0-vs-2... son
    45 modelos. Suena peor, pero cada uno solo ve las imágenes de sus dos
    dígitos — unas 12,000 en vez de 60,000. Para algoritmos que escalan mal
    con el tamaño, como las máquinas de vectores de soporte, sale ganando.

Por eso scikit-learn elige OvO para SVC y OvR para casi todo lo demás.

Este módulo también mide algo que el capítulo destaca: **cuánto ayuda escalar**
a un modelo lineal. La diferencia no es cosmética.

Uso:
    python src/multiclass.py
    python src/multiclass.py --models sgd forest
    # o: make multiclass
"""
import argparse
import time

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from baseline import EXPERIMENT, SEED, build_pipeline
from features import load_data

CV = 3

MODELS = {
    "sgd":        lambda: SGDClassifier(random_state=SEED),
    "forest":     lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1),
    # SVC no escala a 60,000 imágenes; se entrena sobre una submuestra.
    "svc":        lambda: SVC(random_state=SEED),
    "svc_ovo":    lambda: OneVsOneClassifier(SVC(random_state=SEED), n_jobs=-1),
}

# Modelos que solo se corren sobre una parte del train, por costo.
SUBSAMPLE = {"svc": 10_000, "svc_ovo": 10_000}


def estrategia(pipeline: Pipeline, n_clases: int) -> str:
    """Qué montó scikit-learn por debajo para manejar 10 clases.

    Ojo con confundir estrategias multiclase con ensambles: los 100
    `estimators_` de un bosque son sus árboles, no clasificadores binarios
    uno por clase. El bosque maneja las 10 clases de forma nativa.
    """
    model = pipeline.named_steps["model"]

    if isinstance(model, (OneVsOneClassifier, OneVsRestClassifier)):
        n = len(model.estimators_)
        cual = "uno contra uno" if isinstance(model, OneVsOneClassifier) else "uno contra todos"
        return f"{cual}, explícito ({n} clasificadores)"

    # Los modelos lineales delatan la estrategia en la forma de sus
    # coeficientes: una fila por clase significa uno contra todos.
    coef = getattr(model, "coef_", None)
    if coef is not None and coef.ndim == 2 and coef.shape[0] == n_clases:
        return f"uno contra todos, interno ({n_clases} clasificadores)"

    if isinstance(model, SVC):
        pares = n_clases * (n_clases - 1) // 2
        return f"uno contra uno, interno de libsvm ({pares} clasificadores)"

    return "nativamente multiclase"


def run(names: list) -> None:
    X_train, _, y_train, _ = load_data()

    print(f"\n[multiclass] {len(np.unique(y_train))} clases · {len(X_train):,} imágenes")
    print(f"\n  {'modelo':12s} {'n':>8s} {'exactitud':>11s} {'seg':>8s}  estrategia")

    mlflow.set_experiment(EXPERIMENT)
    for name in names:
        n = SUBSAMPLE.get(name, len(X_train))
        Xs, ys = X_train[:n], y_train[:n]

        pipe = build_pipeline(MODELS[name]())
        t0 = time.perf_counter()
        scores = cross_val_score(pipe, Xs, ys, cv=CV, scoring="accuracy", n_jobs=-1)
        secs = time.perf_counter() - t0

        # Ajustar una vez para poder inspeccionar qué estrategia se usó.
        pipe.fit(Xs[:2000], ys[:2000])
        estr = estrategia(pipe, len(np.unique(ys)))

        marca = "" if n == len(X_train) else "  ← submuestra"
        print(f"  {name:12s} {n:8,d} {scores.mean():11.4f} {secs:8.1f}  {estr}{marca}")

        with mlflow.start_run(run_name=f"multiclase_{name}"):
            mlflow.log_params({"tarea": "multiclase_10", "modelo": name,
                               "n_train": n, "cv": CV, "estrategia": estr})
            mlflow.log_metrics({"exactitud": float(scores.mean()),
                                "exactitud_std": float(scores.std()),
                                "segundos": secs})

    _efecto_escalado(X_train, y_train)


def _efecto_escalado(X_train, y_train) -> None:
    """Lo mismo con y sin escalar, para ver cuánto pesa en un modelo lineal."""
    print("\n[multiclass] Cuánto ayuda escalar (modelo lineal, 10,000 imágenes)")
    Xs, ys = X_train[:10_000], y_train[:10_000]

    crudo = Pipeline([("model", SGDClassifier(random_state=SEED))])
    escalado = build_pipeline(SGDClassifier(random_state=SEED))

    a = cross_val_score(crudo, Xs, ys, cv=CV, scoring="accuracy", n_jobs=-1).mean()
    b = cross_val_score(escalado, Xs, ys, cv=CV, scoring="accuracy", n_jobs=-1).mean()

    print(f"  píxeles crudos (0–255)   {a:.4f}")
    print(f"  escalados a 0–1          {b:.4f}   ({b-a:+.4f})")

    with mlflow.start_run(run_name="efecto_escalado"):
        mlflow.log_params({"tarea": "multiclase_10", "modelo": "sgd", "n_train": 10_000})
        mlflow.log_metrics({"exactitud_crudo": float(a), "exactitud_escalado": float(b),
                            "delta": float(b - a)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificación multiclase sobre MNIST.")
    parser.add_argument("--models", nargs="+", default=["sgd", "forest", "svc"],
                        choices=list(MODELS))
    args = parser.parse_args()
    run(args.models)

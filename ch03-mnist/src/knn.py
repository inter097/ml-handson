"""
Ejercicio 1 — k-vecinos afinado
Referencia: Géron cap. 3, ejercicio 1 ("try to build a classifier for the MNIST
dataset that achieves over 97% accuracy on the test set")

k-vecinos no entrena: memoriza. «Ajustar» es guardar las 60,000 imágenes; el
trabajo ocurre al predecir, comparando cada imagen nueva contra las guardadas.

Eso invierte la intuición de costo. Aquí el tiempo se va en la predicción, y
por eso una búsqueda con validación cruzada duele: cada pliegue predice sobre
20,000 imágenes contra las 40,000 restantes.

Medido antes de lanzarlo, sin embargo, resulta más barato de lo que sugiere la
teoría — el exponente ronda n^0.8 porque scikit-learn usa árboles de búsqueda
y paraleliza. Una búsqueda completa toma minutos, no horas.

Los dos hiperparámetros que importan:

  n_neighbors   cuántos vecinos votan
  weights       si todos los vecinos pesan igual («uniform») o si los más
                cercanos pesan más («distance»)

Uso:
    python src/knn.py
    # o: make knn
"""
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from baseline import EXPERIMENT
from features import load_data

REPORTS_DIR = Path("reports")
MEJOR = REPORTS_DIR / "knn_best.json"
CV = 3

# sklearn reserva hasta `working_memory` MB por hilo para los bloques de
# distancias, y el valor por defecto es 1024. Con k-vecinos paralelizado sobre
# 10 núcleos eso puede pedir 10 GB de golpe y tumbar una máquina de 16.
# Medido sobre 20,000 imágenes: 3.00 GB con los valores por defecto contra
# 0.49 GB con este ajuste, y además más rápido.
sklearn.set_config(working_memory=128)

GRID = {
    "n_neighbors": [3, 4, 5],
    "weights": ["uniform", "distance"],
}


def run() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    # float32 en vez del float64 al que sklearn convertiría: los píxeles van de
    # 0 a 255 y no necesitan esa precisión. Ahorra la mitad de memoria en el
    # array que k-vecinos guarda y en cada bloque de distancias.
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    print(f"\n[knn] Búsqueda sobre {len(X_train):,} imágenes · "
          f"{len(GRID['n_neighbors']) * len(GRID['weights'])} combinaciones × {CV} pliegues")

    # Sin escalar: los píxeles ya comparten unidad y rango, y k-vecinos solo
    # necesita que las distancias sean comparables entre sí.
    #
    # ⚠️ NO poner n_jobs=-1 en los dos niveles. GridSearchCV paraleliza con
    # procesos: cada uno recibe su propia copia del train set, y sklearn los
    # convierte a float64 para las distancias — 60,000 × 784 × 8 bytes son
    # 376 MB por copia. Con 10 núcleos eso son 3.8 GB solo en datos, más la
    # sobresuscripción de que cada proceso pida 10 hilos a su vez.
    #
    # KNeighborsClassifier paraleliza con hilos sobre memoria compartida, así
    # que la forma segura es dejarle a él los núcleos y que la búsqueda vaya
    # en serie: una sola copia de los datos.
    search = GridSearchCV(KNeighborsClassifier(n_jobs=-1), GRID,
                          cv=CV, scoring="accuracy", n_jobs=1, verbose=1)
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    secs = time.perf_counter() - t0

    print(f"\n[knn] Resultados de la búsqueda ({secs/60:.1f} min)")
    orden = np.argsort(-search.cv_results_["mean_test_score"])
    for i in orden:
        p = search.cv_results_["params"][i]
        s = search.cv_results_["mean_test_score"][i]
        marca = "  ← mejor" if i == search.best_index_ else ""
        print(f"  n_neighbors={p['n_neighbors']}  weights={p['weights']:9s} {s:.4f}{marca}")

    # El ejercicio pide el 97% sobre el conjunto de PRUEBA, no sobre la
    # validación cruzada. Es la primera vez en el capítulo que se toca.
    preds = search.best_estimator_.predict(X_test)
    test_acc = float(accuracy_score(y_test, preds))

    print(f"\n[knn] Validación cruzada  {search.best_score_:.4f}")
    print(f"[knn] Conjunto de prueba  {test_acc:.4f}   {'✓ supera el 97%' if test_acc > 0.97 else '✗ no llega al 97%'}")

    MEJOR.write_text(json.dumps({"params": search.best_params_,
                                 "cv_accuracy": float(search.best_score_),
                                 "test_accuracy": test_acc}, indent=2))

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="knn_afinado"):
        mlflow.log_params({"modelo": "KNeighborsClassifier", "cv": CV,
                           "n_train": len(X_train), **search.best_params_})
        mlflow.log_metrics({"exactitud_cv": float(search.best_score_),
                            "exactitud_test": test_acc, "segundos": secs})
        print(f"[knn] registrado en MLflow · run {mlflow.active_run().info.run_id[:8]}")


if __name__ == "__main__":
    run()

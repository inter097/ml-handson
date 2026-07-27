"""
Fase 2 — Partición train/test
Referencia: Géron cap. 3, "MNIST"

MNIST ya viene partido: las primeras 60,000 imágenes son el conjunto de
entrenamiento y las últimas 10,000 el de prueba. No se baraja ni se usa
`train_test_split`, y hay dos razones:

  1. Es la partición canónica. Todos los resultados publicados sobre MNIST
     desde 1998 usan exactamente estas 10,000 imágenes de prueba. Rebarajar
     haría que los números no fueran comparables con nada.

  2. Los dos conjuntos vienen de personas distintas — el de entrenamiento de
     empleados del censo estadounidense, el de prueba de estudiantes de
     bachillerato. Mezclarlos volvería el problema artificialmente fácil: el
     modelo vería la letra de la misma persona en ambos lados.

El conjunto de entrenamiento ya viene barajado, así que la validación cruzada
puede cortarlo en pliegues sin que ninguno quede lleno de un solo dígito.

Aquí no hay ingeniería de variables. Un píxel es un píxel; el escalado vive
en el Pipeline de cada modelo, como en el capítulo anterior.

Uso:
    python src/features.py
    # o: make features
"""
from pathlib import Path

import numpy as np

RAW_PATH = Path("data/raw/mnist.npz")
PROCESSED_DIR = Path("data/processed")
TRAIN_SIZE = 60_000


def build() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(RAW_PATH) as d:
        X, y = d["X"], d["y"]

    X_train, X_test = X[:TRAIN_SIZE], X[TRAIN_SIZE:]
    y_train, y_test = y[:TRAIN_SIZE], y[TRAIN_SIZE:]

    np.savez_compressed(PROCESSED_DIR / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(PROCESSED_DIR / "test.npz", X=X_test, y=y_test)

    print(f"[features] train={len(X_train):,} | test={len(X_test):,}")
    print("[features] proporción de cada dígito (%): train / test")
    tr = np.bincount(y_train, minlength=10) / len(y_train) * 100
    te = np.bincount(y_test, minlength=10) / len(y_test) * 100
    for d in range(10):
        print(f"             {d}   {tr[d]:5.2f}  {te[d]:5.2f}")

    # Este número es el que hace interesante todo el capítulo: ningún dígito
    # llega al 12% del total, así que un detector binario de un solo dígito
    # trabaja con clases muy desbalanceadas.
    print(f"[features] el dígito más común es el {tr.argmax()} con {tr.max():.1f}% del train")


def load_data() -> tuple:
    """Carga los splits tal cual: píxeles uint8 de 0 a 255.

    El escalado no se hace aquí a propósito — vive dentro del Pipeline de cada
    modelo, igual que en el capítulo anterior, para que se reajuste en cada
    pliegue de la validación cruzada.
    """
    with np.load(PROCESSED_DIR / "train.npz") as d:
        X_train, y_train = d["X"], d["y"]
    with np.load(PROCESSED_DIR / "test.npz") as d:
        X_test, y_test = d["X"], d["y"]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    build()

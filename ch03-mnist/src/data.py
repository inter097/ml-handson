"""
Fase 1 — Obtener datos
Referencia: Géron cap. 3, "MNIST"

Descarga MNIST desde OpenML y lo guarda en data/raw/mnist.npz.

70,000 imágenes de dígitos escritos a mano, de 28×28 píxeles. Cada imagen
llega aplanada como 784 valores de intensidad, y la etiqueta es el dígito
que representa.

Dos decisiones de almacenamiento:

  uint8 en vez de float64
    OpenML entrega los píxeles como float64: 70,000 × 784 × 8 bytes = 439 MB.
    Los valores reales van de 0 a 255, así que caben en un byte. En uint8 son
    55 MB, y comprimidos ~11 MB. El escalado a float lo hace el modelo.

  .npz en vez de parquet
    Esto es una matriz densa de píxeles, no una tabla con columnas que
    signifiquen cosas distintas. Parquet brilla en lo segundo; aquí solo
    estorba.

Uso:
    python src/data.py
    # o: make data
"""
from pathlib import Path

import numpy as np

RAW_DIR = Path("data/raw")
OUT = RAW_DIR / "mnist.npz"


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUT.exists():
        print(f"[data] ya existe: {OUT} — bórralo para volver a descargar")
        return

    from sklearn.datasets import fetch_openml

    print("[data] descargando MNIST desde OpenML (puede tardar un par de minutos)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    X = mnist.data.astype(np.uint8)
    y = mnist.target.astype(np.uint8)

    np.savez_compressed(OUT, X=X, y=y)

    mb = OUT.stat().st_size / 1e6
    print(f"[data] {X.shape[0]:,} imágenes de {X.shape[1]} píxeles → {OUT} ({mb:.1f} MB)")
    print(f"[data] rango de píxeles: {X.min()}–{X.max()} | etiquetas: {sorted(set(y.tolist()))}")

    counts = np.bincount(y)
    print("[data] imágenes por dígito:")
    print("       " + "  ".join(f"{d}:{c:,}" for d, c in enumerate(counts)))


if __name__ == "__main__":
    download()

"""
Ejercicio 2 — Aumentar los datos desplazando las imágenes
Referencia: Géron cap. 3, ejercicio 2 ("write a function that can shift an MNIST
image in any direction by one pixel... then train your best model on this
expanded training set")

La idea: un 7 desplazado un píxel a la derecha sigue siendo un 7, pero para
k-vecinos —que compara píxel contra píxel— es una imagen distinta. Añadiendo
las cuatro versiones desplazadas de cada dígito, el modelo aprende que la
posición exacta no importa. El conjunto pasa de 60,000 a 300,000 imágenes.

Es la técnica que en visión por computadora se llama aumento de datos, y suele
rendir más que afinar hiperparámetros.

⚠️ Sobre la memoria — este script se escribió después de tumbar una máquina
   de 16 GB con el ejercicio anterior. Las tres precauciones que faltaban:

     1. Paralelizar en UN solo nivel. GridSearchCV usa procesos y cada uno
        copia los datos; k-vecinos usa hilos sobre memoria compartida.
     2. `working_memory` de sklearn a 128 MB. Por defecto son 1024 MB POR
        HILO para los bloques de distancias.
     3. float32 en vez del float64 al que sklearn convertiría.

   Además, aquí se proyecta el pico ANTES de reservar nada y se aborta si no
   cabe con holgura.

Uso:
    python src/augment.py --check          # solo proyecta la memoria, no corre
    python src/augment.py --max-train 20000
    python src/augment.py                  # completo
    # o: make augment
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import sklearn
from scipy.ndimage import shift as nd_shift
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from baseline import EXPERIMENT
from features import load_data

REPORTS_DIR = Path("reports")
MEJOR = REPORTS_DIR / "knn_best.json"

sklearn.set_config(working_memory=128)

DESPLAZAMIENTOS = [(0, 1), (0, -1), (1, 0), (-1, 0)]   # derecha, izquierda, abajo, arriba
FRACCION_SEGURA = 0.45   # del total de RAM; deja sitio al sistema y al navegador


def ram_total_gb() -> float:
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    return int(out.stdout.strip()) / 1024**3


def proyectar(n_base: int, n_test: int) -> dict:
    """Estima el pico antes de reservar memoria."""
    n_aum = n_base * (1 + len(DESPLAZAMIENTOS))
    px = 784
    uint8_gb = n_aum * px * 1 / 1024**3
    float32_gb = n_aum * px * 4 / 1024**3
    # Durante la conversión conviven el uint8 y el float32.
    conversion_gb = uint8_gb + float32_gb
    # Los bloques de distancia están acotados por working_memory por hilo.
    hilos = 10
    bloques_gb = sklearn.get_config()["working_memory"] * hilos / 1024
    pico_gb = max(conversion_gb, float32_gb + bloques_gb) * 1.25   # margen
    return {"n_aumentado": n_aum, "float32_gb": float32_gb,
            "bloques_gb": bloques_gb, "pico_estimado_gb": pico_gb}


def desplazar(imagenes: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Desplaza un lote de imágenes, rellenando con fondo negro."""
    cuadros = imagenes.reshape(-1, 28, 28)
    movidas = nd_shift(cuadros, [0, dy, dx], cval=0, order=0)
    return movidas.reshape(len(imagenes), -1)


def aumentar(X: np.ndarray, y: np.ndarray) -> tuple:
    partes_X, partes_y = [X], [y]
    for dy, dx in DESPLAZAMIENTOS:
        partes_X.append(desplazar(X, dy, dx))
        partes_y.append(y)
    return np.concatenate(partes_X), np.concatenate(partes_y)


def predecir_por_bloques(modelo, X_test: np.ndarray, bloque: int = 1000) -> np.ndarray:
    """Predice en trozos para que el pico no dependa del tamaño del test."""
    salidas = []
    for i in range(0, len(X_test), bloque):
        salidas.append(modelo.predict(X_test[i:i + bloque]))
    return np.concatenate(salidas)


def run(max_train: int | None, solo_check: bool) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()
    if max_train:
        X_train, y_train = X_train[:max_train], y_train[:max_train]

    total = ram_total_gb()
    p = proyectar(len(X_train), len(X_test))
    limite = total * FRACCION_SEGURA

    print(f"\n[augment] RAM total {total:.0f} GB · límite que me impongo {limite:.1f} GB")
    print(f"[augment] {len(X_train):,} imágenes → {p['n_aumentado']:,} tras desplazar")
    print(f"  datos en float32       {p['float32_gb']:5.2f} GB")
    print(f"  bloques de distancias  {p['bloques_gb']:5.2f} GB  (working_memory × hilos)")
    print(f"  pico estimado          {p['pico_estimado_gb']:5.2f} GB")

    if p["pico_estimado_gb"] > limite:
        print(f"\n[augment] ✗ ABORTADO: {p['pico_estimado_gb']:.2f} GB supera el límite.")
        print(f"[augment]   Usa --max-train {int(len(X_train) * limite / p['pico_estimado_gb'] * 0.9):,}")
        return
    print("[augment] ✓ cabe con holgura")

    if solo_check:
        return

    params = {"n_neighbors": 4, "weights": "distance"}
    if MEJOR.exists():
        params = json.loads(MEJOR.read_text())["params"]
        print(f"[augment] usando los mejores parámetros del ejercicio 1: {params}")

    print("\n[augment] Generando las versiones desplazadas...")
    t0 = time.perf_counter()
    Xa, ya = aumentar(X_train, y_train)
    print(f"  {len(Xa):,} imágenes en {time.perf_counter()-t0:.1f}s")

    # Referencia sin aumentar, con los mismos parámetros y el mismo test.
    base = KNeighborsClassifier(**params, n_jobs=-1)
    base.fit(X_train.astype(np.float32), y_train)
    acc_base = float(accuracy_score(y_test, predecir_por_bloques(base, X_test.astype(np.float32))))
    del base
    print(f"\n[augment] Sin aumentar   {acc_base:.4f}")

    modelo = KNeighborsClassifier(**params, n_jobs=-1)
    t0 = time.perf_counter()
    modelo.fit(Xa.astype(np.float32), ya)
    acc = float(accuracy_score(y_test, predecir_por_bloques(modelo, X_test.astype(np.float32))))
    secs = time.perf_counter() - t0

    print(f"[augment] Con aumento    {acc:.4f}   ({acc-acc_base:+.4f})   {secs/60:.1f} min")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="knn_datos_aumentados"):
        mlflow.log_params({"modelo": "KNeighborsClassifier", **params,
                           "n_base": len(X_train), "n_aumentado": len(Xa)})
        mlflow.log_metrics({"exactitud_base": acc_base, "exactitud_aumentado": acc,
                            "mejora": acc - acc_base, "segundos": secs})
        print(f"[augment] registrado en MLflow · run {mlflow.active_run().info.run_id[:8]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aumento de datos por desplazamiento.")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--check", action="store_true", help="Solo proyectar la memoria")
    args = parser.parse_args()
    run(args.max_train, args.check)

"""
Fase 6 — Análisis de errores
Referencia: Géron cap. 3, "Error Analysis"

Saber que el modelo acierta el 91% no dice qué arreglar. La matriz de
confusión de 10×10 sí: muestra **qué dígito se confunde con cuál**.

Dos detalles que cambian por completo lo que se ve:

  Normalizar por fila
    Sin normalizar, la diagonal se lleva todo y las celdas de error quedan
    invisibles. Normalizando por fila, cada celda dice «de todos los 5 reales,
    qué proporción fue clasificada como 3» — que es la pregunta útil.

  Poner la diagonal en cero
    Aun normalizada, la diagonal es tan grande que aplasta la escala de color
    y los errores siguen sin verse. Borrarla deja que el color se reparta solo
    entre los fallos.

Además del mapa, se genera un montaje de las imágenes reales del par que más
se confunde. Ahí se ve si el modelo es tonto o si de verdad son ambiguas.

Uso:
    python src/errors.py
    # o: make errors
"""
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import cross_val_predict

from baseline import EXPERIMENT, SEED, build_pipeline
from features import load_data

REPORTS_DIR = Path("reports")
CV = 3
N_TRAIN = 20_000   # suficiente para que el patrón de errores sea estable


def run() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, _, y_train, _ = load_data()
    X, y = X_train[:N_TRAIN], y_train[:N_TRAIN]

    print(f"\n[errors] Prediciendo con validación cruzada sobre {len(X):,} imágenes...")
    pipe = build_pipeline(SGDClassifier(random_state=SEED))
    preds = cross_val_predict(pipe, X, y, cv=CV, n_jobs=-1)

    cm = confusion_matrix(y, preds)
    exactitud = (preds == y).mean()
    print(f"[errors] exactitud {exactitud:.4f} · {int((preds != y).sum()):,} errores")

    # Tasa de error por fila, sin la diagonal.
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    errores = cm_norm.copy()
    np.fill_diagonal(errores, 0)

    par = np.unravel_index(errores.argmax(), errores.shape)
    real, predicho = int(par[0]), int(par[1])
    tasa = errores[par]

    print(f"\n[errors] Los cinco pares que más se confunden")
    planos = [(errores[i, j], i, j) for i in range(10) for j in range(10) if i != j]
    for t, i, j in sorted(planos, reverse=True)[:5]:
        print(f"  {i} clasificado como {j}   {t:6.2%}  ({cm[i, j]:,} casos)")

    print(f"\n[errors] Peor par: el {real} confundido con el {predicho} ({tasa:.1%})")

    _plot_matriz(cm, cm_norm, errores)
    _plot_par(X, y, preds, real, predicho)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="analisis_errores"):
        mlflow.log_params({"tarea": "multiclase_10", "modelo": "sgd",
                           "n_train": N_TRAIN, "cv": CV})
        mlflow.log_metrics({"exactitud": float(exactitud),
                            "peor_par_real": real,
                            "peor_par_predicho": predicho,
                            "peor_par_tasa": float(tasa)})
        for f in ("confusion.png", "peor_par.png"):
            mlflow.log_artifact(str(REPORTS_DIR / f))
        print(f"[errors] registrado en MLflow · run {mlflow.active_run().info.run_id[:8]}")


def _plot_matriz(cm, cm_norm, errores) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))

    ConfusionMatrixDisplay(cm_norm).plot(
        ax=axes[0], cmap="Greens", colorbar=False, values_format=".0%")
    axes[0].set_title("Normalizada por fila\nla diagonal aplasta todo lo demás",
                      fontsize=11, fontweight="bold")

    ConfusionMatrixDisplay(errores).plot(
        ax=axes[1], cmap="Oranges", colorbar=False, values_format=".0%")
    axes[1].set_title("Sin la diagonal\nahora sí se ven los errores",
                      fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Predicho", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)
        ax.tick_params(labelsize=9)
        for t in ax.texts:
            t.set_fontsize(7)

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion.png", dpi=150)
    plt.close(fig)


def _plot_par(X, y, preds, a: int, b: int, n: int = 25) -> None:
    """Montaje de los cuatro cuadrantes del par que más se confunde."""
    def bloque(real, pred):
        idx = np.flatnonzero((y == real) & (preds == pred))[:n]
        imgs = X[idx].reshape(-1, 28, 28)
        lado = int(np.ceil(np.sqrt(n)))
        canvas = np.zeros((lado * 28, lado * 28), dtype=np.uint8)
        for k, im in enumerate(imgs):
            r, c = divmod(k, lado)
            canvas[r*28:(r+1)*28, c*28:(c+1)*28] = im
        return canvas, len(idx)

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 8))
    combos = [(a, a), (a, b), (b, a), (b, b)]
    titulos = [f"{a} → {a}  correcto", f"{a} → {b}  error",
               f"{b} → {a}  error", f"{b} → {b}  correcto"]

    for ax, (r, p), tit in zip(axes.ravel(), combos, titulos):
        canvas, cuantos = bloque(r, p)
        ax.imshow(canvas, cmap="binary")
        color = "#c2571f" if r != p else "#00937f"
        ax.set_title(f"{tit}  ({cuantos})", fontsize=10, color=color, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"El {a} y el {b}: aciertos arriba-izquierda y abajo-derecha, errores en diagonal",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "peor_par.png", dpi=150)
    plt.close(fig)
    print(f"[errors] gráficas → {REPORTS_DIR}/")


if __name__ == "__main__":
    run()

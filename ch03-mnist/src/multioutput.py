"""
Fase 7 — Multietiqueta y multisalida
Referencia: Géron cap. 3, "Multilabel Classification" y "Multioutput Classification"

Dos generalizaciones de lo anterior, cada una rompiendo un supuesto distinto.

  Multietiqueta — varias respuestas por imagen, cada una binaria
    En vez de «¿cuál dígito?», dos preguntas a la vez: ¿es grande (≥7)? y
    ¿es impar? Una imagen puede ser las dos, una, o ninguna. El caso real
    típico es reconocimiento facial: en una foto con tres personas conocidas,
    la respuesta correcta son tres etiquetas encendidas.

  Multisalida — cada respuesta ya no es binaria sino multiclase
    Llevado al extremo: la salida es una imagen completa. Se entrena con
    imágenes ruidosas como entrada y limpias como objetivo, y el modelo
    aprende a quitar el ruido. Son 784 salidas, cada una con 256 valores
    posibles.

El multisalida deja clara una frontera borrosa: quitar ruido de una imagen
suena a regresión, no a clasificación. La distinción no siempre es nítida.

Ambos usan k-vecinos, que no entrena sino que memoriza — por eso trabajan
sobre submuestras y no sobre las 60,000 imágenes.

Uso:
    python src/multioutput.py
    # o: make multioutput
"""
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from baseline import EXPERIMENT, SEED, build_pipeline
from features import load_data

REPORTS_DIR = Path("reports")
N_MULTILABEL = 10_000
N_DENOISE = 10_000
CV = 3
RUIDO = 100        # amplitud del ruido uniforme añadido a cada píxel


def multietiqueta(X_train, y_train) -> dict:
    """Dos preguntas binarias a la vez sobre la misma imagen."""
    X, y = X_train[:N_MULTILABEL], y_train[:N_MULTILABEL]

    grande = (y >= 7)
    impar = (y % 2 == 1)
    Y = np.c_[grande, impar]

    print(f"\n[multi] Multietiqueta sobre {len(X):,} imágenes")
    print(f"  ¿es grande (≥7)?   {grande.mean():.1%} de las imágenes")
    print(f"  ¿es impar?         {impar.mean():.1%} de las imágenes")

    pipe = build_pipeline(KNeighborsClassifier())
    preds = cross_val_predict(pipe, X, Y, cv=CV, n_jobs=-1)

    f1_grande = f1_score(Y[:, 0], preds[:, 0])
    f1_impar = f1_score(Y[:, 1], preds[:, 1])

    # macro promedia sin pesar por frecuencia; weighted sí. Con etiquetas
    # desbalanceadas dan números distintos y conviene mirar las dos.
    f1_macro = f1_score(Y, preds, average="macro")
    f1_weighted = f1_score(Y, preds, average="weighted")

    peor = "es grande" if f1_grande < f1_impar else "es impar"
    marca_g = "   ← más difícil" if f1_grande < f1_impar else ""
    marca_i = "   ← más difícil" if f1_impar < f1_grande else ""

    print(f"\n  F1 «es grande»     {f1_grande:.4f}{marca_g}")
    print(f"  F1 «es impar»      {f1_impar:.4f}{marca_i}")
    print(f"  F1 macro           {f1_macro:.4f}   promedia las dos por igual")
    print(f"  F1 ponderado       {f1_weighted:.4f}   pesa por frecuencia")

    print(f"\n  La pregunta más difícil resultó ser «{peor}», que no era lo")
    print("  esperado: parece la más simple de las dos. Ninguna de las dos")
    print("  corresponde a una forma visual — el modelo tiene que reconocer")
    print("  el dígito y después responder, así que hereda las confusiones")
    print("  del análisis de errores. Cuál sale peor depende de qué pares")
    print("  confunde y de qué lado de cada pregunta caen.")

    return {"f1_grande": float(f1_grande), "f1_impar": float(f1_impar),
            "f1_macro": float(f1_macro), "f1_weighted": float(f1_weighted)}


def multisalida(X_train, X_test) -> dict:
    """Quitar ruido: la salida es una imagen completa, no una etiqueta."""
    rng = np.random.default_rng(SEED)

    X = X_train[:N_DENOISE]
    ruido = rng.integers(0, RUIDO, size=X.shape, dtype=np.int16)
    X_ruidoso = np.clip(X.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

    Xt = X_test[:20]
    ruido_t = rng.integers(0, RUIDO, size=Xt.shape, dtype=np.int16)
    Xt_ruidoso = np.clip(Xt.astype(np.int16) + ruido_t, 0, 255).astype(np.uint8)

    print(f"\n[multi] Multisalida: quitar ruido, {len(X):,} imágenes de entrenamiento")
    print(f"  entrada  784 píxeles ruidosos")
    print(f"  salida   784 píxeles limpios, cada uno con 256 valores posibles")

    # Sin el Pipeline de escalado: aquí la salida son píxeles en su escala
    # original, y escalar la entrada sin escalar la salida las desalinearía.
    knn = KNeighborsClassifier()
    knn.fit(X_ruidoso, X)
    limpias = knn.predict(Xt_ruidoso)

    # Error absoluto medio por píxel, contra el original sin ruido.
    mae_antes = float(np.abs(Xt_ruidoso.astype(int) - Xt.astype(int)).mean())
    mae_despues = float(np.abs(limpias.astype(int) - Xt.astype(int)).mean())
    print(f"\n  error medio por píxel antes   {mae_antes:6.2f}")
    print(f"  error medio por píxel después {mae_despues:6.2f}   ({mae_despues-mae_antes:+.2f})")

    _plot_denoise(Xt, Xt_ruidoso, limpias)
    return {"mae_con_ruido": mae_antes, "mae_limpiada": mae_despues}


def _plot_denoise(originales, ruidosas, limpias, n: int = 6) -> None:
    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 5))
    filas = [
        (ruidosas, "Con ruido", "#c2571f"),
        (limpias, "Limpiada por el modelo", "#00937f"),
        (originales, "Original", "#7e8783"),
    ]
    for r, (datos, titulo, color) in enumerate(filas):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(datos[c].reshape(28, 28), cmap="binary")
            ax.axis("off")
        axes[r, 0].set_ylabel(titulo)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for s in axes[r, 0].spines.values():
            s.set_visible(False)
        axes[r, 0].set_ylabel(titulo, fontsize=9, color=color, fontweight="bold")

    fig.suptitle("La salida del modelo es una imagen completa, no una etiqueta",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "denoise.png", dpi=150)
    plt.close(fig)
    print(f"  gráfica → {REPORTS_DIR}/denoise.png")


def run() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, _ = load_data()

    m1 = multietiqueta(X_train, y_train)
    m2 = multisalida(X_train, X_test)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="multietiqueta_multisalida"):
        mlflow.log_params({"modelo": "KNeighborsClassifier",
                           "n_multietiqueta": N_MULTILABEL,
                           "n_multisalida": N_DENOISE,
                           "amplitud_ruido": RUIDO, "cv": CV})
        mlflow.log_metrics({**m1, **m2})
        mlflow.log_artifact(str(REPORTS_DIR / "denoise.png"))
        print(f"\n[multi] registrado en MLflow · run {mlflow.active_run().info.run_id[:8]}")


if __name__ == "__main__":
    run()

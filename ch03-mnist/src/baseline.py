"""
Fase 3 — Detector binario y la trampa de la exactitud
Referencia: Géron cap. 3, "Training a Binary Classifier" y "Performance Measures"

Entrena un clasificador que responde una sola pregunta: ¿esta imagen es un 5?

El punto del ejercicio no es el clasificador, es descubrir que **la exactitud
no sirve para medirlo**. Solo el 9% de las imágenes son cincos, así que un
modelo que responda "no es un 5" a todo acierta el 91% de las veces sin haber
aprendido nada. Ese modelo tonto se entrena aquí a propósito, para tenerlo al
lado del real.

Lo que sí mide:

  Matriz de confusión   qué confundió con qué, no cuánto falló
  Precisión             de lo que llamé 5, ¿cuánto era 5?
  Exhaustividad         de todos los cincos que había, ¿cuántos encontré?
  F1                    media armónica de las dos, penaliza el desequilibrio

Todas se calculan con `cross_val_predict`: cada imagen recibe la predicción
de un modelo que no la vio al entrenarse. Medirlas sobre el propio conjunto
de entrenamiento daría números inflados.

Uso:
    python src/baseline.py --digit 5
    # o: make baseline
"""
import argparse

import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from features import load_data

CV = 3
SEED = 42
EXPERIMENT = "mnist-clasificacion"


def build_pipeline(estimator) -> Pipeline:
    """Escalado a 0–1 + modelo.

    Dividir entre 255 en vez de estandarizar: los píxeles ya comparten escala
    y unidad, y así los ceros del fondo siguen siendo ceros. Estandarizar
    columna a columna rompería eso — los píxeles del borde son siempre 0 y
    tienen desviación cero.
    """
    scale = FunctionTransformer(lambda X: X / 255.0, feature_names_out="one-to-one")
    return Pipeline([("scale", scale), ("model", estimator)])


def confusion_table(cm: np.ndarray, digit: int) -> str:
    (tn, fp), (fn, tp) = cm
    return (
        f"                    predijo «no {digit}»   predijo «{digit}»\n"
        f"    es «no {digit}»        {tn:8,d}        {fp:8,d}   ← falsos positivos\n"
        f"    es «{digit}»           {fn:8,d}        {tp:8,d}\n"
        f"                        ↑ falsos negativos"
    )


def run(digit: int) -> None:
    X_train, _, y_train, _ = load_data()
    y_bin = (y_train == digit)

    prevalencia = y_bin.mean()
    print(f"\n[baseline] Detectar el dígito {digit}")
    print(f"[baseline] {y_bin.sum():,} de {len(y_bin):,} imágenes lo son "
          f"({prevalencia:.1%} del total)")

    real = build_pipeline(SGDClassifier(random_state=SEED))
    tonto = build_pipeline(DummyClassifier(strategy="most_frequent"))

    print(f"\n[baseline] Exactitud con validación cruzada ({CV} pliegues)...")
    acc_real = cross_val_score(real, X_train, y_bin, cv=CV, scoring="accuracy", n_jobs=-1)
    acc_tonto = cross_val_score(tonto, X_train, y_bin, cv=CV, scoring="accuracy", n_jobs=-1)

    print(f"  clasificador real   {acc_real.mean():.4f}")
    print(f"  siempre «no {digit}»     {acc_tonto.mean():.4f}   ← no aprendió nada")
    print(f"\n  La diferencia es de solo {(acc_real.mean()-acc_tonto.mean())*100:.1f} puntos.")
    print("  Por eso la exactitud no sirve aquí: casi todo el mérito viene de")
    print("  que la clase positiva es rara, no de que el modelo la reconozca.")

    # cross_val_predict: predicciones limpias para cada imagen, hechas por un
    # modelo que no la tenía en su pliegue de entrenamiento.
    preds = cross_val_predict(real, X_train, y_bin, cv=CV, n_jobs=-1)
    cm = confusion_matrix(y_bin, preds)
    precision = precision_score(y_bin, preds)
    recall = recall_score(y_bin, preds)
    f1 = f1_score(y_bin, preds)

    print("\n[baseline] Matriz de confusión del clasificador real")
    print(confusion_table(cm, digit))

    print(f"\n[baseline] Métricas que sí distinguen")
    print(f"  precisión       {precision:.4f}   de lo que llamó {digit}, cuánto lo era")
    print(f"  exhaustividad   {recall:.4f}   de los {digit} que había, cuántos encontró")
    print(f"  F1              {f1:.4f}")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"binario_{digit}_sgd"):
        mlflow.log_params({"tarea": f"detectar_{digit}", "modelo": "SGDClassifier",
                           "cv": CV, "prevalencia": round(float(prevalencia), 4)})
        mlflow.log_metrics({
            "exactitud": float(acc_real.mean()),
            "exactitud_tonto": float(acc_tonto.mean()),
            "precision": float(precision),
            "exhaustividad": float(recall),
            "f1": float(f1),
            "falsos_positivos": int(cm[0, 1]),
            "falsos_negativos": int(cm[1, 0]),
        })
        print(f"\n[baseline] registrado en MLflow · run {mlflow.active_run().info.run_id[:8]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detector binario de un dígito.")
    parser.add_argument("--digit", type=int, default=5, choices=range(10))
    args = parser.parse_args()
    run(args.digit)

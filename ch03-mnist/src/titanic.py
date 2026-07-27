"""
Ejercicio 3 — Titanic
Referencia: Géron cap. 3, ejercicio 3 ("Tackle the Titanic dataset")

Predecir quién sobrevivió al naufragio. Después de 70,000 imágenes de píxeles
homogéneos, este dataset vuelve a lo desordenado: 1,309 pasajeros con edades
faltantes, camarotes sin registrar, y columnas de tipos mezclados.

Es el ejercicio que conecta los dos capítulos. Toda la maquinaria de
preprocesamiento del capítulo 2 —imputar, escalar, codificar categorías dentro
del Pipeline— se aplica igual; lo que cambia son las métricas con las que se
juzga el resultado, que son las del capítulo 3.

Una nota sobre el desbalance: sobrevivió el 38%. No es el 9% de los cincos,
así que la exactitud engaña menos aquí — pero sigue sin decir si el modelo
falla al predecir muertes o supervivencias, y esas dos cosas no son iguales.

Uso:
    python src/titanic.py
    # o: make titanic
"""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from baseline import EXPERIMENT, SEED

RAW = Path("data/raw/titanic.csv")

# shuffle=True no es opcional aquí. El dataset viene ordenado por clase de
# pasaje: sin barajar, el primer pliegue es 100% primera clase y los dos
# últimos 100% tercera, así que el modelo se entrena con un perfil de pasajero
# y se evalúa con otro. La estratificación por defecto equilibra la
# supervivencia, no la clase, y por eso no lo detecta.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Se descartan a propósito, no por olvido:
#   name, ticket, cabin, boat, body, home.dest → identificadores o texto libre
#   boat y body además son fuga directa: solo tienen valor si sobrevivió
#   (subió a un bote) o si no (se recuperó el cuerpo).
NUMERICAS = ["age", "sibsp", "parch", "fare"]
CATEGORICAS = ["pclass", "sex", "embarked"]
TARGET = "survived"

MODELOS = {
    "regresion_logistica": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    "bosque":              lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "svc":                 lambda: SVC(random_state=SEED),
}


def cargar() -> pd.DataFrame:
    if RAW.exists():
        return pd.read_csv(RAW)

    from sklearn.datasets import fetch_openml

    print("[titanic] descargando desde OpenML...")
    df = fetch_openml("titanic", version=1, as_frame=True, parser="auto").frame
    RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW, index=False)
    return df


def construir_pipeline(estimator) -> Pipeline:
    numericas = Pipeline([
        # La mediana y no la media: la tarifa tiene una cola larguísima —
        # unos pocos pagaron muchísimo más que el resto.
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numericas, NUMERICAS),
        ("cat", categoricas, CATEGORICAS),
    ])
    return Pipeline([("prep", prep), ("model", estimator)])


def run() -> None:
    df = cargar()
    y = df[TARGET].astype(int).values
    X = df[NUMERICAS + CATEGORICAS]

    print(f"\n[titanic] {len(df):,} pasajeros · sobrevivió el {y.mean():.1%}")
    faltantes = X.isna().sum()
    print("[titanic] valores faltantes:",
          ", ".join(f"{c}={n}" for c, n in faltantes[faltantes > 0].items()) or "ninguno")

    print(f"\n  {'modelo':22s} {'exactitud':>11s} {'precisión':>11s} {'exhaust.':>10s}")

    mlflow.set_experiment(EXPERIMENT)
    resultados = {}
    for nombre, hacer in MODELOS.items():
        pipe = construir_pipeline(hacer())
        acc = cross_val_score(pipe, X, y, cv=CV, scoring="accuracy", n_jobs=-1).mean()
        preds = cross_val_predict(pipe, X, y, cv=CV, n_jobs=-1)
        prec = precision_score(y, preds)
        rec = recall_score(y, preds)
        resultados[nombre] = (acc, prec, rec, preds)
        print(f"  {nombre:22s} {acc:11.4f} {prec:11.4f} {rec:10.4f}")

        with mlflow.start_run(run_name=f"titanic_{nombre}"):
            mlflow.log_params({"dataset": "titanic", "modelo": nombre,
                               "cv": "StratifiedKFold(5, shuffle)", "n": len(df)})
            mlflow.log_metrics({"exactitud": float(acc), "precision": float(prec),
                                "exhaustividad": float(rec)})

    mejor = max(resultados, key=lambda k: resultados[k][0])
    _, _, _, preds = resultados[mejor]
    (tn, fp), (fn, tp) = confusion_matrix(y, preds)

    print(f"\n[titanic] Matriz de confusión — {mejor}")
    print("                    predijo murió   predijo sobrevivió")
    print(f"    murió              {tn:8,d}          {fp:8,d}")
    print(f"    sobrevivió         {fn:8,d}          {tp:8,d}")

    print(f"\n  {fn:,} personas que sobrevivieron fueron dadas por muertas,")
    print(f"  y {fp:,} que murieron fueron dadas por vivas. La exactitud las")
    print("  cuenta igual; para un caso real casi nunca cuestan lo mismo.")

    _tasas_por_grupo(df, y)


def _tasas_por_grupo(df: pd.DataFrame, y: np.ndarray) -> None:
    """De dónde sale la señal: qué grupos sobrevivieron más."""
    print("\n[titanic] Supervivencia real por grupo")
    for col in ("sex", "pclass"):
        tasas = df.assign(**{TARGET: y}).groupby(col, observed=True)[TARGET].agg(["mean", "size"])
        print(f"  por {col}:")
        for valor, fila in tasas.iterrows():
            print(f"    {str(valor):10s} {fila['mean']:6.1%}  ({int(fila['size']):,} personas)")


if __name__ == "__main__":
    run()

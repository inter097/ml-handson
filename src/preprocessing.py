"""
Preprocesamiento compartido
Referencia: Géron cap. 2 "Prepare the Data for ML Algorithms"

Un solo lugar define cómo se transforman los datos crudos, y train.py,
tune.py y analysis.py lo reutilizan. Así no hay tres versiones del
preprocesamiento que puedan desincronizarse.

El ColumnTransformer aplica ramas distintas según el tipo de columna:

  numéricas    → SimpleImputer(median) + StandardScaler
  categóricas  → OneHotEncoder

Todo vive dentro del Pipeline del modelo, así que se reajusta en cada fold
de la validación cruzada y viaja dentro del artefacto guardado en MLflow.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED_DIR = Path("data/processed")

CATEGORICAL = ["ocean_proximity"]

# Tipos que MLflow/skops debe aceptar al serializar el Pipeline completo.
# Vive aquí para que train.py y tune.py no mantengan dos copias distintas.
SKOPS_TRUSTED = [
    "numpy.dtype",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBRegressor",
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
    "sklearn.pipeline.Pipeline",
    "sklearn.compose._column_transformer.ColumnTransformer",
    "sklearn.impute._base.SimpleImputer",
    "sklearn.preprocessing._data.StandardScaler",
    "sklearn.preprocessing._encoders.OneHotEncoder",
]


def load_data() -> tuple:
    """Devuelve DataFrames (no arrays) — el ColumnTransformer usa los nombres."""
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").values.ravel()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").values.ravel()
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in CATEGORICAL if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    numeric_branch = Pipeline([
        # Los 207 NaN de total_bedrooms llegan aquí como AveBedrms/bedrooms_ratio.
        # Mediana y no media: la distribución tiene cola larga.
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformers = [("num", numeric_branch, numeric)]
    if categorical:
        # handle_unknown="ignore": la categoría ISLAND tiene 5 filas en total y
        # puede no aparecer en algún fold de la CV. Sin esto, reventaría.
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        )

    return ColumnTransformer(transformers)


def build_pipeline(estimator, X: pd.DataFrame) -> Pipeline:
    """Preprocesamiento + modelo en un solo objeto serializable."""
    return Pipeline([
        ("prep", build_preprocessor(X)),
        ("model", estimator),
    ])


def feature_names(pipeline: Pipeline) -> list:
    """Nombres de las columnas que ve el modelo, ya expandido el one-hot."""
    return list(pipeline.named_steps["prep"].get_feature_names_out())

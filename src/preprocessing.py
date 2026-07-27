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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED_DIR = Path("data/processed")

CATEGORICAL = ["ocean_proximity"]
GEO = ["Latitude", "Longitude"]

# Tipos que MLflow/skops debe aceptar al serializar el Pipeline completo.
# Vive aquí para que train.py y tune.py no mantengan dos copias distintas.
SKOPS_TRUSTED = [
    "numpy.dtype",
    "preprocessing.ClusterSimilarity",
    "sklearn.cluster._kmeans.KMeans",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBRegressor",
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
    "sklearn.pipeline.Pipeline",
    "sklearn.compose._column_transformer.ColumnTransformer",
    "sklearn.impute._base.SimpleImputer",
    "sklearn.preprocessing._data.StandardScaler",
    "sklearn.preprocessing._encoders.OneHotEncoder",
]


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """Convierte lat/long en similitud a N barrios representativos.

    Referencia: Géron cap. 2, "Custom Transformers".

    Problema que resuelve: un árbol solo puede partir el espacio con cortes
    rectangulares sobre latitud y longitud, así que aproximar "cerca de San
    Francisco" le cuesta muchos splits y nunca queda fino.

    Qué hace: KMeans encuentra N centros geográficos y cada fila se reemplaza
    por su similitud RBF a cada centro — decae suave con la distancia. El
    modelo recibe N features tipo "qué tan cerca estás del barrio 3".

    gamma controla qué tan rápido decae: alto = solo cuenta lo muy cercano.
    """

    def __init__(self, n_clusters: int = 10, gamma: float = 1.0, random_state: int = 42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, input_features=None):
        return [f"cluster_{i}_similarity" for i in range(self.n_clusters)]


def load_data() -> tuple:
    """Devuelve DataFrames (no arrays) — el ColumnTransformer usa los nombres."""
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").values.ravel()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").values.ravel()
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame, use_geo_clusters: bool = True) -> ColumnTransformer:
    categorical = [c for c in CATEGORICAL if c in X.columns]
    geo = [c for c in GEO if c in X.columns] if use_geo_clusters else []
    numeric = [c for c in X.columns if c not in categorical]

    numeric_branch = Pipeline([
        # Los 207 NaN de total_bedrooms llegan aquí como AveBedrms/bedrooms_ratio.
        # Mediana y no media: la distribución tiene cola larga.
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Lat/Long siguen en la rama numérica además de generar los clusters: las
    # coordenadas crudas y la similitud a centroides aportan cosas distintas y
    # los árboles pueden usar ambas.
    transformers = [("num", numeric_branch, numeric)]
    if geo:
        transformers.append(("geo", ClusterSimilarity(), geo))
    if categorical:
        # handle_unknown="ignore": la categoría ISLAND tiene 5 filas en total y
        # puede no aparecer en algún fold de la CV. Sin esto, reventaría.
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        )

    return ColumnTransformer(transformers)


def build_pipeline(estimator, X: pd.DataFrame, use_geo_clusters: bool = True) -> Pipeline:
    """Preprocesamiento + modelo en un solo objeto serializable."""
    return Pipeline([
        ("prep", build_preprocessor(X, use_geo_clusters=use_geo_clusters)),
        ("model", estimator),
    ])


def feature_names(pipeline: Pipeline) -> list:
    """Nombres de las columnas que ve el modelo, ya expandido el one-hot."""
    return list(pipeline.named_steps["prep"].get_feature_names_out())

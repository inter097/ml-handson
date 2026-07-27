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

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

PROCESSED_DIR = Path("data/processed")

CATEGORICAL = ["ocean_proximity"]
GEO = ["Latitude", "Longitude"]

# Tipos que MLflow/skops debe aceptar al serializar el Pipeline completo.
# Vive aquí para que train.py y tune.py no mantengan dos copias distintas.
SKOPS_TRUSTED = [
    "numpy.dtype",
    "preprocessing.ClusterSimilarity",
    "preprocessing.KNNGeoFeature",
    "preprocessing.StandardScalerClone",
    "sklearn.cluster._kmeans.KMeans",
    "sklearn.ensemble._forest.RandomForestRegressor",
    "sklearn.feature_selection._from_model.SelectFromModel",
    "sklearn.neighbors._regression.KNeighborsRegressor",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBRegressor",
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
    "sklearn.pipeline.Pipeline",
    "sklearn.compose._column_transformer.ColumnTransformer",
    "sklearn.impute._base.SimpleImputer",
    "sklearn.preprocessing._data.StandardScaler",
    "sklearn.preprocessing._encoders.OneHotEncoder",
]


class StandardScalerClone(BaseEstimator, TransformerMixin):
    """StandardScaler reimplementado desde cero.

    Referencia: Géron cap. 2, ejercicio 6.

    No aporta nada al modelo — el de sklearn hace lo mismo y mejor. El punto
    es entender qué contrato cumple un transformer para poder escribir los
    propios: heredar de BaseEstimator (da get_params/set_params, y con eso
    funciona la búsqueda de hiperparámetros) y de TransformerMixin (da
    fit_transform gratis).

    Detalles que sí importan y suelen olvidarse:
      - los atributos aprendidos llevan guion bajo final (mean_, scale_)
      - fit guarda n_features_in_ para detectar formas incompatibles después
      - transform valida que el modelo esté ajustado antes de usarse
    """

    def __init__(self, with_mean: bool = True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        # Una columna constante tiene desviación 0; dividir entre ella daría
        # inf. sklearn hace lo mismo: la deja pasar sin escalar.
        self.scale_ = np.where(scale == 0, 1.0, scale)
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                f"Se ajustó con {self.n_features_in_} columnas y se recibieron {X.shape[1]}"
            )
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = check_array(X) * self.scale_
        return X + self.mean_ if self.with_mean else X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class KNNGeoFeature(BaseEstimator, TransformerMixin):
    """Predicción de un k-NN sobre lat/long, usada como feature.

    Referencia: Géron cap. 2, ejercicio 4.

    La idea: un k-NN sobre coordenadas responde "¿cuánto cuestan las casas
    vecinas?". Esa respuesta entra al modelo como una columna más.

    El riesgo es el sobreajuste: si el k-NN se entrena y predice sobre las
    mismas filas, cada casa ve su propio precio entre sus vecinos y la feature
    filtra el target. Por eso en fit se usa `cross_val_predict` — cada fila
    recibe la predicción de un k-NN que NO la vio. En transform, sobre datos
    nuevos, ya se puede usar el k-NN entrenado con todo.
    """

    def __init__(self, n_neighbors: int = 10, weights: str = "distance", cv: int = 5):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.cv = cv

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("KNNGeoFeature necesita y para entrenar el k-NN")
        X = check_array(X)
        self.knn_ = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights)
        # Predicciones fuera de fold para las filas de entrenamiento: evita que
        # cada casa se vea a sí misma reflejada en la feature.
        self.oof_ = cross_val_predict(clone(self.knn_), X, y, cv=self.cv).reshape(-1, 1)
        self.knn_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.knn_.predict(check_array(X)).reshape(-1, 1)

    def fit_transform(self, X, y=None, **kwargs):
        # Sobre el train devuelve las predicciones fuera de fold, no las del
        # k-NN completo. TransformerMixin llamaría fit().transform() y filtraría.
        self.fit(X, y)
        return self.oof_

    def get_feature_names_out(self, input_features=None):
        return np.array(["knn_geo_price"], dtype=object)


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


def build_preprocessor(
    X: pd.DataFrame,
    use_geo_clusters: bool = True,
    use_knn_geo: bool = False,
) -> ColumnTransformer:
    categorical = [c for c in CATEGORICAL if c in X.columns]
    geo_cols = [c for c in GEO if c in X.columns]
    geo = geo_cols if use_geo_clusters else []
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
    if use_knn_geo and geo_cols:
        transformers.append(("knn", KNNGeoFeature(), geo_cols))
    if categorical:
        # handle_unknown="ignore": la categoría ISLAND tiene 5 filas en total y
        # puede no aparecer en algún fold de la CV. Sin esto, reventaría.
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        )

    return ColumnTransformer(transformers)


def build_pipeline(
    estimator,
    X: pd.DataFrame,
    use_geo_clusters: bool = True,
    use_knn_geo: bool = False,
    select_features: bool = False,
    select_threshold: str = "median",
) -> Pipeline:
    """Preprocesamiento + modelo en un solo objeto serializable.

    select_features añade el paso de selección del ejercicio 3 de Géron:
    un RandomForest mide la importancia de cada columna ya transformada y
    descarta las que quedan por debajo del umbral. Va después del
    preprocesamiento porque solo entonces existen las columnas del one-hot
    y las similitudes de cluster.
    """
    steps = [("prep", build_preprocessor(X, use_geo_clusters, use_knn_geo))]
    if select_features:
        steps.append(("select", SelectFromModel(
            RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            threshold=select_threshold,
        )))
    steps.append(("model", estimator))
    return Pipeline(steps)


def feature_names(pipeline: Pipeline) -> list:
    """Nombres de las columnas que ve el modelo, ya expandido el one-hot."""
    return list(pipeline.named_steps["prep"].get_feature_names_out())

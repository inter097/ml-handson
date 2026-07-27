# ml-handson-california-housing

**[→ Ver el caso de estudio](https://ml-handson-california-housing-26e4.vercel.app)**

Pipeline de Machine Learning completo sobre el dataset **California Housing**, guiado por el libro
*Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — Aurélien Géron (cap. 2).

El objetivo es aprender cada fase del ciclo de vida de un modelo ML siguiendo las mejores prácticas
de la industria: código separado por responsabilidad, pipeline reproducible con un solo comando,
y tracking automático de experimentos con MLflow.

---

## Estructura del proyecto

```
.
├── src/
│   ├── data.py           # Fase 1 — Descarga del dataset
│   ├── features.py       # Fase 2 — Feature engineering + split
│   ├── preprocessing.py  # ColumnTransformer compartido (imputar/escalar/one-hot)
│   ├── train.py          # Fase 3a — Entrenamiento baseline + logging MLflow
│   ├── tune.py           # Fase 3b — RandomizedSearchCV
│   ├── analysis.py       # Importancia de features + curvas de aprendizaje
│   ├── evaluate.py       # Fase 4 — Evaluación final del mejor modelo
│   └── predict.py        # Fase 5 — Inferencia sobre datos crudos
├── notebooks/
│   └── 01_eda.ipynb   # Exploración visual (no forma parte del pipeline)
├── data/              # gitignored — generado con `make data`
│   ├── raw/
│   └── processed/
├── site/              # Caso de estudio estático — generado con `make site`
├── reports/           # Sí va en git — generado con `make evaluate`
├── mlruns/            # gitignored — gestionado por MLflow
├── Makefile           # Pipeline completo
├── requirements.txt
└── .gitignore
```

---

## Setup (una sola vez)

```bash
git clone https://github.com/inter097/ml-handson-california-housing.git
cd ml-handson-california-housing
make setup
source venv/bin/activate   # Mac/Linux
```

---

## Pipeline completo

```bash
make all          # descarga datos → procesa features → entrena 3 modelos (baseline)
make ui           # abre http://localhost:5000 para comparar runs
```

O paso a paso:

```bash
make data           # Descarga California Housing → data/raw/housing.parquet
make features       # Limpieza + feature engineering → data/processed/
make train-all      # Baseline: 3 modelos con params fijos
make tune-all       # Tuning: busca mejores hiperparámetros (RandomizedSearchCV)
make ui             # Compara todos los runs en la UI
```

---

## Exploración (EDA)

El notebook es solo para exploración visual, no forma parte del pipeline automatizado.

```bash
source venv/bin/activate
jupyter notebook notebooks/01_eda.ipynb
```

Corre `make data` antes de abrir el notebook.

---

## Fases del pipeline

### Fase 1 — Obtener datos (`src/data.py`)
*Géron §2: "Get the Data"*

Descarga el CSV original del libro desde `ageron/data` y lo persiste en Parquet.
20,640 filas y 10 columnas (9 features + target `MedHouseVal`).

**Por qué el CSV del libro y no `fetch_california_housing`:** la versión de sklearn
viene recortada — descarta la columna categórica `ocean_proximity` e imputa en
silencio los 207 faltantes de `total_bedrooms`. El libro enseña justamente a tratar
esas dos cosas, así que se usa el CSV crudo. Las columnas se renombran al estilo
sklearn y el target se divide entre 100,000 para mantener la escala de las métricas.

⚠️ **Ojo al comparar runs viejos:** el CSV del libro trae las filas en distinto orden
que sklearn, así que con la misma semilla el split cae en casas distintas. Las métricas
de runs anteriores a este cambio **no son comparables** con las de ahora. En MLflow
distínguelos por el param `has_ocean_proximity`.

### Fase 2 — Features (`src/features.py`)
*Géron §2: "Prepare the Data for ML Algorithms"*

- **Feature engineering**: ratios derivados (`rooms_per_household`, `bedrooms_ratio`, `population_per_household`)
- **Split train/test**: 80/20 **estratificado por categoría de ingreso**, semilla fija
- **Sin transformar**: los splits se guardan crudos, incluidos los NaN y la columna
  categórica. Todo el preprocesamiento vive en `src/preprocessing.py`, dentro del
  `Pipeline` de cada modelo

**Por qué el preprocesamiento no va en esta fase:** si se escalara/imputara aquí, esos
pasos verían el train set completo antes de la validación cruzada, y cada fold de
validación quedaría contaminado con estadísticas calculadas sobre él mismo — el
`cv_rmse` saldría optimista. Dentro del `Pipeline`, sklearn los reajusta en cada fold.

Beneficio extra: el artefacto guardado en MLflow contiene preprocesamiento + modelo.
Quien lo cargue le pasa datos crudos directamente, sin replicar nada.

**Por qué el split va estratificado:** con un split aleatorio puro, el test set puede
quedar con más casas ricas que la población real y el RMSE mediría una California que
no existe. `MedInc` se parte en 5 categorías y se fuerza la misma proporción en train
y test. `make features` imprime el sesgo que esto evita — en este dataset, el split
aleatorio subrepresenta la categoría más pobre en **−9.4%**.

### `src/preprocessing.py` — el `ColumnTransformer`

Un solo módulo define las transformaciones y lo reutilizan `train.py`, `tune.py` y
`analysis.py`, así no hay tres copias que se desincronicen:

| Rama | Columnas | Pasos |
|---|---|---|
| `num` | las 11 numéricas | `SimpleImputer(median)` → `StandardScaler` |
| `geo` | `Latitude`, `Longitude` | `ClusterSimilarity` |
| `cat` | `ocean_proximity` | `OneHotEncoder(handle_unknown="ignore")` |

`handle_unknown="ignore"` importa: la categoría `ISLAND` tiene **5 filas en todo el
dataset** y puede no aparecer en algún fold de la CV. Sin eso, revienta.

#### `ClusterSimilarity` — transformer propio
*Géron §2: "Custom Transformers"*

Un árbol solo puede partir el espacio con cortes rectangulares sobre latitud y
longitud, así que aproximar "cerca de San Francisco" le cuesta muchos splits y nunca
queda fino. `ClusterSimilarity` corre KMeans sobre las coordenadas y reemplaza cada
fila por su **similitud RBF a cada centroide** — o sea, N features del tipo "qué tan
cerca estás del barrio 3".

Las coordenadas crudas se conservan además de los clusters: aportan cosas distintas.

Medido con ablación (mismo split, única diferencia el transformer):

| Modelo | sin geo | con geo | Δ |
|---|---|---|---|
| random_forest | 0.4986 | 0.4630 | **−0.0356** |
| xgboost | 0.4751 | 0.4575 | −0.0176 |
| decision_tree | 0.6164 | 0.6030 | −0.0134 |
| linear_regression | 0.6984 | 0.6855 | −0.0129 |

Es la feature que más ha movido la aguja en el proyecto — bastante más que
`ocean_proximity`, que resultó casi redundante con lat/long.

### Fase 3a — Entrenamiento baseline (`src/train.py`)
*Géron §2: "Select and Train a Model"*

Cada modelo se envuelve en `Pipeline([("scaler", StandardScaler()), ("model", estimador)])`.

Cada ejecución crea un run en MLflow con:
- parámetros del modelo
- métricas: RMSE, MAE, R²
- artefacto del **Pipeline completo** serializado (descargable desde la UI)

Modelos disponibles: `linear_regression`, `random_forest`, `gradient_boosting`

Los parámetros son defaults razonables — **no son los mejores**. Son el baseline para saber desde dónde mejoramos.

### Fase 3b — Búsqueda de hiperparámetros (`src/tune.py`)
*Géron §2: "Fine-Tune Your Model"*

**Por qué los parámetros fijos no son suficientes:**
`n_estimators=100` es arbitrario. El modelo puede mejorar cambiando `max_depth`, `learning_rate`, `min_samples_leaf`, etc. Probar todas las combinaciones posibles (GridSearchCV) sería lentísimo. La solución: **RandomizedSearchCV** samplea `n_iter` combinaciones aleatorias del espacio — estadísticamente tan efectivo como buscar todo, pero mucho más rápido.

**Cómo mejora el modelo:**
```
Espacio de búsqueda (random_forest):
  n_estimators:    [50, 100, 200, 300]
  max_features:    ["sqrt", "log2", 0.5, 0.8]
  max_depth:       [None, 10, 20, 30]
  min_samples_split: [2, 5, 10]
  min_samples_leaf:  [1, 2, 4]
  → 4×4×4×3×3 = 576 combinaciones posibles
  → con n_iter=20 se prueban 20 aleatorias con CV=5
```

```bash
make tune-random_forest            # 20 combinaciones (rápido, ~2 min)
make tune-random_forest N_ITER=50  # más exhaustivo
make tune-all                      # random_forest + gradient_boosting
```

Cada combinación probada se guarda como **run anidado** dentro del run padre
`<modelo>_tuned`, con su `cv_rmse`, `cv_rmse_std` y `rank`. En la UI expande el run
padre para ver las N combinaciones y entender qué hiperparámetro movió la aguja.

Compara el `cv_rmse` (estimado real de generalización via cross-validation) vs el
RMSE del baseline.

**Nota sobre `svr`:** quedó fuera de `make tune-all`. SVR con kernel RBF escala O(n²)
y sobre 16,512 filas una búsqueda tarda >20 min sin acercarse a xgboost. Sigue
disponible con `make tune-svr`, idealmente sobre una submuestra como sugiere Géron.

#### El preprocesamiento también se tunea

La búsqueda no solo explora hiperparámetros del modelo, también del preprocesamiento:

```
prep__num__imputer__strategy:  ["median", "mean"]
prep__geo__n_clusters:         [5, 10, 15, 20, 30, 45, 60]
prep__geo__gamma:              [0.1, 0.3, 1.0, 3.0]
```

Esto **solo es posible porque el preprocesamiento vive dentro del Pipeline**. La CV
reajusta el imputador y el KMeans en cada fold, así que la búsqueda puede decidir
cómo imputar y cuántos barrios usar sin filtrar información. Es el ejercicio de
Géron de "explorar opciones de preparación con la búsqueda de hiperparámetros".

Desactívalo con `python src/tune.py --model xgboost --no-tune-prep`.

### Fase 4 — Evaluación final (`src/evaluate.py`)
*Géron §2: "Evaluate Your System on the Test Set"*

Carga el mejor run de MLflow y genera:
- `reports/evaluation.md` con métricas **y el intervalo de confianza del RMSE**
- `reports/evaluation.png` con gráficas de residuos

**Por qué el intervalo de confianza:** el RMSE del test set sale de una muestra
concreta de casas; con otras 4,128 casas habría dado distinto. El intervalo acota
cuánto puede moverse el valor real. Sirve para no perseguir decimales: **si dos
modelos tienen intervalos que se traslapan, la diferencia entre ellos cabe dentro
del ruido del muestreo** y no puedes afirmar que uno gane.

Se calcula sobre los errores cuadrados con `scipy.stats.t.interval` y se le aplica
la raíz a los dos extremos, porque el RMSE es la raíz de una media.

### Fase 5 — Inferencia (`src/predict.py`)
*Géron §2: "Launch, Monitor, and Maintain Your System"*

```bash
make predict RUN_ID=<id>                  # demo con 5 filas del test set
make predict RUN_ID=<id> INPUT=casas.csv  # sobre un CSV propio
```

Recibe datos **crudos** — sin escalar, sin imputar, sin one-hot, con NaN si los hay —
y el artefacto se encarga de todo. No replica ningún paso de preprocesamiento, que es
exactamente donde se rompen los despliegues reales (*training/serving skew*): el día
que cambias el imputador en el entrenamiento y se te olvida cambiarlo en producción.

Esa garantía es el pago del refactor del Pipeline.

---

## Resultados

Modelos tuneados, split estratificado, con `ClusterSimilarity` y `ocean_proximity`:

| Modelo | CV RMSE | Test RMSE | IC 95% | R² |
|---|---|---|---|---|
| **extra_trees** | 0.4085 | **0.4063** | [0.3844, 0.4271] | 0.8767 |
| random_forest | 0.4179 | 0.4235 | [0.4015, 0.4443] | 0.8660 |
| xgboost | 0.4230 | 0.4291 | [0.4071, 0.4500] | 0.8625 |
| gradient_boosting | 0.4475 | 0.4514 | [0.4292, 0.4726] | 0.8478 |
| mlp | 0.5007 | 0.4967 | — | 0.8157 |
| decision_tree | 0.5454 | 0.5517 | — | 0.7726 |
| ridge | 0.6051 | 0.6282 | — | 0.7052 |

**Los tres primeros son estadísticamente indistinguibles** — sus intervalos de
confianza se traslapan. Decir "extra_trees ganó" es sobreinterpretar: con otro test
set de 4,128 casas el orden entre ellos podría invertirse. El único demostrablemente
peor del grupo de arriba es `gradient_boosting`, y por poco.

### Qué movió la aguja, en orden

| Cambio | Efecto |
|---|---|
| `ClusterSimilarity` sobre lat/long | −0.036 en random_forest — **lo que más rindió** |
| Tuning de hiperparámetros | −0.02 a −0.08 según el modelo |
| Tuning del preprocesamiento | incluido arriba; eligió 20–45 clusters según el modelo |
| `ocean_proximity` | −0.01 en promedio, y **+0.004 en xgboost** |
| Split estratificado | no mejora el RMSE — hace que el número sea *confiable* |

La lección que más costó: `ocean_proximity` resultó casi redundante con `Latitude`/
`Longitude`, pese a que XGBoost la reporta como su feature más importante.
**Importancia alta ≠ mejora predictiva** cuando la información ya estaba disponible
por otra vía.

---

## Cómo iterar modelos

### La regla de oro
- **MLflow** = tracking de experimentos (parámetros, métricas, modelos)
- **Git** = cambios de código + reportes finales
- **No mezclar**: no hagas commits para guardar "versiones" de experimentos

### Flujo de iteración

```
┌─ 1. Entrena ──────────────────────────────────────────┐
│  make train-random_forest                             │
│  # → MLflow loggea el run automáticamente             │
└───────────────────────────────────────────────────────┘
         ↓
┌─ 2. Compara en la UI ─────────────────────────────────┐
│  make ui                                              │
│  # → http://localhost:5000                            │
│  # Ordena por RMSE, compara parámetros vs métricas    │
└───────────────────────────────────────────────────────┘
         ↓
┌─ 3. Prueba variaciones ───────────────────────────────┐
│  # Edita src/train.py, cambia hiperparámetros         │
│  make train-random_forest   # nuevo run en MLflow     │
│  # MLflow guarda TODOS los runs, compáralos en la UI  │
└───────────────────────────────────────────────────────┘
         ↓
┌─ 4. Evalúa el ganador ────────────────────────────────┐
│  # Copia el run_id del mejor modelo en la UI          │
│  make evaluate RUN_ID=abc123def456                    │
└───────────────────────────────────────────────────────┘
         ↓
┌─ 5. Commitea el reporte ──────────────────────────────┐
│  git add reports/                                     │
│  git commit -m "eval: random_forest RMSE=0.4821"      │
│  # Este commit sí es significativo                    │
└───────────────────────────────────────────────────────┘
```

### Variaciones a explorar (por orden de impacto)

| Qué cambiar | Dónde | Ejemplo |
|---|---|---|
| Nuevos transformers | `src/preprocessing.py` | lo que más ha rendido — ver `ClusterSimilarity` |
| Espacio de búsqueda | `src/tune.py` → `SEARCH_SPACES` / `PREP_PARAMS` | ampliar rangos que topen en el borde |
| Nuevas features | `src/features.py` | log-transform, interacciones |
| Otros modelos | `src/train.py` → `MODELS` | `LightGBM`, `CatBoost` |
| Hiperparámetros baseline | `src/train.py` → `MODELS` | `n_estimators=200` |

**Truco al leer los resultados de una búsqueda:** si el mejor valor de un parámetro
es el máximo o el mínimo del rango que le diste, el óptimo probablemente está fuera.
Pasó con `n_clusters`: con tope en 30 la búsqueda elegía justo 30, así que el rango
se extendió a 60.

---

## Cobertura del capítulo 2

| Sección del libro | Estado |
|---|---|
| Get the Data | ✅ CSV original, no la versión recortada de sklearn |
| Create a Test Set | ✅ estratificado por categoría de ingreso |
| Explore and Visualize | ✅ `notebooks/01_eda.ipynb` |
| Prepare the Data | ✅ imputación, one-hot, escalado, `ClusterSimilarity` |
| Select and Train a Model | ✅ 9 modelos |
| Fine-Tune Your Model | ✅ búsqueda sobre modelo **y** preprocesamiento |
| Evaluate on the Test Set | ✅ con intervalo de confianza |
| Launch, Monitor, Maintain | ⚠️ `predict.py` cubre inferencia; falta monitoreo |

**Ejercicios del capítulo pendientes:**
- Tunear SVR sobre una submuestra (aquí quedó fuera por costo)
- Un transformer que seleccione las features más importantes
- Reemplazar `RandomizedSearchCV` por búsqueda bayesiana

---

## Caso de estudio web (`site/`)

Una página estática que cuenta el proceso y deja explorar las predicciones sobre un
mapa de California.

```bash
make site RUN_ID=<id del modelo>   # genera site/index.html
make site-serve                    # http://localhost:8000
```

`site/template.html` es la página; `src/build_site.py` inyecta las predicciones y
produce `site/index.html` — **un solo archivo de ~170 KB, sin dependencias externas**.

**Por qué las predicciones van precalculadas y no vía API:** el artefacto de XGBoost
pesa 2.4 MB, pero sus dependencias (`numpy`, `scipy`, `scikit-learn`, `pandas`) suman
**262 MB**, por encima del límite de 250 MB de los runtimes serverless. Precalcular las
4,128 predicciones del test set cuesta 127 KB y elimina el backend por completo. Los
filtros y las métricas sí se computan en vivo, en el navegador.

### Desplegarlo

Es HTML estático sin build, así que sirve en cualquier lado:

**Desplegado en:** https://ml-handson-california-housing-26e4.vercel.app

| Destino | Cómo |
|---|---|
| Vercel | Importar el repo — `vercel.json` ya trae la configuración |
| GitHub Pages | Settings → Pages → rama `main`, carpeta `/site` |
| VPS con nginx | `root /var/www/houses;` y copiar `site/index.html` |
| Cualquier CDN | Subir `site/index.html` tal cual |

Para el subdominio `houses.eliuth.dev`: agrégalo en *Vercel → Settings → Domains* y
crea el `CNAME` que indique en el DNS de Cloudflare. Hay que hacer ambas cosas — solo
el registro DNS no basta, Vercel necesita tener el dominio registrado en el proyecto
para emitir el certificado.

**Regenerar tras reentrenar:** `make site RUN_ID=<nuevo id>` y commitear
`site/index.html`. El HTML lleva las predicciones dentro, así que si cambia el modelo
hay que regenerarlo o la página mostrará las viejas.

---

## Por qué no Docker (por ahora)

`venv` + `requirements.txt` + `Makefile` da reproducibilidad suficiente para un proyecto de aprendizaje.
Docker agrega complejidad sin beneficio real hasta que quieras:
- desplegar el modelo como API
- compartirlo con un equipo con diferentes OS

Cuando llegues a esa etapa, `Dockerfile` + `docker compose` es el siguiente paso natural.

---

## Referencia

**Libro:** Géron, A. (2023). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3rd ed.). O'Reilly.

Cada archivo `src/*.py` incluye la sección del libro que le corresponde en el docstring.

# ml-desde-cero

> ⚠️ **Pendiente:** Renombrar este repo en GitHub → Settings → General → "Repository name" → `ml-desde-cero`

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
│   └── evaluate.py       # Fase 4 — Evaluación final del mejor modelo
├── notebooks/
│   └── 01_eda.ipynb   # Exploración visual (no forma parte del pipeline)
├── data/              # gitignored — generado con `make data`
│   ├── raw/
│   └── processed/
├── reports/           # Sí va en git — generado con `make evaluate`
├── mlruns/            # gitignored — gestionado por MLflow
├── Makefile           # Pipeline completo
├── requirements.txt
└── .gitignore
```

---

## Setup (una sola vez)

```bash
git clone https://github.com/inter097/mi-musica.git ml-desde-cero
cd ml-desde-cero
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
- **Split train/test**: 80/20, semilla fija para reproducibilidad
- **Sin transformar**: los splits se guardan crudos, incluidos los NaN y la columna
  categórica. Todo el preprocesamiento vive en `src/preprocessing.py`, dentro del
  `Pipeline` de cada modelo

**Por qué el preprocesamiento no va en esta fase:** si se escalara/imputara aquí, esos
pasos verían el train set completo antes de la validación cruzada, y cada fold de
validación quedaría contaminado con estadísticas calculadas sobre él mismo — el
`cv_rmse` saldría optimista. Dentro del `Pipeline`, sklearn los reajusta en cada fold.

Beneficio extra: el artefacto guardado en MLflow contiene preprocesamiento + modelo.
Quien lo cargue le pasa datos crudos directamente, sin replicar nada.

### `src/preprocessing.py` — el `ColumnTransformer`

Un solo módulo define las transformaciones y lo reutilizan `train.py`, `tune.py` y
`analysis.py`, así no hay tres copias que se desincronicen:

| Rama | Columnas | Pasos |
|---|---|---|
| `num` | las 11 numéricas | `SimpleImputer(median)` → `StandardScaler` |
| `cat` | `ocean_proximity` | `OneHotEncoder(handle_unknown="ignore")` |

`handle_unknown="ignore"` importa: la categoría `ISLAND` tiene **5 filas en todo el
dataset** y puede no aparecer en algún fold de la CV. Sin eso, revienta.

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

### Fase 4 — Evaluación final (`src/evaluate.py`)
*Géron §2: "Evaluate Your System on the Test Set"*

Carga el mejor run de MLflow y genera:
- `reports/evaluation.md` con métricas
- `reports/evaluation.png` con gráficas de residuos

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
| Hiperparámetros | `src/train.py` → `MODELS` | `n_estimators=200` |
| Nuevas features | `src/features.py` | log-transform, interacciones |
| Otros modelos | `src/train.py` → `MODELS` | `SVR`, `XGBRegressor` |
| Estrategia de split | `src/features.py` | stratified por income quartile |

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

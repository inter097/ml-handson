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
│   ├── data.py        # Fase 1 — Descarga del dataset
│   ├── features.py    # Fase 2 — Limpieza y feature engineering
│   ├── train.py       # Fase 3 — Entrenamiento + logging MLflow
│   └── evaluate.py    # Fase 4 — Evaluación final del mejor modelo
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
make all          # descarga datos → procesa features → entrena 3 modelos
make ui           # abre http://localhost:5000 para comparar runs
```

O paso a paso:

```bash
make data         # Descarga California Housing → data/raw/housing.parquet
make features     # Limpieza + feature engineering → data/processed/
make train-all    # Entrena los 3 modelos y loggea en MLflow
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

Descarga el dataset de scikit-learn y lo persiste en Parquet.
El dataset tiene 20,640 filas y 9 columnas (8 features + target `MedHouseVal`).

### Fase 2 — Features (`src/features.py`)
*Géron §2: "Prepare the Data for ML Algorithms"*

- **Feature engineering**: ratios derivados (`rooms_per_household`, `bedrooms_ratio`, `population_per_household`)
- **Split train/test**: 80/20 estratificado, semilla fija para reproducibilidad
- **Escalado**: `StandardScaler` ajustado **solo en train** para evitar data leakage

### Fase 3 — Entrenamiento (`src/train.py`)
*Géron §2: "Select and Train a Model"*

Cada ejecución crea un run en MLflow con:
- parámetros del modelo
- métricas: RMSE, MAE, R²
- artefacto del modelo serializado (descargable desde la UI)

Modelos disponibles: `linear_regression`, `random_forest`, `gradient_boosting`

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

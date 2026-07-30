.PHONY: all setup data features train-all tune-all analysis evaluate cost cost-check predict site svr-study ui clean help

# El entorno y el registro de experimentos viven en la raíz del repo y se
# comparten entre capítulos. ROOT se resuelve desde la ruta de este Makefile,
# así que funciona igual si se invoca desde aquí o desde la raíz.
ROOT   := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)
PYTHON = $(ROOT)/venv/bin/python
PIP    = $(ROOT)/venv/bin/pip

# MLflow 3 usa una base sqlite relativa al directorio actual. Sin esto, cada
# capítulo crearía la suya y no se podrían comparar experimentos entre sí.
export MLFLOW_TRACKING_URI := sqlite:///$(ROOT)/mlflow.db

# ── Entorno ─────────────────────────────────────────────────────────────────

setup:
	python3 -m venv $(ROOT)/venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@echo "✓ Entorno listo. Activa con: source $(ROOT)/venv/bin/activate"

# ── Pipeline completo ────────────────────────────────────────────────────────

all: data features train-all
	@echo "✓ Pipeline completo."

data:
	$(PYTHON) src/data.py

features:
	$(PYTHON) src/features.py

# make train-linear_regression | make train-random_forest | make train-gradient_boosting
train-%:
	$(PYTHON) src/train.py --model $*

train-all:
	$(PYTHON) src/train.py --model linear_regression
	$(PYTHON) src/train.py --model ridge
	$(PYTHON) src/train.py --model decision_tree
	$(PYTHON) src/train.py --model random_forest
	$(PYTHON) src/train.py --model extra_trees
	$(PYTHON) src/train.py --model gradient_boosting
	$(PYTHON) src/train.py --model xgboost
	$(PYTHON) src/train.py --model svr
	$(PYTHON) src/train.py --model mlp

# ── Búsqueda de hiperparámetros (RandomizedSearchCV) ────────────────────────
# make tune-random_forest | make tune-gradient_boosting
# N_ITER=50 para buscar más combinaciones (default: 20)

tune-%:
	$(PYTHON) src/tune.py --model $* --n-iter $(or $(N_ITER),20)

tune-all:
	$(PYTHON) src/tune.py --model ridge --n-iter $(or $(N_ITER),10)
	$(PYTHON) src/tune.py --model decision_tree --n-iter $(or $(N_ITER),20)
	$(PYTHON) src/tune.py --model random_forest --n-iter $(or $(N_ITER),20)
	$(PYTHON) src/tune.py --model extra_trees --n-iter $(or $(N_ITER),20)
	$(PYTHON) src/tune.py --model gradient_boosting --n-iter $(or $(N_ITER),20)
	$(PYTHON) src/tune.py --model xgboost --n-iter $(or $(N_ITER),30)
	$(PYTHON) src/tune.py --model mlp --n-iter $(or $(N_ITER),15)
# svr queda fuera a propósito, pero no por el tamaño del dataset: medido con
# `make svr-study`, SVR escala ~n^1.3 y un ajuste completo toma segundos. Lo que
# hace lenta la búsqueda es C=100 con kernel poly o linear — hasta 121× más
# lento que rbf con C=1. Y aun tuneado se queda en 0.53 contra 0.43 de xgboost.
# Corre `make tune-svr` si lo quieres.

# ── Análisis: importancia de features + curvas de aprendizaje ────────────────

analysis:
	$(PYTHON) src/analysis.py

# ── Evaluación del mejor modelo ──────────────────────────────────────────────
# Uso: make evaluate RUN_ID=<id copiado del mlflow ui>

evaluate:
	$(PYTHON) src/evaluate.py --run-id $(RUN_ID)

# ── Costo de servir cada modelo ──────────────────────────────────────────────
# Carga cada artefacto en su propio subproceso y mide tamaño, memoria y latencia.
# `cost-check` proyecta el pico sin cargar nada y aborta si pasa el 45% de la RAM.

cost-check:
	$(PYTHON) src/cost.py --check

cost:
	$(PYTHON) src/cost.py

# ── Inferencia ───────────────────────────────────────────────────────────────
# Uso: make predict RUN_ID=<id>            → demo con 5 filas del test set
#      make predict RUN_ID=<id> INPUT=x.csv → predice sobre un CSV crudo

predict:
	$(PYTHON) src/predict.py --run-id $(RUN_ID) $(if $(INPUT),--input $(INPUT),--demo)

# ── Datos para el sitio ──────────────────────────────────────────────────────
# Uso: make site RUN_ID=<id>   → genera web/public/data/ch02.json
# La página la construye Astro: cd web && npm run dev

site:
	$(PYTHON) src/build_site.py --run-id $(RUN_ID)

# ── MLflow UI ────────────────────────────────────────────────────────────────

ui:
	$(ROOT)/venv/bin/mlflow ui

# ── Limpieza ─────────────────────────────────────────────────────────────────

clean:
	rm -rf data/raw data/processed mlruns mlartifacts

help:
	@echo ""
	@echo "  make setup          Crear venv e instalar dependencias"
	@echo "  make all            Pipeline completo (data → features → train-all)"
	@echo "  make data           Descargar dataset"
	@echo "  make features       Procesar features"
	@echo "  make train-MODEL    Entrenar un modelo (linear_regression, random_forest, gradient_boosting)"
	@echo "  make train-all      Entrenar todos los modelos (params fijos, baseline)"
	@echo "  make tune-MODEL     Buscar mejores hiperparámetros (random_forest, gradient_boosting)"
	@echo "  make tune-all       Tuning de todos los modelos"
	@echo "  make tune-MODEL N_ITER=50  Más iteraciones de búsqueda"
	@echo "  make analysis       Importancia de features + curvas de aprendizaje → reports/"
	@echo "  make ui             Abrir MLflow UI en http://localhost:5000"
	@echo "  make evaluate RUN_ID=<id>  Evaluar el mejor run y guardar reporte"
	@echo "  make cost           Medir tamaño, memoria y latencia de cada modelo"
	@echo "  make cost-check     Solo proyectar el pico de memoria, sin cargar nada"
	@echo "  make predict RUN_ID=<id>   Predecir (--demo o INPUT=archivo.csv)"
	@echo "  make site RUN_ID=<id>      Generar los datos del sitio (web/public/data/)"
	@echo "  make clean          Borrar datos y runs locales"
	@echo ""

# ── Estudio de SVR (Géron cap. 2, ejercicio 1) ───────────────────────────────
# Mide el escalado O(n^p) y tunea sobre submuestra. QUICK=1 para versión corta.

svr-study:
	$(PYTHON) src/svr_study.py $(if $(QUICK),--quick,)

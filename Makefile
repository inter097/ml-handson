.PHONY: all setup data features train-all tune-all analysis evaluate predict ui clean help

PYTHON = venv/bin/python
PIP    = venv/bin/pip

# ── Entorno ─────────────────────────────────────────────────────────────────

setup:
	python3 -m venv venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@echo "✓ Entorno listo. Activa con: source venv/bin/activate"

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
# svr queda fuera a propósito: SVR con kernel RBF escala O(n²) y sobre 16,512
# filas una búsqueda de 15 combinaciones × 5 folds tarda >20 min sin llegar a
# competir (baseline 0.54 vs 0.44 de xgboost). Corre `make tune-svr` si lo
# quieres, idealmente sobre una submuestra como sugiere Géron.

# ── Análisis: importancia de features + curvas de aprendizaje ────────────────

analysis:
	$(PYTHON) src/analysis.py

# ── Evaluación del mejor modelo ──────────────────────────────────────────────
# Uso: make evaluate RUN_ID=<id copiado del mlflow ui>

evaluate:
	$(PYTHON) src/evaluate.py --run-id $(RUN_ID)

# ── Inferencia ───────────────────────────────────────────────────────────────
# Uso: make predict RUN_ID=<id>            → demo con 5 filas del test set
#      make predict RUN_ID=<id> INPUT=x.csv → predice sobre un CSV crudo

predict:
	$(PYTHON) src/predict.py --run-id $(RUN_ID) $(if $(INPUT),--input $(INPUT),--demo)

# ── MLflow UI ────────────────────────────────────────────────────────────────

ui:
	venv/bin/mlflow ui

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
	@echo "  make predict RUN_ID=<id>   Predecir (--demo o INPUT=archivo.csv)"
	@echo "  make clean          Borrar datos y runs locales"
	@echo ""

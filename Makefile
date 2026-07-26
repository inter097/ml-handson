.PHONY: all setup data features train-all ui clean help

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
	$(PYTHON) src/train.py --model random_forest
	$(PYTHON) src/train.py --model gradient_boosting

# ── Evaluación del mejor modelo ──────────────────────────────────────────────
# Uso: make evaluate RUN_ID=<id copiado del mlflow ui>

evaluate:
	$(PYTHON) src/evaluate.py --run-id $(RUN_ID)

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
	@echo "  make train-all      Entrenar todos los modelos"
	@echo "  make ui             Abrir MLflow UI en http://localhost:5000"
	@echo "  make evaluate RUN_ID=<id>  Evaluar el mejor run y guardar reporte"
	@echo "  make clean          Borrar datos y runs locales"
	@echo ""

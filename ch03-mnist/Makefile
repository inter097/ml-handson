.PHONY: all setup data features baseline metrics ui clean help

# El entorno y el registro de experimentos viven en la raíz del repo y se
# comparten entre capítulos. ROOT se resuelve desde la ruta de este Makefile.
ROOT   := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)
PYTHON = $(ROOT)/venv/bin/python

# MLflow 3 usa una base sqlite relativa al directorio actual. Sin esto, cada
# capítulo crearía la suya y no se podrían comparar experimentos entre sí.
export MLFLOW_TRACKING_URI := sqlite:///$(ROOT)/mlflow.db

# ── Pipeline ────────────────────────────────────────────────────────────────

all: data features baseline
	@echo "✓ Pipeline completo."

data:
	$(PYTHON) src/data.py

features:
	$(PYTHON) src/features.py

# ── El detector binario y la trampa de la exactitud ─────────────────────────
# Uso: make baseline           → detecta el dígito 5
#      make baseline DIGIT=8   → cualquier otro

baseline:
	$(PYTHON) src/baseline.py --digit $(or $(DIGIT),5)

# ── Umbral, precisión/exhaustividad y ROC ───────────────────────────────────
# Uso: make metrics                → objetivo 90% de precisión
#      make metrics PRECISION=0.99 → más exigente

metrics:
	$(PYTHON) src/metrics.py --digit $(or $(DIGIT),5) --target-precision $(or $(PRECISION),0.90)

# ── Multiclase y análisis de errores ────────────────────────────────────────

multiclass:
	$(PYTHON) src/multiclass.py $(if $(MODELS),--models $(MODELS),)

errors:
	$(PYTHON) src/errors.py

multioutput:
	$(PYTHON) src/multioutput.py

# ── Ejercicios del capítulo ─────────────────────────────────────────────────

titanic:
	$(PYTHON) src/titanic.py

spam:
	$(PYTHON) src/spam.py

knn:
	$(PYTHON) src/knn.py

augment:
	$(PYTHON) src/augment.py

# ── MLflow UI ───────────────────────────────────────────────────────────────

ui:
	$(ROOT)/venv/bin/mlflow ui

# ── Limpieza ────────────────────────────────────────────────────────────────

clean:
	rm -rf data/raw data/processed

help:
	@echo ""
	@echo "  make data            Descargar MNIST desde OpenML"
	@echo "  make features        Partición canónica 60k/10k"
	@echo "  make baseline        Detector binario + la trampa de la exactitud"
	@echo "  make baseline DIGIT=8  Detectar otro dígito"
	@echo "  make metrics         Umbral, curvas PR y ROC → reports/"
	@echo "  make metrics PRECISION=0.99  Objetivo de precisión distinto"
	@echo "  make multiclass      Los 10 dígitos: OvR vs OvO + efecto del escalado"
	@echo "  make errors          Matriz de confusión 10x10 y qué confunde con qué"
	@echo "  make multioutput     Multietiqueta y quitar ruido de imágenes"
	@echo "  make titanic         Ejercicio 3 — clasificador de supervivencia"
	@echo "  make spam            Ejercicio 4 — filtro de spam desde cero"
	@echo "  make knn             Ejercicio 1 — k-vecinos afinado (>97%)"
	@echo "  make augment         Ejercicio 2 — aumentar datos desplazando imágenes"
	@echo "  make ui              Abrir MLflow UI"
	@echo "  make clean           Borrar datos locales"
	@echo ""

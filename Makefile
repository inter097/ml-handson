.PHONY: help setup ui clean-mlflow list

# Delegador: cada capítulo tiene su propio Makefile con sus fases.
# Desde aquí se puede invocar cualquiera sin cambiar de directorio:
#
#     make ch02 T=train-all
#     make ch02 T="tune-xgboost N_ITER=30"
#
# O entrar al capítulo y usar su Makefile directamente, que es lo habitual:
#
#     cd ch02-california-housing && make train-all

ROOT     := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CHAPTERS := $(sort $(patsubst %/Makefile,%,$(wildcard ch*/Makefile)))

# El registro de experimentos es uno solo para todo el libro: un experimento
# por capítulo, comparables entre sí desde la misma interfaz.
export MLFLOW_TRACKING_URI := sqlite:///$(ROOT)/mlflow.db

ch%:
	@$(MAKE) -C $(firstword $(wildcard ch$**)) $(T)

setup:
	python3 -m venv $(ROOT)/venv
	$(ROOT)/venv/bin/pip install -q --upgrade pip
	@for c in $(CHAPTERS); do \
		[ -f $$c/requirements.txt ] && echo "  instalando $$c/requirements.txt" && \
		$(ROOT)/venv/bin/pip install -q -r $$c/requirements.txt; \
	done
	@echo "✓ Entorno listo. Activa con: source venv/bin/activate"

ui:
	$(ROOT)/venv/bin/mlflow ui

list:
	@echo ""
	@echo "  Capítulos en el repo:"
	@for c in $(CHAPTERS); do echo "    $$c"; done
	@echo ""

# Borra solo los artefactos de MLflow, no los datos de cada capítulo.
# mlruns/ llega a pesar varios GB tras muchas búsquedas.
clean-mlflow:
	rm -rf $(ROOT)/mlruns $(ROOT)/mlartifacts $(ROOT)/mlflow.db

help:
	@echo ""
	@echo "  make setup              Crear venv e instalar dependencias de todos los capítulos"
	@echo "  make list               Listar los capítulos del repo"
	@echo "  make ui                 Abrir MLflow UI (todos los capítulos juntos)"
	@echo "  make ch02 T=train-all   Ejecutar un objetivo de un capítulo"
	@echo "  make clean-mlflow       Borrar runs y artefactos de MLflow"
	@echo ""
	@echo "  Lo habitual es entrar al capítulo:"
	@echo "    cd ch02-california-housing && make help"
	@echo ""

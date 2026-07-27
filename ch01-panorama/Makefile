.PHONY: lifesat help

# El entorno vive en la raíz del repo y se comparte entre capítulos.
ROOT   := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)
PYTHON = $(ROOT)/venv/bin/python

# Este capítulo no registra experimentos en MLflow: son 27 países y dos
# modelos de una línea. Medir su "desempeño" con validación cruzada sería
# ceremonia sin sustancia.

lifesat:
	$(PYTHON) src/lifesat.py

help:
	@echo ""
	@echo "  make lifesat   Los dos estilos de generalización sobre el mismo dato"
	@echo ""

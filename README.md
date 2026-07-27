# ml-handson

Trabajando el libro *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*
— Aurélien Géron, 3ª edición — un capítulo por carpeta.

Cada capítulo es un proyecto completo y autónomo: su propio `Makefile`, su código, sus
reportes. Lo que se comparte es el entorno de Python y el registro de experimentos, para
poder comparar resultados entre capítulos desde una sola interfaz.

---

## Capítulos

| # | Carpeta | Tema | Estado |
|---|---|---|---|
| 2 | [`ch02-california-housing`](ch02-california-housing/) | Proyecto de punta a punta — regresión | ✅ Completo · [caso de estudio](https://houses.eliuth.dev) |
| 3 | [`ch03-mnist`](ch03-mnist/) | Clasificación — métricas que no engañan | ✅ Completo |

Los capítulos 1 y 4 en adelante se irán agregando con la misma estructura.

---

## Arranque

```bash
git clone https://github.com/inter097/ml-handson.git
cd ml-handson
make setup                      # venv en la raíz + dependencias de cada capítulo
source venv/bin/activate
```

Después, cada capítulo se trabaja desde su carpeta:

```bash
cd ch02-california-housing
make help
make all
```

O sin cambiar de directorio:

```bash
make ch02 T=train-all
make ch02 T="tune-xgboost N_ITER=30"
```

---

## Qué se comparte y qué no

| | Dónde vive | Por qué |
|---|---|---|
| Entorno Python | `venv/` en la raíz | Un solo lugar que activar; los capítulos comparten casi todas las librerías |
| Experimentos | `mlflow.db` en la raíz | Un experimento por capítulo, comparables desde `make ui` |
| Datos | `<capítulo>/data/` | Cada dataset con su capítulo. Fuera de git, se regeneran con `make data` |
| Código y reportes | `<capítulo>/` | Un capítulo no importa nada de otro |

**Sobre MLflow:** la versión 3 usa una base SQLite relativa al directorio de trabajo.
Sin fijarla, cada capítulo crearía la suya y no se podrían comparar experimentos entre
sí. Los `Makefile` exportan `MLFLOW_TRACKING_URI` apuntando a la raíz.

**Sobre las dependencias:** de momento un solo `venv` sirve. A partir del capítulo 10
entra TensorFlow, que pesa mucho y suele pelear con las versiones de otras librerías —
ahí probablemente haya que separar entornos por parte del libro.

---

## Estructura

```
.
├── ch02-california-housing/    # Capítulo 2 — completo
│   ├── src/                    #   pipeline por responsabilidad
│   ├── notebooks/              #   exploración
│   ├── reports/                #   gráficas y evaluación
│   ├── site/                   #   caso de estudio publicado
│   ├── data/                   #   gitignored
│   ├── Makefile
│   └── requirements.txt
├── venv/                       # gitignored — compartido
├── mlflow.db                   # gitignored — compartido
├── mlruns/                     # gitignored — artefactos de MLflow
├── vercel.json                 # despliegue del sitio del capítulo 2
└── Makefile                    # delegador
```

---

**Referencia:** Géron, A. (2023). *Hands-On Machine Learning with Scikit-Learn, Keras &
TensorFlow* (3ª ed.). O'Reilly.

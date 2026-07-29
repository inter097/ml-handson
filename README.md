# ml-handson

Trabajando el libro *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*
de Aurélien Géron, 3ª edición, un capítulo por carpeta.

Cada capítulo es un proyecto completo y autónomo: su propio `Makefile`, su código, sus
reportes. Lo que se comparte es el entorno de Python y el registro de experimentos, para
poder comparar resultados entre capítulos desde una sola interfaz.

---

## Capítulos

| # | Carpeta | Tema | Estado |
|---|---|---|---|
| 1 | [`ch01-panorama`](ch01-panorama/) | El panorama del ML | ✅ [Página](https://ml.eliuth.dev/ch01) |
| 2 | [`ch02-california-housing`](ch02-california-housing/) | Regresión de punta a punta | ✅ [Caso de estudio](https://ml.eliuth.dev/ch02) |
| 3 | [`ch03-mnist`](ch03-mnist/) | Clasificación y las métricas que engañan | ✅ [Página](https://ml.eliuth.dev/ch03) |

**Sitio:** https://ml.eliuth.dev

Del 4 en adelante se irán agregando con la misma estructura.

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
| Entorno Python | `venv/` en la raíz | Un solo lugar que activar |
| Experimentos | `mlflow.db` en la raíz | Un experimento por capítulo, comparables desde `make ui` |
| Datos | `<capítulo>/data/` | Fuera de git, se regeneran con `make data` |
| Código y reportes | `<capítulo>/` | Un capítulo no importa nada de otro |
| Sitio | `web/` | Astro, una carpeta por capítulo |

El detalle de por qué está así, junto con las convenciones de código y de escritura,
en [`CLAUDE.md`](CLAUDE.md).

---

## Estructura

```
.
├── ch01-panorama/              # Capítulo 1
├── ch02-california-housing/    # Capítulo 2
│   ├── src/                    #   pipeline por responsabilidad
│   ├── notebooks/              #   exploración
│   ├── reports/                #   gráficas y evaluación
│   ├── data/                   #   gitignored
│   ├── Makefile
│   └── requirements.txt
├── ch03-mnist/                 # Capítulo 3
├── web/                        # El sitio, en Astro
│   ├── src/pages/              #   una carpeta por capítulo
│   ├── src/data/capitulos.ts   #   las partes de cada capítulo
│   └── public/data/            #   JSON que genera Python
├── venv/                       # gitignored, compartido
├── mlflow.db                   # gitignored, compartido
├── vercel.json                 # despliegue
├── package.json                # delegador, para que Vercel detecte Node
└── Makefile                    # delegador
```

---

**Referencia:** Géron, A. (2023). *Hands-On Machine Learning with Scikit-Learn, Keras &
TensorFlow* (3ª ed.). O'Reilly.

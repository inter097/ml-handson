# Guía para agentes de IA

Repo de un capítulo por carpeta del libro *Hands-On Machine Learning with
Scikit-Learn, Keras & TensorFlow* — Aurélien Géron, 3ª edición.

Consulta la documentación oficial antes de escribir código:
- scikit-learn: https://scikit-learn.org/stable/
- MLflow: https://mlflow.org/docs/latest/
- pandas: https://pandas.pydata.org/docs/

---

## Estructura

Cada capítulo vive en `chNN-<tema>/` y es autónomo: su `Makefile`, su `src/`, sus
reportes. **Un capítulo nunca importa código de otro** — si algo se repite, se copia.
Duplicar es más barato que acoplar capítulos que enseñan cosas distintas.

Lo que sí se comparte, y vive en la raíz:

| Recurso | Ruta | Nota |
|---|---|---|
| Entorno Python | `venv/` | Uno solo. A partir del cap. 10 (TensorFlow) quizá haya que separar |
| Experimentos | `mlflow.db` | Un experimento de MLflow por capítulo |

**MLflow 3 usa una base SQLite relativa al directorio de trabajo.** Sin fijarla, cada
capítulo crearía la suya y no se podrían comparar experimentos. Los `Makefile` exportan
`MLFLOW_TRACKING_URI` apuntando a la raíz — no quitar esa línea.

---

## Convenciones que sigue el repo

- El preprocesamiento va **dentro** del `Pipeline` de sklearn, nunca antes del split.
  Si se aplica antes, la validación cruzada se contamina y el error sale optimista.
- Todo lo que se afirme en un README debe estar medido. Este repo ya documentó una
  explicación inventada sobre el escalado de SVR; se corrigió midiéndola.
- Los datasets se regeneran con `make data`, nunca se commitean.
- Los reportes en `reports/` sí se commitean: sirven para comparar entre versiones.

---

## Despliegue del caso de estudio del capítulo 2

El sitio vive en **https://houses.eliuth.dev** y se redespliega solo en cada push a
`main`. La configuración está en `vercel.json` en la raíz, con `outputDirectory`
apuntando a `ch02-california-housing/site`.

Cómo quedó armado, por si hay que replicarlo en otro capítulo:

1. **Vercel** — el dominio se registra en el proyecto:
   `vercel domains add <sub>.eliuth.dev <proyecto>`
2. **Cloudflare** — ahí vive el DNS de `eliuth.dev`. Vercel pide un registro **A** a
   `76.76.21.21`, en modo **DNS only** (`proxied: false`). Con el proxy activo Vercel
   no valida el dominio y no emite el certificado.

Hacen falta **los dos pasos**: solo el registro DNS no basta.

**Credenciales** — nunca pegarlas en un chat; quedan en la transcripción:
- Vercel: `vercel login` (OAuth por navegador, la sesión queda en la máquina).
- Cloudflare: token con permiso *Zone · DNS · Edit*, en `~/.config/cf/token` con
  permisos `600`. Basta la API REST con `curl`. Leerlo siempre por sustitución de
  comandos (`$(cat ...)`), nunca imprimirlo.

Ojo: los tokens de cuenta (prefijo `cfat_`) **no** se validan en
`/user/tokens/verify` sino en `/accounts/<account_id>/tokens/verify`.

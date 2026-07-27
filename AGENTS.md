# Guía para agentes de IA

Consulta la documentación oficial antes de escribir código:
- scikit-learn: https://scikit-learn.org/stable/
- MLflow: https://mlflow.org/docs/latest/
- pandas: https://pandas.pydata.org/docs/

**Libro de referencia principal:** *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — Aurélien Géron (3ª edición).

---

## Despliegue del caso de estudio

El sitio vive en **https://houses.eliuth.dev** y se redespliega solo en cada push a
`main`. La configuración de Vercel está en `vercel.json`.

Cómo quedó armado, por si hay que rehacerlo o replicarlo en otro subdominio:

1. **Vercel** — el dominio se registra en el proyecto:
   `vercel domains add <sub>.eliuth.dev ml-handson-california-housing`
2. **Cloudflare** — ahí vive el DNS de `eliuth.dev` (nameservers `*.ns.cloudflare.com`).
   Vercel pide un registro **A** a `76.76.21.21`, creado en modo **DNS only**
   (`proxied: false`). Con el proxy activo Vercel no valida el dominio y no emite
   el certificado.

Hacen falta **los dos pasos**: solo el registro DNS no basta, porque Vercel no emite
certificado para un dominio que no tiene registrado en el proyecto.

**Credenciales** — nunca pegarlas en un chat; quedan en la transcripción:
- Vercel: `vercel login` (OAuth por navegador, la sesión queda en la máquina).
- Cloudflare: token con permiso *Zone · DNS · Edit*, en `~/.config/cf/token` con
  permisos `600`. No hace falta CLI, basta la API REST con `curl`. Leerlo siempre por
  sustitución de comandos (`$(cat ...)`), nunca imprimirlo.

Ojo: los tokens de cuenta (prefijo `cfat_`) **no** se validan en
`/user/tokens/verify` sino en `/accounts/<account_id>/tokens/verify`.

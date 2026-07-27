# Guía para agentes de IA

Consulta la documentación oficial antes de escribir código:
- scikit-learn: https://scikit-learn.org/stable/
- MLflow: https://mlflow.org/docs/latest/
- pandas: https://pandas.pydata.org/docs/

**Libro de referencia principal:** *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — Aurélien Géron (3ª edición).

---

## Pendientes

### Dominio propio para el caso de estudio

El sitio vive en `https://ml-handson-california-housing-26e4.vercel.app`. La URL
autogenerada se ve mal para portafolio; falta apuntarlo a **`houses.eliuth.dev`**.

Son dos pasos en dos sistemas distintos, y hacen falta **los dos** — solo el registro
DNS no basta, porque Vercel no emite el certificado si el dominio no está registrado
en el proyecto:

1. **Vercel** → *Settings → Domains → Add* → `houses.eliuth.dev`. Ahí también conviene
   renombrar el proyecto para perder el sufijo `-26e4`.
2. **Cloudflare** (ahí vive el DNS de `eliuth.dev`) → crear el `CNAME` que indique
   Vercel, en modo **DNS only** (nube gris). Con el proxy activo Vercel no puede
   validar el dominio y la emisión del certificado se queda a medias.

**Credenciales necesarias** (ninguna debe pegarse en un chat):
- Vercel: `vercel login` — flujo OAuth por navegador, la sesión queda en la máquina.
  El CLI ya está instalado.
- Cloudflare: token con alcance *Edit zone DNS* limitado a `eliuth.dev`, guardado en
  `~/.config/cf/token` con permisos `600`. No hace falta CLI; basta la API REST con
  `curl`. Leerlo siempre por sustitución de comandos, nunca imprimirlo.

Mientras tanto la URL de Vercel funciona y el despliegue es automático en cada push.

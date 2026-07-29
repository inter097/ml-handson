# Convenciones del proyecto

Cuaderno de trabajo del libro *Hands-On Machine Learning with Scikit-Learn, Keras &
TensorFlow* (Géron, 3ª ed.), un capítulo por carpeta, publicado en
**https://ml.eliuth.dev**.

`AGENTS.md` es un enlace a este mismo archivo: las reglas viven en un solo sitio para
que no se desincronicen.

Consultar la documentación oficial antes de escribir código:
[scikit-learn](https://scikit-learn.org/stable/) ·
[MLflow](https://mlflow.org/docs/latest/) ·
[pandas](https://pandas.pydata.org/docs/) ·
[Astro](https://docs.astro.build/)

---

## La regla de fondo

**Si algo se afirma, se mide.** Ya se corrigieron varias afirmaciones que parecían
obvias y no resistieron la medición: que cierta variable mejoraría el modelo, que SVR
escalaba con el cuadrado de los datos, que una etiqueta sería más difícil que otra.
Ninguna era cierta.

Antes de escribir una cifra o una causa en una página, ejecutar la medición. Si no se
puede medir, no se escribe.

---

## Estructura del repositorio

Cada capítulo vive en `chNN-<tema>/` y es autónomo: su `Makefile`, su `src/`, sus
reportes. **Un capítulo nunca importa código de otro.** Si algo se repite, se copia.
Duplicar es más barato que acoplar capítulos que enseñan cosas distintas.

Lo que sí se comparte vive en la raíz:

| Recurso | Ruta | Nota |
|---|---|---|
| Entorno Python | `venv/` | Uno solo. A partir del cap. 10 (TensorFlow) quizá haya que separar |
| Experimentos | `mlflow.db` | Un experimento de MLflow por capítulo |
| Sitio | `web/` | Astro, una carpeta por capítulo dentro de `src/pages/` |

**MLflow 3 usa una base SQLite relativa al directorio de trabajo.** Sin fijarla, cada
capítulo crearía la suya y no se podrían comparar experimentos. Los `Makefile` exportan
`MLFLOW_TRACKING_URI` apuntando a la raíz. No quitar esa línea.

---

## Convenciones de código

- El preprocesamiento va **dentro** del `Pipeline` de sklearn, nunca antes de la
  partición. Aplicado antes, la validación cruzada se contamina y el error sale
  optimista.
- Los datasets se regeneran con `make data` y no se commitean. Los reportes de
  `reports/` sí: sirven para comparar entre versiones.
- Cuidado con el paralelismo anidado. Un `GridSearchCV(n_jobs=-1)` sobre un estimador
  que ya lleva `n_jobs=-1` copia los datos por cada proceso y agota la memoria. Ya pasó
  una vez y hubo que reiniciar la máquina.

---

## Cómo se escribe

### Tercera persona, siempre

Nunca segunda persona. Ni imperativos dirigidos al lector, ni posesivos.

| No | Sí |
|---|---|
| Mueve el control y verás… | El control mueve el PIB |
| Todo esto corre en tu navegador | Esta demo se ejecuta en el navegador |
| Si entrenas con fotos de internet | Si el entrenamiento usa fotos de internet |
| Miras los correos y escribes reglas | Se examinan los correos y se escribe una regla |
| Mediste cien veces y elegiste | Fueron cien mediciones, y se eligió |

Tampoco primera persona del plural: «le impusimos una forma» → «se le impuso una forma».

### Nada de rayas

**No usar `—` en prosa.** Se lee como texto generado por una máquina. Cada uso pide una
solución distinta según lo que esté haciendo la raya:

| Lo que hacía | Cómo se escribe |
|---|---|
| Explicar lo anterior | Dos puntos: «el modelo se pudre, porque el mundo cambia» |
| Meter un inciso | Paréntesis: «(ingreso medio, antigüedad, coordenadas)» |
| Encadenar una idea nueva | Punto y frase aparte |
| Enumerar | Coma |
| Separar en un título | Punto medio: «MNIST · clasificación» |

Único uso permitido: **marcador de valor ausente** en la interfaz, cuando un contador
todavía no tiene dato.

### Longitud

Una idea se dice **una sola vez**, en el lugar donde se puede ver ocurrir.

- Si un dato ya está visible en la interfaz, no se repite en prosa. La interfaz es la
  explicación.
- Ningún párrafo anuncia lo que la página va a mostrar. Ya se está viendo.
- Nada de definiciones de manual. «Regresión», no «Regresión. La respuesta es una
  cantidad, no una categoría».
- Si una frase cabe en cinco palabras, va en cinco.

### Formato

- Texto **alineado a la izquierda**, nunca justificado.
- Los errores se documentan igual que los aciertos. Un cuaderno donde todo sale bien a
  la primera no enseña nada.

---

## Cómo se organiza

### Una parte, un archivo

Cada capítulo se parte en piezas independientes, no en secciones de un archivo largo:

```
/chNN                    Resumen conceptual del capítulo. Solo teoría.
/chNN/<dataset>          El desarrollo de ese dataset o ejercicio.
/chNN/<dataset>/demo     Su demo. Lo más simplificada posible.
```

Correspondencia en el repositorio:

```
web/src/pages/chNN/index.astro
web/src/pages/chNN/<dataset>/index.astro
web/src/pages/chNN/<dataset>/demo.astro
```

El capítulo lista sus prácticas como tarjetas. Nunca explica el dataset dentro de su
propia página.

### La barra de navegación

`web/src/data/capitulos.ts` define las partes de cada capítulo en orden de lectura.
`Base.astro` la pinta arriba de todas las páginas del capítulo y marca la actual
comparando con la ruta.

Un capítulo nuevo **solo añade su entrada ahí**. No se toca ninguna página.

Con una sola parte no se pinta: una barra de un botón no es navegación.

---

## Gráficas

### Tamaños

- `viewBox` de **680 × 330** para las gráficas de demo.
- Van dentro de `.wrap` (la columna de texto), **no** de `.wide`. Una gráfica más ancha
  que la prosa rompe la alineación de la página.
- Secciones consecutivas que son una sola unidad llevan `.compacta`. Por defecto cada
  `<section>` tiene hasta 4.5 rem arriba y abajo, y entre dos seguidas se suman.

### Qué tecnología para qué

| Caso | Herramienta | Por qué |
|---|---|---|
| Muchos puntos (miles) | Canvas 2D | 4,128 nodos del DOM arrastran el navegador |
| Pocos puntos (decenas) | SVG generado en JS | Más simple, sin escalado por densidad de píxeles |
| Gráficas estáticas | matplotlib, exportado a PNG | Se generan una vez y se commitean |

**Ninguna librería de gráficas.** Ni D3, ni Chart.js. Pesan más que el código propio.

### Etiquetas

Los nombres se dibujan **al final**, encima de todo lo demás, con contorno del color del
fondo. Cuando dos caen cerca en el eje X se alternan arriba y abajo, y junto al borde se
anclan hacia dentro.

---

## Restricciones técnicas del sitio

**Astro y nada más.** Cero dependencias, cero integraciones. Sin framework de UI, sin
framework de CSS. Solo `.astro`, un CSS a mano con variables, y TypeScript sin librerías.

Cosas que costaron encontrar y no hay que volver a tropezar:

- **`compressHTML: false`** en `astro.config.mjs`. Activado (que es el defecto) colapsa
  el salto de línea antes de una etiqueta en línea y pega las palabras: «de ahí la
  \<strong\>validación cruzada» salía como «lavalidación». Lo que ahorra es ruido frente
  a gzip.
- **Los estilos con ámbito de Astro no alcanzan al DOM que inyecta el script.** Los
  elementos creados desde JS nacen sin el atributo `data-astro-cid-*` y las reglas los
  ignoran. Van con `:global()`.
- **`pre` necesita `overflow-x: auto`.** Sin eso una línea larga de código arrastra la
  página entera en horizontal en móvil.
- Los datos los genera Python en `web/public/data/chNN.json`. `.gitignore` bloquea
  `data/`, así que hay una excepción explícita para `web/public/data/`.

---

## Despliegue

El sitio se redespliega solo en cada push a `main`. `vercel.json` está en la raíz, con
`outputDirectory` apuntando a `web/dist`.

El `package.json` de la raíz es un delegador que solo existe para que Vercel detecte
Node en vez de Python. Sin él el despliegue falla con «No python entrypoint found», y
`vercel.json` necesita `"framework": null` por el mismo motivo.

Cómo quedó armado el dominio, por si hay que replicarlo:

1. **Vercel**: `vercel domains add <sub>.eliuth.dev <proyecto>`
2. **Cloudflare**, que es donde vive el DNS de `eliuth.dev`. Vercel pide un registro
   **A** a `76.76.21.21` en modo **DNS only** (`proxied: false`). Con el proxy activo
   Vercel no valida el dominio y no emite el certificado.

Hacen falta los dos pasos: solo el registro DNS no basta.

**Credenciales.** Nunca pegarlas en un chat, quedan en la transcripción.

- Vercel: `vercel login`, OAuth por navegador, la sesión queda en la máquina.
- Cloudflare: token con permiso *Zone · DNS · Edit*, en `~/.config/cf/token` con
  permisos `600`. Basta la API REST con `curl`. Leerlo siempre por sustitución de
  comandos (`$(cat ...)`), nunca imprimirlo.

Los tokens de cuenta (prefijo `cfat_`) **no** se validan en `/user/tokens/verify` sino
en `/accounts/<account_id>/tokens/verify`.

---

## Antes de dar algo por terminado

1. `npm run build` en `web/` sin errores.
2. Cada página nueva o tocada, verificada en navegador: **HTTP 200, sin errores de
   consola, sin desbordamiento horizontal a 390 y a 1280 px**.
3. Si hay demo, comprobar que sus números **coinciden con los que imprime Python**.
4. `grep` de rayas (`—`) y de segunda persona en las páginas tocadas.

Servidor local para ir viendo cambios: `cd web && npm run dev` (queda en segundo plano;
`npx astro dev stop` para bajarlo). Recarga sola al guardar. El JSON de datos no: ese
hay que regenerarlo con el `make` del capítulo.

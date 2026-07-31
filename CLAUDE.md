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

## Cómo se avanza por el libro

Un capítulo por carpeta, en orden. **Los ejercicios de fin de capítulo se hacen dentro
de su capítulo**, no se acumulan para el final: un capítulo no está terminado hasta que
sus ejercicios están resueltos y publicados.

Así es como ya venía saliendo. El capítulo 3 resolvió sus cuatro, y el 2 resolvió sus
seis sin proponérselo, porque el proyecto los necesitaba (`svr_study.py`,
`RandomizedSearchCV`, `SelectFromModel`, el transformador de k-vecinos, la búsqueda
sobre el preprocesamiento y el clon de `StandardScaler`).

**Cuántos ejercicios tiene un capítulo se comprueba, no se recuerda.** La fuente es el
notebook oficial de la tercera edición, que trae la sección «Exercise solutions» al final:

```
https://raw.githubusercontent.com/ageron/handson-ml3/main/0N_<nombre>.ipynb
```

Aquí ya se afirmó que el capítulo 2 tenía ocho y que le faltaban dos. Tiene seis, y el
notebook cierra el sexto con «All good! That's all for today!». Los dos inventados
llegaron a `CLAUDE.md` y a una página publicada.

Un ejercicio que resuelve algo que el desarrollo ya usa no necesita página propia: vive
donde se usa, y su detalle se despliega con `<details>` donde se lista. Solo el ejercicio
con dataset propio o resultado medible propio se documenta por separado.

### Los tres bloques de un ejercicio

El libro es la guía y dice qué pide cada ejercicio. El cuaderno es lo propio y mide sobre
su pipeline. Confundir los dos fue lo que dejó ejercicios marcados como resueltos por algo
parecido pero distinto de lo que el enunciado pedía. Cada ejercicio lleva los tres bloques,
en este orden:

1. **Qué pide**, literal del notebook oficial.
2. **La versión del libro**, ejecutada. Su código y su cifra.
3. **Qué pasó aquí**, sobre el pipeline del capítulo, con la ruta del archivo.

Cuando los dos coinciden, el tercero es una línea. Cuando no, ahí está lo que vale la pena
publicar. Ejecutar la rejilla literal del ejercicio 1 del capítulo 2 dejó una medición que
el atajo no daba: el kernel lineal se estanca en 0.6452 desde `C = 10`, y el tramo caro del
enunciado cuesta 48 veces más para devolver la misma cifra.

**Tope de costo: dos horas de reloj proyectadas.** Por encima, se recorta y se publica el
número por el que se recortó. Un recorte medido y declarado es parte del resultado; uno
silencioso es lo que hubo que corregir.

Estado al 2026-07-29: capítulos 1, 2 y 3 terminados y publicados, con todos sus
ejercicios. Sigue el 4.

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

**La máquina de desarrollo tiene 16 GB y no hay margen.** Un trabajo que en otro equipo
solo iría lento, aquí tumba el sistema. Antes de lanzar un entrenamiento pesado hay que
**proyectar el pico de memoria y decirlo**, no medirlo con una versión reducida y
ejecutar después otra distinta. Los scripts que pueden crecer llevan `--check`, que
proyecta sin reservar y aborta si el pico estimado supera el 45% de la RAM.

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

### Los seis tics que se cuelan igual

Cambiar los verbos a tercera persona no basta. El texto sigue sonando a alguien hablando
por seis vías más, y son las que hay que revisar antes de dar una página por buena:

| Tic | No | Sí |
|---|---|---|
| Demostrativo que señala con el dedo | **Eso** no entra en ningún plan gratuito | Queda por encima de lo que admite un plan gratuito |
| Adverbio de queja o de énfasis | y **encima** tarda más · **justo** el reflejo | y tarda más · el reflejo |
| Objeto que actúa como persona | La tabla **enseña** algo · el capítulo **intenta** corregir | En la tabla **aparece** · el capítulo **desaconseja** |
| Verbo con un sujeto oculto que es el autor | Seis salieron **sin buscarlos** | Seis **ya estaban resueltos** |
| Sentencia de veredicto | No es una concesión. Es la decisión correcta, y es la que se toma en producción todos los días | Cuando la precisión no distingue, el desempate lo pone el costo |
| Fragmento que continúa una charla | **Con** una consecuencia incómoda: | De ahí se sigue una consecuencia incómoda. |

El veredicto es el peor de los seis: tres frases para dictar una moraleja que el dato ya
demostró. Una basta, y va en indicativo.

Barrido antes de cerrar, sobre las páginas tocadas:

```
grep -nE "\b(Eso|Ahí|encima|obviamente|por supuesto|o sea|digamos)\b" <archivos>
grep -nE "\b(nuestro|nuestra|nosotros|hicimos|vimos|medimos|tenemos|podemos|tu |tus |verás)\b" <archivos>
```

Da falsos positivos («por encima de» es preposición, «Quien mira» es tercera persona).
Se revisan a mano, no se corrigen a ciegas.

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

### Qué es una demo

**Una demo se usa: el visitante mueve las variables y el modelo responde.** Entra un
caso, sale una predicción. Eso, y nada más, es lo que lleva la ruta `/demo`.

Una gráfica que muestra el error, un mapa coloreado o una correlación **no son demos**.
Son figuras, y su sitio es la página del desarrollo, junto al texto que explican.

| Es demo | No es demo |
|---|---|
| Controles que fijan las variables y devuelven una predicción | Un mapa de los errores del modelo |
| Comparar dos modelos sobre el mismo caso introducido | Una curva de aprendizaje |
| Mover un umbral y ver cambiar la decisión | Una dispersión de predicho contra real |

La diferencia práctica: una demo necesita el modelo del lado del navegador, así que hay
que exportar lo aprendido a JSON (coeficientes, centros de los clusters, medias y escalas
del escalador, los árboles si el modelo es de árboles) y reimplementar la predicción en
TypeScript. Una figura solo necesita datos ya calculados.

Estado al 2026-07-30: la página `/ch02/california-housing/demo` es un mapa de errores, o
sea una figura. Queda por hacer la demo de verdad.

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
5. Releer buscando **los seis tics**, que el `grep` no atrapa solos. Un texto en tercera
   persona todavía puede sonar a alguien opinando.

Servidor local para ir viendo cambios: `cd web && npm run dev` (queda en segundo plano;
`npx astro dev stop` para bajarlo). Recarga sola al guardar. El JSON de datos no: ese
hay que regenerarlo con el `make` del capítulo.

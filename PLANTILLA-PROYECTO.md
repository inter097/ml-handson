# Plantilla para un proyecto nuevo

Guion para empezar con un dataset cualquiera aplicando lo que el libro y este cuaderno
ya dejaron medido. No es teoría: cada regla trae el número que la sostiene, y cuando el
número salió de una medición propia, se cita de dónde.

Crece con los capítulos. Al terminar cada uno se revisa qué de aquí cambió.

Estado: capítulos 1, 2 y 3.

Última revisión: 2026-07-31.

---

## Fase 0 · Antes de cargar nada

Cuatro preguntas cuya respuesta cambia todo lo demás. Contestarlas por escrito.

1. **¿Qué se predice y en qué unidad?** Cantidad o categoría. Si es cantidad, en qué
   escala, y si hace falta transformarla (un objetivo con cola larga suele pedir
   logaritmo, y entonces el error se interpreta en esa escala).
2. **¿Cómo se va a usar la predicción?** Por lotes cada noche o en línea con latencia.
   Esto decide el tamaño de modelo admisible mucho antes que la precisión.
3. **¿Qué métrica decide?** No la que suene bien: la que corresponde al costo real de
   equivocarse. Ver la tabla de la fase 6.
4. **¿Cuál es la línea base?** La regla tonta que el modelo tiene que superar: la
   mediana, la clase mayoritaria, o lo que se esté usando hoy. Sin línea base, un RMSE
   no significa nada.

Y el límite de la máquina, que no es un detalle: **16 GB sin margen**. Antes de un
entrenamiento pesado, proyectar el pico y decirlo. Los scripts que pueden crecer llevan
`--check`, que proyecta sin reservar y aborta si el pico estimado pasa del 45% de la RAM.

---

## Fase 1 · Mirar lo justo, y partir

El orden importa y es contraintuitivo: **se mira poco, se parte, y el análisis de
verdad va después**. Mirar el dataset entero antes de partir es la forma más común de
contaminar el test, porque la contaminación no pasa por el código sino por las
decisiones que se toman después de haber visto.

```python
df.info()                    # tipos, nulos, memoria
df.describe()                # rangos, colas, valores imposibles
df["categoria"].value_counts()
```

Con eso basta para partir. Los histogramas y las correlaciones van en la fase 3, sobre
el entrenamiento.

### Elegir el tipo de partición: cuatro preguntas en orden

1. **¿Los datos tienen fecha y el modelo predice el futuro?**
   Corte temporal, no aleatorio. Una partición aleatoria mete el futuro en el
   entrenamiento y el error sale ridículo. La validación cruzada también cambia, a
   `TimeSeriesSplit`. Aquí termina la decisión.
2. **¿Varias filas pertenecen a la misma entidad?** (paciente, usuario, tienda)
   `GroupShuffleSplit` o `GroupKFold`. Si las filas de un paciente se reparten entre las
   dos partes, el modelo lo memoriza y la prueba mide memoria, no generalización.
3. **¿El dataset va a crecer o a regenerarse?**
   Hash de un identificador estable. Si es un archivo congelado, basta la semilla.
4. **¿Hay una variable dominante con estratos desiguales, o pocas filas?**
   Estratificar, encima de lo que haya salido de las tres anteriores.

Las preguntas 1 y 2 no están en el capítulo 2 y son las que más modelos rompen en
producción.

### Semilla o hash

Compiten: se elige una, poner las dos no aporta.

```python
# Archivo congelado
train_test_split(df, test_size=0.2, random_state=42)

# Dataset que crece: la decisión se toma fila por fila, sin mirar el resto
def en_test(identificador, ratio=0.2):
    return crc32(np.int64(identificador)) < ratio * 2**32
```

El hash aguanta que se añadan o borren filas sin mover ninguna de lado. El identificador
tiene que ser una propiedad de la fila, no su posición en el archivo: un índice cambia
en cuanto se borra algo.

### Estratificar

Ortogonal a lo anterior y se combina con cualquiera.

```python
df["estrato"] = pd.cut(df["variable_dominante"],
                       bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
train, test = train_test_split(df, test_size=0.2, stratify=df["estrato"], random_state=42)
for parte in (train, test):
    parte.drop("estrato", axis=1, inplace=True)   # era andamio, no variable
```

Medido en California Housing: una partición aleatoria simple **subrepresentaba el
estrato más pobre en 9.4%**. Ese número es lo que justifica el rodeo.

Comprobar que salió bien, no suponerlo:

```python
(test["estrato"].value_counts(normalize=True) /
 df["estrato"].value_counts(normalize=True) - 1).round(4)
```

### La regla que da sentido a todo

**El test no se toca hasta la evaluación final.** Ni para mirar, ni para elegir modelo,
ni para decidir un umbral. Toda comparación sale de validación cruzada sobre el
entrenamiento.

- [ ] Tipo de partición elegido con las cuatro preguntas
- [ ] Semilla o hash, no ambos
- [ ] Estratificación verificada con números
- [ ] Columna de estrato borrada
- [ ] El test guardado y sin abrir

---

## Fase 2 · Explorar, ya solo sobre el entrenamiento

Sobre una copia, para no estropear el original.

```python
explora = train.copy()
explora.hist(bins=50, figsize=(12, 8))
explora.corr(numeric_only=True)["objetivo"].sort_values(ascending=False)
```

Qué buscar, en este orden:

- **Valores imposibles y topes.** Un máximo que se repite mucho suele ser un techo
  administrativo (el valor 500,001 de California Housing). Decidir si se recorta, se
  marca con una bandera, o se dejan fuera esas filas.
- **Colas largas.** Asimetría alta. Candidatas a logaritmo, pero medir: en este
  cuaderno el logaritmo del libro **empeoró 0.0059 de RMSE** sobre un SVR, porque los
  datos de sklearn ya venían promediados por vivienda y la cola que el logaritmo venía
  a comprimir ya no era la del libro.
- **Correlaciones**, sabiendo que solo ven relaciones lineales. Una parábola perfecta da
  correlación cero.
- **Combinaciones de variables.** Los cocientes suelen ganarle a sus componentes
  (habitaciones por vivienda le gana a habitaciones totales).

- [ ] Histogramas de todo
- [ ] Topes y valores imposibles decididos
- [ ] Correlaciones con el objetivo
- [ ] Dos o tres combinaciones probadas

---

## Fase 3 · Preprocesamiento, dentro del Pipeline

**Nada de transformar antes de partir, y nada de transformar fuera del `Pipeline`.**

Un `StandardScaler` ajustado antes de la partición aprende la media de los datos de
validación, y el error sale optimista sin que nada falle ni avise. Dentro del
`Pipeline`, sklearn lo reajusta en cada pliegue.

```python
Pipeline([
    ("prep", ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("esc", StandardScaler())]), numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ])),
    ("modelo", estimador),
])
```

Detalles que ya costaron una depuración cada uno:

- `handle_unknown="ignore"` en el one-hot. Una categoría rara puede no aparecer en algún
  pliegue de la validación cruzada, y sin eso revienta.
- Mediana y no media al imputar cuando la distribución tiene cola.
- La imputación va **antes** del escalado, y las dos dentro de la misma rama.

### Qué modelo necesita qué

| Modelo | Escalado | Colas largas | One-hot |
|---|---|---|---|
| Lineales, SVM, k-NN, redes | Obligatorio | Sensibles: miden distancias | Sí |
| Árboles, bosques, boosting | Indiferente | Indiferentes: solo miran el orden | Sí (o categórico nativo) |

Por eso un logaritmo no cambia nada en un bosque y sí en un SVR. Es la misma razón por
la que una tabla de resultados mezcla modelos que necesitan cosas distintas del mismo
pipeline.

### Transformadores propios

Cuando hace falta uno, el contrato mínimo, y **verificado con `check_estimator`**, que
encuentra lo que la prueba a mano no ve:

```python
class MiTransformador(TransformerMixin, BaseEstimator):   # el orden importa
    def __init__(self, parametro=1):
        self.parametro = parametro          # sin validar y sin renombrar

    def fit(self, X, y=None):
        if hasattr(X, "columns"):           # antes de validar: después ya no hay nombres
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        X = validate_data(self, X)          # guarda n_features_in_
        self.aprendido_ = ...               # guion bajo final
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        ...
```

En este cuaderno, ese clon tenía **cuatro defectos** y dos solo aparecieron con
`check_estimator`: los nombres de columna nunca se guardaban porque se leían después de
validar, y los mixins estaban heredados al revés.

### La fuga silenciosa

Un transformador que entrena un modelo dentro (k-vecinos, medias por categoría,
codificación por objetivo) **filtra el objetivo si predice sobre las mismas filas con
las que se ajustó**. La solución es asimétrica:

```python
def fit_transform(self, X, y=None, **kw):
    self.fit(X, y)
    return self.oof_          # cross_val_predict: cada fila la predice un modelo que no la vio

def transform(self, X):
    return self.modelo_.predict(X)    # datos nuevos: no hay solapamiento posible
```

Medido: con la fuga, la columna tenía **correlación 0.993** con el objetivo; corregida,
**0.845**. La solución oficial del libro para ese ejercicio tiene la fuga.

- [ ] Todo el preprocesamiento dentro del `Pipeline`
- [ ] Nada ajustado antes de partir
- [ ] `check_estimator` sobre cada transformador propio
- [ ] Los transformadores con modelo dentro, con predicciones fuera de pliegue

---

## Fase 4 · Comparar modelos con validación cruzada

Varios modelos de familias distintas, con los parámetros por defecto, antes de afinar
nada. Afinar el modelo equivocado es la forma más cara de perder una tarde.

```python
for nombre, modelo in candidatos.items():
    scores = cross_val_score(build_pipeline(modelo, X_train), X_train, y_train,
                             cv=5, scoring="neg_root_mean_squared_error")
    print(f"{nombre}: {-scores.mean():.4f} ± {scores.std():.4f}")
```

**La desviación importa tanto como la media.** Dos modelos con la misma media y
desviaciones distintas no son el mismo modelo.

Familias que conviene tener en la primera ronda: una lineal (línea base), un árbol
solo (para ver el sobreajuste), un bosque, un boosting, y si el tamaño lo permite un
SVM o un k-NN.

- [ ] Al menos cuatro familias
- [ ] Media y desviación de cada una
- [ ] Comparadas contra la línea base de la fase 0

---

## Fase 5 · Ajustar hiperparámetros

`RandomizedSearchCV` por defecto. La rejilla solo cuando el espacio es pequeño y
discreto.

```python
param_distribs = {
    "modelo__C": loguniform(1, 1000),      # escala: logaritmo-uniforme
    "modelo__max_depth": randint(3, 30),   # conteo: entera
}
```

Por qué la aleatoria: con una rejilla de 8 × 6, cada hiperparámetro toma solo 8 o 6
valores distintos por muchas combinaciones que se prueben. Con 48 sorteos hay 48 valores
distintos en cada eje. Cuando pocos hiperparámetros dominan, que es lo normal, la
aleatoria encuentra mejores valores con el mismo presupuesto (Bergstra y Bengio, 2012).

**`loguniform` y no `uniform` para todo lo que importe de forma multiplicativa** (`C`,
`gamma`, tasas de aprendizaje, regularización). Con una uniforme entre 20 y 200,000, el
99% de los sorteos cae por encima de 2,000 y la zona baja no se explora nunca.

Tres reglas que costaron su medición:

- **Si el óptimo toca el borde del rango, el rango está mal.** Con el número de barrios
  acotado a 30, la búsqueda elegía exactamente 30. Se extendió a 60.
- **El preprocesamiento también se busca.** Estrategia de imputación, número de
  clusters, parámetros de los transformadores propios. Solo es posible porque vive
  dentro del `Pipeline`.
- **Guardar todos los candidatos, no solo el ganador.** Cada uno como ejecución anidada
  en MLflow. Sin eso, `cv_results_` se descarta al terminar y queda una cifra sin
  contexto. Con 137 candidatos guardados se puede ver después qué movió la aguja.

### Antes de lanzar una búsqueda larga

**Proyectar el costo con lo ya medido, no con la intuición.** `ParameterSampler` con la
misma semilla devuelve los mismos candidatos que va a sortear la búsqueda, así que se
puede estimar cada uno antes de ajustar nada.

Ejemplo real de este cuaderno: la búsqueda del libro proyectaba **18.1 horas**, y el 90%
se iba en siete candidatos de un tramo que otra medición ya había mostrado plano. Con un
tope por candidato bajó a 1.7 horas, y los siete saltados quedaron anotados con su
proyección.

**Tope de dos horas de reloj proyectadas.** Por encima, se recorta y se publica el
número por el que se recortó.

Y el que tumbó la máquina una vez: **cuidado con el paralelismo anidado.** Un
`GridSearchCV(n_jobs=-1)` sobre un estimador que ya lleva `n_jobs=-1` copia los datos por
cada proceso y agota la memoria.

- [ ] Distribuciones continuas, escala correcta
- [ ] Ningún óptimo tocando un borde
- [ ] Opciones de preprocesamiento en el mismo espacio
- [ ] Costo proyectado antes de lanzar
- [ ] Sin `n_jobs=-1` anidado

---

## Fase 6 · La métrica, según el problema

### Regresión

| Métrica | Cuándo |
|---|---|
| RMSE | Por defecto. Castiga los errores grandes |
| MAE | Hay atípicos que no interesa perseguir |
| Error relativo | El error tolerable escala con la magnitud |

### Clasificación

**La exactitud no sirve con clases desbalanceadas.** Con un 10% de positivos, decir
siempre «no» da 90% de exactitud y cero utilidad.

| Métrica | Cuándo |
|---|---|
| Precisión | Una falsa alarma cuesta cara |
| Recall | Un caso perdido cuesta caro |
| F1 | Las dos importan parecido |
| ROC AUC | Clases equilibradas, comparar modelos |
| PR AUC | **Positivos raros.** La ROC se ve engañosamente buena |

La matriz de confusión antes que cualquier número resumen: dice qué se confunde con qué,
que es lo que sugiere la siguiente feature.

```python
y_pred = cross_val_predict(pipe, X_train, y_train, cv=3)   # predicciones limpias
ConfusionMatrixDisplay.from_predictions(y_train, y_pred, normalize="true")
```

El umbral es una decisión aparte del modelo, y se elige sobre la curva de precisión
contra recall, con la validación cruzada, nunca con el test.

---

## Fase 7 · Analizar antes de dar por bueno

- **Importancia de variables.** Las que no aportan se pueden quitar, pero medir el
  efecto: en este cuaderno un `SelectFromModel` que descartaba la mitad de las columnas
  **costó 0.0068 de RMSE**. La selección no es gratis por defecto.
- **Curvas de aprendizaje.** Distinguen «faltan datos» de «falta capacidad».
- **Los peores errores, mirados de uno en uno.** Es donde aparecen las variables que
  faltan y las etiquetas mal puestas.
- **Costo de servir.** Tamaño del artefacto, memoria y latencia por predicción. Un
  modelo mejor que no cabe en el presupuesto no es mejor.

---

## Fase 8 · El test, una sola vez

```python
final = mejor_pipeline.fit(X_train, y_train)      # reajustado con todo el entrenamiento
rmse = root_mean_squared_error(y_test, final.predict(X_test))
```

Y el intervalo de confianza, porque una cifra sola no dice si la diferencia con otro
modelo es real:

```python
errores = (final.predict(X_test) - y_test) ** 2
stats.t.interval(0.95, len(errores)-1, loc=errores.mean(),
                 scale=stats.sem(errores)) ** 0.5
```

**Si el resultado en test es peor que en validación, no se vuelve a afinar.** Afinar
contra el test lo convierte en un segundo conjunto de validación y deja el proyecto sin
estimación honesta. Se reporta la diferencia y se explica.

- [ ] Test usado una vez
- [ ] Intervalo de confianza
- [ ] Diferencia con la validación explicada

---

## Fase 9 · Servir el modelo en el navegador

Un modelo que predice en la página, sin servidor, es la demo más convincente de un
portafolio. Lo que hay que saber antes de intentarlo:

- **El tamaño manda sobre la precisión.** Aquí el campeón ocupaba 703 MB de artefacto y
  quedó descartado; el que cabía pesaba 2.4 MB y costaba 0.023 de RMSE. Los árboles se
  exportan a JSON, los modelos lineales son cuatro números, y las redes ya piden un
  runtime aparte.
- **Se verifica contra Python, siempre.** El exportador deja unas filas con su predicción
  y la página las comprueba al cargar. Sin eso, una demo que se desvía sigue devolviendo
  números creíbles.
- **Cuidado con la precisión numérica.** XGBoost compara umbrales en float32 y JavaScript
  calcula en float64: comparar en doble precisión desviaba la predicción 0.0155. Y
  redondear los umbrales al exportarlos, hasta 0.35.
- **El presupuesto se mide contra el servidor, no en local.** Vercel comprime al vuelo con
  un nivel bajo de brotli: el archivo que en local daba 186 KB llegaba en 296.
- **Las entradas que el usuario no da hay que rellenarlas**, y con qué se rellenan importa.
  La mediana global rompía los casos raros; los valores del caso real, cuando existen,
  cuestan 103 dólares de desvío frente a 6,385.

---

## Fase 10 · Guardar y vigilar

- El `Pipeline` entero serializado, no solo el modelo: el preprocesamiento va dentro.
- La versión de los datos, del código y de las librerías junto al artefacto.
- Métricas en producción, porque **el modelo se pudre**: el mundo cambia y los datos de
  entrada dejan de parecerse a los de entrenamiento.
- Una alarma sobre la distribución de las entradas, que se degrada antes que la métrica.

---

## Los errores que ya se cometieron aquí

Vale más que cualquier lista de buenas prácticas, porque cada uno pasó de verdad.

| Error | Cómo se vio | Cuánto costó |
|---|---|---|
| Suponer que SVR escala con n² | Medido: n^1.33 | Media tarde buscando en el sitio equivocado |
| Suponer que el logaritmo del libro ayudaba | 8 mejoras de 42 candidatos | 16 minutos, y un resultado publicable |
| Un transformador que predice sobre sus propias filas | Correlación 0.993 contra 0.845 | Una feature que parecía la mejor del proyecto |
| Un clon de transformador que «funcionaba» | `check_estimator` sacó cuatro fallos | Una página publicada afirmando lo que no era |
| Un rango de búsqueda que tocaba el borde | El óptimo salía siempre en 30 | Una búsqueda entera repetida |
| Contar ejercicios de memoria | El notebook oficial tenía seis, no ocho | Dos ejercicios inventados, publicados |
| `n_jobs=-1` anidado | La máquina se quedó sin memoria | Un reinicio |
| Medir el tamaño comprimido en local | El servidor sirve otro nivel de brotli | 110 KB de presupuesto imaginario |
| Redondear umbrales «que no se notan» | La predicción se desviaba 0.35 | Una demo que mentía con cara de seguridad |

La regla que sale de todos: **si algo se afirma, se mide.** Y si no se puede medir, no
se escribe.

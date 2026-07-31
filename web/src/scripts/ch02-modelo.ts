/**
 * El modelo del capítulo 2, prediciendo en el navegador
 *
 * Reproduce lo que hace el Pipeline de scikit-learn, paso a paso y en el mismo
 * orden, porque el modelo espera sus 36 columnas exactamente donde las aprendió:
 *
 *   1. Los tres cocientes de features.py
 *   2. Imputación por mediana y estandarización, sobre las 11 numéricas
 *   3. Similitud RBF a los 20 barrios, desde latitud y longitud
 *   4. One-hot de ocean_proximity, 5 columnas
 *   5. Los 300 árboles, sumados sobre base_score
 *
 * El JSON lo genera `ch02-california-housing/src/export_demo.py`, que además
 * mete cinco filas con su predicción de Python. `verificarParidad` las comprueba
 * al cargar: si el número no coincide, la demo estaría mintiendo con cara de
 * seguridad, y así se ve en consola en lugar de descubrirse tarde.
 */

type Nodo = { v: number } | { f: number; u: number; i: Nodo; d: Nodo };

export type Modelo = {
  modelo: string;
  rmse: number;
  ic_campeon: [number, number];
  base_score: number;
  prep: {
    columnas_numericas: string[];
    medianas_imputador: number[];
    media_escalador: number[];
    escala_escalador: number[];
    centros_barrios: number[][];
    gamma_barrios: number;
    categorias: string[];
    salida: string[];
  };
  medianas_ocultas: Record<string, number>;
  rangos: Record<string, { min: number; max: number; inicio: number }>;
  penalizacion_ocultas: number;
  arboles: Nodo[];
  paridad: { entrada: Record<string, number | string | null>; prediccion: number }[];
};

/** Lo que la interfaz recoge: seis números, un punto y una categoría. */
export type Caso = {
  MedInc: number;
  HouseAge: number;
  AveRooms: number;
  AveBedrms: number;
  Population: number;
  AveOccup: number;
  Latitude: number;
  Longitude: number;
  ocean_proximity: string;
};

/**
 * Las 36 columnas, en el orden en que el ColumnTransformer las dejó.
 *
 * Los cocientes se calculan aquí y no en la interfaz porque son parte del
 * preprocesamiento, no del caso: quien usa la demo no los teclea.
 */
function preparar(caso: Caso, m: Modelo): number[] {
  const p = m.prep;
  const derivadas: Record<string, number> = {
    ...caso,
    rooms_per_household: caso.AveRooms / caso.AveOccup,
    bedrooms_ratio: caso.AveBedrms / caso.AveRooms,
    population_per_household: caso.Population / caso.AveOccup,
  };

  // Rama numérica: imputar y estandarizar, en ese orden.
  const num = p.columnas_numericas.map((col, i) => {
    const bruto = derivadas[col];
    const valor = Number.isFinite(bruto) ? bruto : p.medianas_imputador[i];
    return (valor - p.media_escalador[i]) / p.escala_escalador[i];
  });

  // Rama geográfica: similitud RBF a cada centro, exp(-gamma · d²). Los centros
  // están en las unidades crudas de latitud y longitud, sin escalar, igual que
  // en el KMeans que los encontró.
  const geo = p.centros_barrios.map(([lat, lon]) => {
    const d2 = (caso.Latitude - lat) ** 2 + (caso.Longitude - lon) ** 2;
    return Math.exp(-p.gamma_barrios * d2);
  });

  // Rama categórica: one-hot con handle_unknown="ignore", o sea que una
  // categoría desconocida deja las cinco columnas en cero en vez de fallar.
  const cat = p.categorias.map((c) => (c === caso.ocean_proximity ? 1 : 0));

  return [...num, ...geo, ...cat];
}

/** Recorre un árbol. XGBoost va por la izquierda cuando el valor es menor. */
function hoja(nodo: Nodo, x: Float32Array): number {
  let actual = nodo;
  while (!("v" in actual)) {
    actual = x[actual.f] < Math.fround(actual.u) ? actual.i : actual.d;
  }
  return actual.v;
}

/**
 * La predicción: base_score más la suma de las 300 hojas.
 *
 * El vector va en `Float32Array` porque **XGBoost compara en float32**, y en
 * JavaScript todo es float64 por defecto. Con los umbrales comparados en doble
 * precisión, las filas que caen junto a un corte se iban por la rama contraria
 * y la predicción se desviaba hasta 0.0155. El preprocesamiento sí se calcula
 * en doble precisión, igual que en Python, y solo se estrecha al final.
 */
export function predecir(caso: Caso, m: Modelo): number {
  const x = Float32Array.from(preparar(caso, m));
  let suma = m.base_score;
  for (const arbol of m.arboles) suma += hoja(arbol, x);
  return suma;
}

/**
 * Compara contra las cinco filas que Python dejó en el JSON.
 *
 * La tolerancia es 1e-4 y no cero: los umbrales se exportaron redondeados a
 * cinco decimales, así que un valor que caiga justo encima de un corte puede
 * irse por la otra rama. Una diferencia mayor significa que algún paso del
 * preprocesamiento no se reprodujo igual.
 */
export function verificarParidad(m: Modelo, tolerancia = 1e-4): boolean {
  let todas = true;
  for (const { entrada, prediccion } of m.paridad) {
    const propia = predecir(entrada as unknown as Caso, m);
    const error = Math.abs(propia - prediccion);
    if (error > tolerancia) {
      console.error(
        `[ch02] Paridad rota: Python ${prediccion.toFixed(6)}, ` +
        `navegador ${propia.toFixed(6)}, diferencia ${error.toExponential(2)}`,
      );
      todas = false;
    }
  }
  return todas;
}

export async function cargarModelo(url = "/data/ch02-modelo.json"): Promise<Modelo> {
  const respuesta = await fetch(url);
  if (!respuesta.ok) throw new Error(`No se pudo cargar el modelo: ${respuesta.status}`);
  return respuesta.json();
}

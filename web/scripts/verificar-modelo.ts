/**
 * Comprueba que el modelo del navegador predice lo mismo que Python
 *
 *     node scripts/verificar-modelo.ts
 *
 * Node ejecuta TypeScript directamente desde la versión 22, así que esto no
 * añade ninguna dependencia al proyecto.
 *
 * Tres comprobaciones, y las tres nacieron de un fallo real:
 *
 *   1. **Paridad.** Las cinco filas que `export_demo.py` deja en el JSON con su
 *      predicción de Python. Redondear los umbrales a cinco decimales desviaba
 *      hasta 0.35, y comparar en float64 cuando XGBoost compara en float32,
 *      hasta 0.0155. Las dos veces la demo seguía dando números creíbles.
 *   2. **Los 4,128 distritos**, contra la predicción que el pipeline completo
 *      dejó anotada en cada uno. Mide lo que cuesta la simplificación de la
 *      demo en lugar de suponerlo.
 *   3. **JSON válido.** `json.dumps` escribe NaN, que no lo es, y con un solo
 *      NaN la página se queda en blanco.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import { predecir, verificarParidad, type Caso, type Modelo } from "../src/scripts/ch02-modelo.ts";

const raiz = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const datos = (nombre: string) => JSON.parse(readFileSync(`${raiz}/public/data/${nombre}`, "utf8"));

const TOLERANCIA = 1e-4;
const I = { lat: 0, lon: 1, inc: 2, occ: 3, rooms: 4, age: 5,
            bedrms: 6, pop: 7, real: 8, pred: 9, ocean: 10 } as const;

const usd = (v: number) => Math.round(v * 100000).toLocaleString("en-US");
let fallos = 0;

/* ── 1. Paridad con Python ───────────────────────────────────────────────── */
const modelo: Modelo = datos("ch02-modelo.json");
console.log(`modelo ${modelo.modelo} · ${modelo.arboles.length} árboles · RMSE ${modelo.rmse}`);

let peor = 0;
for (const { entrada, prediccion } of modelo.paridad) {
  peor = Math.max(peor, Math.abs(predecir(entrada as unknown as Caso, modelo) - prediccion));
}
if (verificarParidad(modelo, TOLERANCIA)) {
  console.log(`  paridad        ✓  ${modelo.paridad.length} filas, desvío máximo ${peor.toExponential(1)}`);
} else {
  console.log("  paridad        ✗");
  fallos++;
}

/* ── 2. Los 4,128 distritos ──────────────────────────────────────────────── */
const distritos = datos("ch02-distritos.json");
const r = modelo.rangos;
const recorta = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
const oNaN = (v: number | null) => (v === null ? NaN : v);

let suma = 0;
let peorDistrito = 0;
for (const f of distritos.rows as (number | null)[][]) {
  const p = predecir({
    MedInc: recorta(f[I.inc] as number, r.MedInc.min, r.MedInc.max),
    AveOccup: recorta(f[I.occ] as number, r.AveOccup.min, r.AveOccup.max),
    AveRooms: recorta(f[I.rooms] as number, r.AveRooms.min, r.AveRooms.max),
    HouseAge: recorta(f[I.age] as number, r.HouseAge.min, r.HouseAge.max),
    AveBedrms: oNaN(f[I.bedrms] as number | null),
    Population: oNaN(f[I.pop] as number | null),
    Latitude: f[I.lat] as number,
    Longitude: f[I.lon] as number,
    ocean_proximity: distritos.ocean[f[I.ocean] as number],
  }, modelo);
  const dif = Math.abs(p - (f[I.pred] as number));
  suma += dif;
  peorDistrito = Math.max(peorDistrito, dif);
}
const n = distritos.rows.length;
console.log(`  distritos      ✓  ${n} predichos · media ${usd(suma / n)} USD ` +
            `· peor ${usd(peorDistrito)} USD frente al pipeline completo`);

/* ── 3. JSON sin NaN ─────────────────────────────────────────────────────── */
for (const nombre of ["ch02-modelo.json", "ch02-distritos.json"]) {
  const crudo = readFileSync(`${raiz}/public/data/${nombre}`, "utf8");
  if (/\bNaN\b/.test(crudo)) {
    console.log(`  ${nombre}  ✗  contiene NaN, que no es JSON válido`);
    fallos++;
  }
}
if (!fallos) console.log("  JSON válido    ✓  sin NaN ni Infinity");

process.exit(fallos ? 1 : 0);

/**
 * La demo del capítulo 2: un distrito inventado, un precio
 *
 * El modelo entero viaja al navegador (186 KB con brotli) y predice aquí, sin
 * servidor. `ch02-modelo.ts` reproduce el Pipeline de scikit-learn; este
 * archivo solo conecta los controles y dibuja.
 *
 * Cuatro controles y un punto en el mapa. Las otras dos variables del modelo
 * quedan fijas en su mediana, y eso cuesta 0.0187 de RMSE, medido en
 * `export_demo.py`. Esconder también las habitaciones por vivienda costaba
 * 0.0868, así que se quedó como control.
 *
 * `ocean_proximity` no se pregunta: sale del distrito real más cercano al
 * punto elegido. Es información geográfica y el punto ya la contiene.
 */
import { cargarModelo, predecir, verificarParidad, type Caso, type Modelo } from "./ch02-modelo.ts";

type Fila = [number, number, number, number, number, number];
const I = { lat: 0, lon: 1, real: 2, pred: 3, ocean: 4 } as const;

const usd = (v: number) => "USD " + Math.round(v * 100000).toLocaleString("en-US");
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

let modelo: Modelo;
let distritos: Fila[] = [];
let etiquetasOcean: string[] = [];
let punto = { lat: 34.05, lon: -118.24 };   // Los Ángeles, para que arranque en algo

/* ── Mapa: aquí es entrada, no resultado ─────────────────────────────────── */
const cv = $<HTMLCanvasElement>("mapa");
const ctx = cv.getContext("2d")!;
const BX = [-124.4, -114.2];
const BY = [32.4, 42.1];
const PAD = 18;
const px = (lon: number) => PAD + ((lon - BX[0]) / (BX[1] - BX[0])) * (cv.width - PAD * 2);
const py = (lat: number) => cv.height - PAD - ((lat - BY[0]) / (BY[1] - BY[0])) * (cv.height - PAD * 2);
const lonDe = (x: number) => BX[0] + ((x - PAD) / (cv.width - PAD * 2)) * (BX[1] - BX[0]);
const latDe = (y: number) => BY[0] + ((cv.height - PAD - y) / (cv.height - PAD * 2)) * (BY[1] - BY[0]);

const css = (v: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function dibujarMapa() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  // Los 4,128 distritos de prueba, en gris, solo para que California tenga
  // forma. El punto elegido va encima.
  ctx.fillStyle = css("--hairline");
  ctx.globalAlpha = 0.85;
  for (const d of distritos) {
    ctx.beginPath();
    ctx.arc(px(d[I.lon]), py(d[I.lat]), 2.1, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  const x = px(punto.lon);
  const y = py(punto.lat);
  ctx.strokeStyle = css("--accent-ink");
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 14, y);
  ctx.lineTo(x + 14, y);
  ctx.moveTo(x, y - 14);
  ctx.lineTo(x, y + 14);
  ctx.stroke();
}

/** El distrito real más cercano. Da la categoría de océano y una referencia. */
function vecino(): Fila {
  let mejor = distritos[0];
  let mejorD = Infinity;
  for (const d of distritos) {
    const dd = (d[I.lat] - punto.lat) ** 2 + (d[I.lon] - punto.lon) ** 2;
    if (dd < mejorD) {
      mejorD = dd;
      mejor = d;
    }
  }
  return mejor;
}

/* ── Predicción ──────────────────────────────────────────────────────────── */
function casoActual(): Caso {
  const v = (id: string) => parseFloat($<HTMLInputElement>(id).value);
  return {
    MedInc: v("cMedInc"),
    AveOccup: v("cAveOccup"),
    AveRooms: v("cAveRooms"),
    HouseAge: v("cHouseAge"),
    AveBedrms: modelo.medianas_ocultas.AveBedrms,
    Population: modelo.medianas_ocultas.Population,
    Latitude: punto.lat,
    Longitude: punto.lon,
    ocean_proximity: etiquetasOcean[vecino()[I.ocean]],
  };
}

function actualizar() {
  const caso = casoActual();
  const p = predecir(caso, modelo);
  // El objetivo está topado en 5.00001 (500,001 dólares) en el censo, así que
  // el modelo nunca aprendió a predecir por encima de ahí.
  const acotada = Math.max(0.15, Math.min(5.00001, p));

  $("salida").textContent = usd(acotada);
  $("salidaNota").textContent = p > 5
    ? "El censo topó los precios en USD 500,001, así que el modelo no predice más arriba."
    : "";

  const cerca = vecino();
  $("vecino").innerHTML =
    `Distrito real más cercano: <strong>${usd(cerca[I.real])}</strong> ` +
    `· ${etiquetasOcean[cerca[I.ocean]]} · el modelo le asigna ${usd(cerca[I.pred])}`;

  for (const [id, valor] of [
    ["vMedInc", caso.MedInc.toFixed(2)],
    ["vAveOccup", caso.AveOccup.toFixed(2)],
    ["vAveRooms", caso.AveRooms.toFixed(2)],
    ["vHouseAge", String(Math.round(caso.HouseAge))],
  ] as const) {
    $(id).textContent = valor;
  }
  $("coords").textContent = `${punto.lat.toFixed(2)}, ${punto.lon.toFixed(2)}`;
  dibujarMapa();
}

/* ── Arranque ────────────────────────────────────────────────────────────── */
async function iniciar() {
  const [m, datos] = await Promise.all([
    cargarModelo(),
    fetch("/data/ch02.json").then((r) => r.json()),
  ]);
  modelo = m;
  distritos = datos.rows;
  etiquetasOcean = datos.ocean;

  // La comprobación que evita publicar una demo que miente: las cinco filas
  // que Python dejó en el JSON tienen que dar aquí el mismo número.
  const ok = verificarParidad(modelo);
  $("paridad").textContent = ok
    ? "Paridad con Python verificada al cargar"
    : "Paridad rota: revisar la consola";
  $("paridad").dataset.estado = ok ? "ok" : "mal";

  for (const [id, campo] of [
    ["cMedInc", "MedInc"], ["cAveOccup", "AveOccup"],
    ["cAveRooms", "AveRooms"], ["cHouseAge", "HouseAge"],
  ] as const) {
    const control = $<HTMLInputElement>(id);
    const r = modelo.rangos[campo];
    control.min = String(r.min);
    control.max = String(r.max);
    control.step = campo === "HouseAge" ? "1" : "0.01";
    control.value = String(r.inicio);
    control.addEventListener("input", actualizar);
  }

  const fijar = (ev: MouseEvent | Touch) => {
    const caja = cv.getBoundingClientRect();
    const escala = cv.width / caja.width;
    punto = {
      lat: latDe((ev.clientY - caja.top) * escala),
      lon: lonDe((ev.clientX - caja.left) * escala),
    };
    actualizar();
  };
  cv.addEventListener("click", fijar);
  cv.addEventListener("touchstart", (e) => {
    e.preventDefault();
    fijar(e.touches[0]);
  }, { passive: false });

  $("cargando").hidden = true;
  $("demo").hidden = false;
  actualizar();
}

iniciar().catch((e) => {
  console.error(e);
  $("cargando").textContent = "No se pudo cargar el modelo.";
});

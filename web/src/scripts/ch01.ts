/**
 * Demo del capítulo 1 — los dos estilos de generalización.
 *
 * Los dos modelos se evalúan aquí, en el navegador, sin servidor. Pueden:
 * la recta son dos números y el k-vecinos son 27 filas. Que quepan es
 * exactamente lo que el capítulo enseña, así que la demo lo encarna en vez
 * de explicarlo.
 */

interface Pais { pais: string; pib: number; satisfaccion: number }
interface Datos {
  k: number;
  r2: number;
  lineal: { pendiente: number; interseccion: number };
  nuevo: { nombre: string; pib: number; lineal: number; vecinos: number };
  paises: Pais[];
}

const css = (v: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const D: Datos = await fetch("/data/ch01.json").then((r) => r.json());
const { pendiente, interseccion } = D.lineal;

/* ── Los dos modelos ───────────────────────────────────────────────────── */

const porModelo = (pib: number) => pendiente * pib + interseccion;

/** Devuelve la predicción y qué países la produjeron. */
function porInstancia(pib: number): { valor: number; vecinos: number[] } {
  const orden = D.paises
    .map((p, i) => ({ i, d: Math.abs(p.pib - pib) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, D.k)
    .map((o) => o.i);
  const valor = orden.reduce((s, i) => s + D.paises[i].satisfaccion, 0) / orden.length;
  return { valor, vecinos: orden };
}

/* ── Escalas ───────────────────────────────────────────────────────────── */

const W = 680, H = 330;
const M = { t: 18, r: 18, b: 44, l: 46 };

const pibs = D.paises.map((p) => p.pib);
const sats = D.paises.map((p) => p.satisfaccion);
const X0 = Math.min(...pibs) * 0.92, X1 = Math.max(...pibs) * 1.04;
const Y0 = Math.min(...sats) - 0.45, Y1 = Math.max(...sats) + 0.35;

const px = (v: number) => M.l + ((v - X0) / (X1 - X0)) * (W - M.l - M.r);
const py = (v: number) => H - M.b - ((v - Y0) / (Y1 - Y0)) * (H - M.t - M.b);

/* ── Estado ────────────────────────────────────────────────────────────── */

let pibActual = D.nuevo.pib;

const svg = document.getElementById("grafica") as unknown as SVGSVGElement;
const slider = document.getElementById("pib") as HTMLInputElement;

slider.min = String(Math.round(X0));
slider.max = String(Math.round(X1));
slider.step = "50";
slider.value = String(Math.round(pibActual));

/* ── Dibujo ────────────────────────────────────────────────────────────── */

const el = (tag: string, attrs: Record<string, string | number>, texto?: string) => {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (texto !== undefined) n.textContent = texto;
  return n;
};

function dibujar() {
  const { valor: vVecinos, vecinos } = porInstancia(pibActual);
  const vLineal = porModelo(pibActual);
  const activos = new Set(vecinos);

  const COOL = css("--cool"), WARM = css("--warm");
  const MUTED = css("--muted"), HAIR = css("--hairline"), INK = css("--ink-2");

  svg.textContent = "";

  // Rejilla horizontal
  for (let v = Math.ceil(Y0 * 2) / 2; v <= Y1; v += 0.5) {
    svg.append(el("line", { x1: M.l, x2: W - M.r, y1: py(v), y2: py(v), stroke: HAIR, "stroke-width": 1 }));
    svg.append(el("text", {
      x: M.l - 8, y: py(v) + 4, "text-anchor": "end",
      fill: MUTED, "font-size": 10, "font-family": "var(--mono)",
    }, v.toFixed(1)));
  }

  // Eje X
  for (let v = 30000; v <= X1; v += 10000) {
    if (v < X0) continue;
    svg.append(el("text", {
      x: px(v), y: H - M.b + 18, "text-anchor": "middle",
      fill: MUTED, "font-size": 10, "font-family": "var(--mono)",
    }, (v / 1000) + "k"));
  }
  svg.append(el("text", {
    x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle",
    fill: MUTED, "font-size": 10.5, "font-family": "var(--mono)",
  }, "PIB per cápita (USD)"));

  // Escalera del k-vecinos
  const pasos: string[] = [];
  const N = 700;
  for (let i = 0; i <= N; i++) {
    const x = X0 + ((X1 - X0) * i) / N;
    pasos.push(`${i ? "L" : "M"}${px(x).toFixed(1)},${py(porInstancia(x).valor).toFixed(1)}`);
  }
  svg.append(el("path", {
    d: pasos.join(" "), fill: "none", stroke: COOL,
    "stroke-width": 2, "stroke-dasharray": "5 4",
  }));

  // Recta
  svg.append(el("line", {
    x1: px(X0), y1: py(porModelo(X0)), x2: px(X1), y2: py(porModelo(X1)),
    stroke: WARM, "stroke-width": 2,
  }));

  // Marcador vertical
  svg.append(el("line", {
    x1: px(pibActual), x2: px(pibActual), y1: M.t, y2: H - M.b,
    stroke: MUTED, "stroke-width": 1, "stroke-dasharray": "3 3",
  }));

  // Países. Los tres activos se resaltan: son los que el k-vecinos está usando.
  for (const [i, p] of D.paises.entries()) {
    const on = activos.has(i);
    svg.append(el("circle", {
      cx: px(p.pib), cy: py(p.satisfaccion), r: on ? 6.5 : 3.6,
      fill: on ? COOL : MUTED, "fill-opacity": on ? 1 : 0.45,
      stroke: on ? css("--paper") : "none", "stroke-width": 2,
    }));
  }

  // Las dos predicciones
  for (const [v, c] of [[vLineal, WARM], [vVecinos, COOL]] as [number, string][]) {
    svg.append(el("circle", {
      cx: px(pibActual), cy: py(v), r: 6,
      fill: c, stroke: css("--paper"), "stroke-width": 2.5,
    }));
  }

  // Los nombres van al final, sobre todo lo demás. Cuando dos vecinos caen
  // cerca en el eje X sus etiquetas se pisarían, así que se alternan arriba y
  // abajo; junto al borde se anclan hacia dentro para no salirse.
  const ordenados = [...vecinos].sort((a, b) => D.paises[a].pib - D.paises[b].pib);
  ordenados.forEach((i, n) => {
    const p = D.paises[i];
    const x = px(p.pib);
    const arriba = n % 2 === 0;
    const ancla = x < M.l + 50 ? "start" : x > W - M.r - 50 ? "end" : "middle";
    svg.append(el("text", {
      x, y: py(p.satisfaccion) + (arriba ? -13 : 21), "text-anchor": ancla,
      fill: INK, "font-size": 10.5, "font-family": "var(--mono)",
      stroke: css("--paper"), "stroke-width": 3.5, "paint-order": "stroke",
    }, p.pais));
  });

  // Lecturas
  document.getElementById("vLineal")!.textContent = vLineal.toFixed(2);
  document.getElementById("vVecinos")!.textContent = vVecinos.toFixed(2);
  document.getElementById("vPib")!.textContent =
    "USD " + Math.round(pibActual).toLocaleString("es-MX");
  // Quiénes son los vecinos ahora mismo
  document.getElementById("vecinos")!.innerHTML = vecinos
    .map((i) => {
      const p = D.paises[i];
      return `<li><span class="n">${p.pais}</span><span class="s">${p.satisfaccion.toFixed(1)}</span></li>`;
    })
    .join("");

  // La tabla de lo que guarda el k-vecinos, con los activos marcados
  document.getElementById("almacen")!.innerHTML = D.paises
    .map((p, i) =>
      `<span class="fila" data-on="${activos.has(i) ? 1 : 0}">${p.pais}</span>`)
    .join("");
}

/* ── Interacción ───────────────────────────────────────────────────────── */

slider.addEventListener("input", () => {
  pibActual = Number(slider.value);
  dibujar();
});

// Arrastrar sobre la gráfica hace lo mismo que el control.
const desdeEvento = (ev: PointerEvent) => {
  const b = svg.getBoundingClientRect();
  const x = ((ev.clientX - b.left) / b.width) * W;
  const v = X0 + ((x - M.l) / (W - M.l - M.r)) * (X1 - X0);
  pibActual = Math.min(Math.max(v, X0), X1);
  slider.value = String(Math.round(pibActual));
  dibujar();
};
let arrastrando = false;
svg.addEventListener("pointerdown", (ev) => {
  arrastrando = true;
  svg.setPointerCapture((ev as PointerEvent).pointerId);
  desdeEvento(ev as PointerEvent);
});
svg.addEventListener("pointermove", (ev) => arrastrando && desdeEvento(ev as PointerEvent));
svg.addEventListener("pointerup", () => (arrastrando = false));

document.getElementById("aChipre")!.addEventListener("click", () => {
  pibActual = D.nuevo.pib;
  slider.value = String(Math.round(pibActual));
  dibujar();
});

// Los colores salen de variables CSS: al cambiar de tema hay que repintar.
document.addEventListener("tema-cambiado", dibujar);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", dibujar);

dibujar();

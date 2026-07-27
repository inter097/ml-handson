/**
 * Interactividad del caso de estudio del capítulo 2.
 *
 * Las predicciones se calcularon una vez con el modelo entrenado y viven en
 * /data/ch02.json. Aquí solo se filtran y se recalculan estadísticas — eso sí
 * ocurre en el navegador, sobre datos reales, sin servidor detrás.
 */

type Fila = [number, number, number, number, number, number];
const I = { lat: 0, lon: 1, real: 2, pred: 3, ocean: 4, inc: 5 } as const;

const css = (v: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const usd = (v: number) => "USD " + Math.round(v * 100000).toLocaleString("en-US");
const f3 = (v: number) => v.toFixed(3);

/** Medido cargando cada artefacto y cronometrando 20 predicciones. */
const MODELOS = [
  { name: "extra_trees", rmse: 0.4063, lo: 0.3844, hi: 0.4271, mb: 703.4, ram: 1294, ms: 28.85 },
  { name: "random_forest", rmse: 0.4235, lo: 0.4015, hi: 0.4443, mb: 199.3, ram: null, ms: null },
  { name: "xgboost", rmse: 0.4291, lo: 0.4071, hi: 0.4500, mb: 2.4, ram: 91, ms: 2.25 },
  { name: "gradient_boosting", rmse: 0.4514, lo: 0.4292, hi: 0.4726, mb: 7.3, ram: null, ms: null },
];

const TODOS = [
  { name: "extra_trees", cv: 0.4085, rmse: 0.4063, r2: 0.8767 },
  { name: "random_forest", cv: 0.4179, rmse: 0.4235, r2: 0.8660 },
  { name: "xgboost", cv: 0.4230, rmse: 0.4291, r2: 0.8625 },
  { name: "gradient_boosting", cv: 0.4475, rmse: 0.4514, r2: 0.8478 },
  { name: "mlp", cv: 0.5007, rmse: 0.4967, r2: 0.8157 },
  { name: "svr", cv: null, rmse: 0.5091, r2: 0.8064 },
  { name: "decision_tree", cv: 0.5454, rmse: 0.5517, r2: 0.7726 },
  { name: "ridge", cv: 0.6051, rmse: 0.6282, r2: 0.7052 },
  { name: "linear_regression", cv: null, rmse: 0.6855, r2: 0.6489 },
];

const APORTES = [
  { k: "Similitud geográfica (K-means)", v: -0.0356, m: "random_forest" },
  { k: "Ajuste de hiperparámetros", v: -0.0284, m: "xgboost" },
  { k: "Similitud geográfica (K-means)", v: -0.0176, m: "xgboost" },
  { k: "Precio de las casas vecinas", v: -0.0146, m: "random_forest" },
  { k: "Precio de las casas vecinas", v: -0.0121, m: "xgboost" },
  { k: "Proximidad al océano", v: -0.0111, m: "random_forest" },
  { k: "Selección de variables", v: +0.0068, m: "random_forest" },
  { k: "Proximidad al océano", v: +0.0039, m: "xgboost" },
];

let ROWS: Fila[] = [];
let OCEAN: string[] = [];
let visibles: Fila[] = [];

const estado = { ocean: new Set<number>(), inc: new Set<number>(), modo: "error" };

/* ── Color ─────────────────────────────────────────────────────────────── */
const hex2rgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const mezclar = (a: number[], b: number[], t: number) =>
  a.map((v, i) => Math.round(v + (b[i] - v) * t));
const rgb = (c: number[]) => `rgb(${c[0]},${c[1]},${c[2]})`;

function colorDe(r: Fila): string {
  if (estado.modo === "error") {
    const e = r[I.pred] - r[I.real];
    const t = Math.max(-1, Math.min(1, e));
    const mid = hex2rgb(css("--mid"));
    return t < 0
      ? rgb(mezclar(mid, hex2rgb(css("--cool")), -t))
      : rgb(mezclar(mid, hex2rgb(css("--warm")), t));
  }
  const v = estado.modo === "real" ? r[I.real] : r[I.pred];
  const t = Math.max(0, Math.min(1, (v - 0.15) / (5.0 - 0.15)));
  return rgb(mezclar(hex2rgb(css("--mid")), hex2rgb(css("--cool")), 0.18 + t * 0.82));
}

/* ── Mapa ──────────────────────────────────────────────────────────────── */
const cv = document.getElementById("map") as HTMLCanvasElement;
const ctx = cv.getContext("2d")!;
const BX = [-124.4, -114.2];
const BY = [32.4, 42.1];
const PAD = 26;
const px = (lon: number) => PAD + ((lon - BX[0]) / (BX[1] - BX[0])) * (cv.width - PAD * 2);
const py = (lat: number) => cv.height - PAD - ((lat - BY[0]) / (BY[1] - BY[0])) * (cv.height - PAD * 2);

function dibujarMapa(filas: Fila[]) {
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.globalAlpha = 0.62;
  for (const r of filas) {
    ctx.fillStyle = colorDe(r);
    ctx.beginPath();
    ctx.arc(px(r[I.lon]), py(r[I.lat]), 3.1, 0, 6.2832);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

/* ── Tooltip ───────────────────────────────────────────────────────────── */
const tip = document.getElementById("tip")!;
cv.addEventListener("pointermove", (ev) => {
  const b = cv.getBoundingClientRect();
  const s = cv.width / b.width;
  const mx = (ev.clientX - b.left) * s;
  const my = (ev.clientY - b.top) * s;
  let best: Fila | null = null;
  let bd = 900;
  for (const r of visibles) {
    const dx = px(r[I.lon]) - mx;
    const dy = py(r[I.lat]) - my;
    const d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = r; }
  }
  if (!best) { tip.dataset.on = "0"; return; }
  const err = best[I.pred] - best[I.real];
  tip.innerHTML =
    `<div class="head">${OCEAN[best[I.ocean]]} · ${best[I.lat].toFixed(2)}, ${best[I.lon].toFixed(2)}</div>` +
    `<dl><dt>Real</dt><dd>${usd(best[I.real])}</dd>` +
    `<dt>Predicho</dt><dd>${usd(best[I.pred])}</dd>` +
    `<dt>Error</dt><dd style="color:${err < 0 ? "var(--cool)" : "var(--warm)"}">${err > 0 ? "+" : ""}${usd(err)}</dd></dl>`;
  tip.dataset.on = "1";
  const lx = px(best[I.lon]) / s;
  const ly = py(best[I.lat]) / s;
  tip.style.left = Math.min(Math.max(lx + 14, 4), b.width - 190) + "px";
  tip.style.top = Math.max(ly - 52, 4) + "px";
});
cv.addEventListener("pointerleave", () => (tip.dataset.on = "0"));

/* ── Estadísticas en vivo ──────────────────────────────────────────────── */
function dibujarVivo(filas: Fila[]) {
  const el = document.getElementById("live")!;
  const n = filas.length;
  if (!n) {
    el.innerHTML = `<div><span class="k">Sin datos</span><span class="v">—</span></div>`;
    return;
  }
  let se = 0, ae = 0, sesgo = 0;
  for (const r of filas) {
    const e = r[I.pred] - r[I.real];
    se += e * e; ae += Math.abs(e); sesgo += e;
  }
  el.innerHTML = [
    ["Distritos", n.toLocaleString("es-MX")],
    ["RMSE", f3(Math.sqrt(se / n))],
    ["Error medio abs.", usd(ae / n)],
    ["Sesgo", (sesgo > 0 ? "+" : "") + usd(sesgo / n)],
  ].map(([k, v]) => `<div><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
}

function dibujarLeyenda() {
  const el = document.getElementById("legend")!;
  el.innerHTML = estado.modo === "error"
    ? `<span class="ramp"><span class="lbl">subestima</span>` +
      `<i style="background:linear-gradient(90deg,${css("--cool")},${css("--mid")},${css("--warm")})"></i>` +
      `<span class="lbl">sobreestima</span></span><span class="hint">gris = sin error</span>`
    : `<span class="ramp"><span class="lbl">USD 15k</span>` +
      `<i style="background:linear-gradient(90deg,${css("--mid")},${css("--cool")})"></i>` +
      `<span class="lbl">USD 500k</span></span>`;
}

function dibujarTablaMapa(filas: Fila[]) {
  const top = [...filas]
    .sort((a, b) => Math.abs(b[I.pred] - b[I.real]) - Math.abs(a[I.pred] - a[I.real]))
    .slice(0, 12);
  document.getElementById("tblMap")!.innerHTML =
    `<table><caption class="eyebrow" style="text-align:left;padding-bottom:.5rem">Los 12 errores mayores del subconjunto</caption>` +
    `<thead><tr><th>Zona</th><th>Lat</th><th>Lon</th><th>Real</th><th>Predicho</th><th>Error</th></tr></thead><tbody>` +
    top.map((r) => {
      const e = r[I.pred] - r[I.real];
      return `<tr><td>${OCEAN[r[I.ocean]]}</td><td>${r[I.lat].toFixed(2)}</td><td>${r[I.lon].toFixed(2)}</td>` +
        `<td>${usd(r[I.real])}</td><td>${usd(r[I.pred])}</td><td>${(e > 0 ? "+" : "") + usd(e)}</td></tr>`;
    }).join("") + `</tbody></table>`;
}

/* ── Gráfica: intervalos de confianza ──────────────────────────────────── */
function dibujarIC() {
  const W = 720, H = 236, L = 152, R = 92, T = 26, B = 34;
  const x0 = 0.375, x1 = 0.485;
  const X = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const alto = (H - T - B) / MODELOS.length;
  const mejor = MODELOS[0];
  let s = "";
  for (let t = x0; t <= x1 + 1e-9; t += 0.02) {
    s += `<line class="grid" x1="${X(t).toFixed(1)}" y1="${T}" x2="${X(t).toFixed(1)}" y2="${H - B}"/>` +
      `<text x="${X(t).toFixed(1)}" y="${H - B + 15}" text-anchor="middle">${t.toFixed(2)}</text>`;
  }
  s += `<rect x="${X(mejor.lo)}" y="${T}" width="${X(mejor.hi) - X(mejor.lo)}" height="${H - T - B}" fill="var(--accent)" opacity=".07"/>`;
  MODELOS.forEach((m, i) => {
    const y = T + alto * i + alto / 2;
    const gana = i === 0;
    const col = gana ? "var(--accent)" : "var(--muted)";
    s += `<text class="lbl" x="${L - 12}" y="${y + 4}" text-anchor="end">${m.name}</text>` +
      `<line class="band" x1="${X(m.lo)}" y1="${y}" x2="${X(m.hi)}" y2="${y}" stroke="${col}" opacity="${gana ? .9 : .55}"/>` +
      `<line class="whisk" x1="${X(m.lo)}" y1="${y - 5}" x2="${X(m.lo)}" y2="${y + 5}" stroke="${col}"/>` +
      `<line class="whisk" x1="${X(m.hi)}" y1="${y - 5}" x2="${X(m.hi)}" y2="${y + 5}" stroke="${col}"/>` +
      `<circle class="halo" cx="${X(m.rmse)}" cy="${y}"/>` +
      `<circle class="dot" cx="${X(m.rmse)}" cy="${y}" fill="${col}"/>` +
      `<text class="val" x="${W - R + 12}" y="${y + 4}">${f3(m.rmse)}</text>`;
  });
  s += `<text x="${X(mejor.hi)}" y="${T - 9}" text-anchor="middle" fill="var(--accent)">límite del mejor</text>`;
  s += `<text x="${L}" y="${H - 4}" fill="var(--muted)">RMSE — menor es mejor</text>`;
  document.getElementById("ci")!.innerHTML = s;

  document.getElementById("tblCi")!.innerHTML =
    `<table><thead><tr><th>Modelo</th><th>RMSE</th><th>IC 95% inferior</th><th>IC 95% superior</th><th>¿Distinguible?</th></tr></thead><tbody>` +
    MODELOS.map((m, i) => {
      const solapa = !(m.lo > mejor.hi || m.hi < mejor.lo);
      return `<tr data-hi="${i === 0 ? 1 : 0}"><td>${m.name}</td><td>${f3(m.rmse)}</td><td>${f3(m.lo)}</td><td>${f3(m.hi)}</td>` +
        `<td>${i === 0 ? "—" : solapa ? "No, se traslapan" : "Sí, es peor"}</td></tr>`;
    }).join("") + `</tbody></table>`;
}

/* ── Gráfica: cuánto aportó cada decisión ──────────────────────────────── */
function dibujarAportes() {
  const W = 720, H = 352, L = 232, R = 78, T = 20, B = 36;
  const lo = -0.04, hi = 0.01;
  const X = (v: number) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  const alto = (H - T - B) / APORTES.length;
  let s = "";
  for (let t = -0.04; t <= 0.0101; t += 0.01) {
    s += `<line class="grid" x1="${X(t).toFixed(1)}" y1="${T}" x2="${X(t).toFixed(1)}" y2="${H - B}"/>` +
      `<text x="${X(t).toFixed(1)}" y="${H - B + 15}" text-anchor="middle">${t > 0 ? "+" : ""}${t.toFixed(3)}</text>`;
  }
  s += `<line class="axis" x1="${X(0)}" y1="${T}" x2="${X(0)}" y2="${H - B}"/>`;
  APORTES.forEach((d, i) => {
    const y = T + alto * i + alto / 2;
    const h = 11;
    const mejora = d.v < 0;
    const col = mejora ? "var(--cool)" : "var(--warm)";
    const xa = Math.min(X(0), X(d.v));
    const w = Math.abs(X(d.v) - X(0));
    // Los valores van en columna fija a la derecha: junto a la barra chocaban
    // con la etiqueta cuando la barra llegaba al margen.
    s += `<text class="lbl" x="${L - 12}" y="${y - 1}" text-anchor="end">${d.k}</text>` +
      `<text x="${L - 12}" y="${y + 12}" text-anchor="end">${d.m}</text>` +
      `<rect x="${xa}" y="${y - h / 2}" width="${Math.max(w, 2)}" height="${h}" fill="${col}" opacity=".8" rx="2"/>` +
      `<text class="val" x="${W - R + 14}" y="${y + 4}">${d.v > 0 ? "+" : ""}${d.v.toFixed(4)}</text>`;
  });
  s += `<text x="${L}" y="${H - 4}" fill="var(--muted)">cambio en RMSE — izquierda es mejora</text>`;
  document.getElementById("aportes")!.innerHTML = s;

  document.getElementById("tblAportes")!.innerHTML =
    `<table><thead><tr><th>Decisión</th><th>Modelo</th><th>Δ RMSE</th><th>Efecto</th></tr></thead><tbody>` +
    APORTES.map((d) => `<tr><td>${d.k}</td><td>${d.m}</td><td>${d.v > 0 ? "+" : ""}${d.v.toFixed(4)}</td>` +
      `<td>${d.v < 0 ? "Mejora" : "Empeora"}</td></tr>`).join("") + `</tbody></table>`;
}

/* ── Tablas estáticas ──────────────────────────────────────────────────── */
function dibujarCosto() {
  const maxMb = Math.max(...MODELOS.map((m) => m.mb));
  document.getElementById("tblCosto")!.innerHTML =
    `<table><thead><tr><th>Modelo</th><th>RMSE</th><th>Artefacto</th><th>Memoria</th><th>Latencia</th></tr></thead><tbody>` +
    MODELOS.map((m) => `<tr data-hi="${m.name === "xgboost" ? 1 : 0}"><td>${m.name}</td><td>${f3(m.rmse)}</td>` +
      `<td><span class="barcell"><i style="width:${Math.max(3, (m.mb / maxMb) * 90)}px"></i>${m.mb} MB</span></td>` +
      `<td>${m.ram ? m.ram.toLocaleString("en-US") + " MB" : "—"}</td>` +
      `<td>${m.ms ? m.ms.toFixed(2) + " ms" : "—"}</td></tr>`).join("") + `</tbody></table>`;
}

function dibujarTodos() {
  document.getElementById("tblTodos")!.innerHTML =
    `<table><thead><tr><th>Modelo</th><th>RMSE (validación)</th><th>RMSE (prueba)</th><th>R²</th></tr></thead><tbody>` +
    TODOS.map((m, i) => `<tr data-hi="${i === 0 ? 1 : 0}"><td>${m.name}</td><td>${m.cv ? f3(m.cv) : "—"}</td>` +
      `<td>${f3(m.rmse)}</td><td>${f3(m.r2)}</td></tr>`).join("") + `</tbody></table>`;
}

/* ── Filtros ───────────────────────────────────────────────────────────── */
function chip(texto: string, activo: boolean, alPulsar: (b: HTMLButtonElement) => void) {
  const b = document.createElement("button");
  b.className = "chip";
  b.type = "button";
  b.textContent = texto;
  b.setAttribute("aria-pressed", activo ? "true" : "false");
  b.addEventListener("click", () => alPulsar(b));
  return b;
}

function montarFiltros() {
  const fOcean = document.getElementById("fOcean")!;
  OCEAN.forEach((o, i) =>
    fOcean.appendChild(chip(o, false, (b) => {
      estado.ocean.has(i) ? estado.ocean.delete(i) : estado.ocean.add(i);
      b.setAttribute("aria-pressed", estado.ocean.has(i) ? "true" : "false");
      render();
    })),
  );

  const fInc = document.getElementById("fInc")!;
  ["1 · más bajo", "2", "3", "4", "5 · más alto"].forEach((lab, i) =>
    fInc.appendChild(chip(lab, false, (b) => {
      const c = i + 1;
      estado.inc.has(c) ? estado.inc.delete(c) : estado.inc.add(c);
      b.setAttribute("aria-pressed", estado.inc.has(c) ? "true" : "false");
      render();
    })),
  );

  const fModo = document.getElementById("fModo")!;
  ([["error", "Error"], ["real", "Precio real"], ["pred", "Precio predicho"]] as const)
    .forEach(([k, lab], i) =>
      fModo.appendChild(chip(lab, i === 0, (b) => {
        estado.modo = k;
        [...fModo.children].forEach((c) =>
          c.setAttribute("aria-pressed", c === b ? "true" : "false"));
        render();
      })),
    );
}

function montarToggles() {
  ([["tblMapBtn", "tblMap"], ["tblCiBtn", "tblCi"], ["tblAportesBtn", "tblAportes"]] as const)
    .forEach(([bid, tid]) => {
      const btn = document.getElementById(bid)!;
      const tbl = document.getElementById(tid) as HTMLElement;
      btn.addEventListener("click", () => {
        const abierto = btn.getAttribute("aria-pressed") === "true";
        btn.setAttribute("aria-pressed", abierto ? "false" : "true");
        btn.textContent = abierto ? "Ver tabla" : "Ocultar tabla";
        tbl.hidden = abierto;
      });
    });
}

function render() {
  visibles = ROWS.filter(
    (r) =>
      (estado.ocean.size === 0 || estado.ocean.has(r[I.ocean])) &&
      (estado.inc.size === 0 || estado.inc.has(r[I.inc])),
  );
  dibujarMapa(visibles);
  dibujarVivo(visibles);
  dibujarLeyenda();
  dibujarTablaMapa(visibles);
}

/* ── Arranque ──────────────────────────────────────────────────────────── */
const datos = await fetch("/data/ch02.json").then((r) => r.json());
ROWS = datos.rows;
OCEAN = datos.ocean;

montarFiltros();
montarToggles();
render();
dibujarIC();
dibujarAportes();
dibujarCosto();
dibujarTodos();

// Los colores salen de variables CSS, así que al cambiar de tema hay que
// repintar todo lo que se dibuja en JS.
const repintar = () => { render(); dibujarIC(); dibujarAportes(); };
document.addEventListener("tema-cambiado", repintar);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", repintar);

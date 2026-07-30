/**
 * Las gráficas y tablas de resultados: /ch02/california-housing
 *
 * Dos fuentes, y conviene no confundirlas. El RMSE con su intervalo y los
 * aportes vienen de `make evaluate` y `make analysis`, y están escritos aquí
 * porque son la conclusión del capítulo y no cambian. El costo de servir se lee
 * de /data/ch02-costo.json, que produce `make cost`.
 *
 * El mapa interactivo vive aparte, en ch02-mapa.ts, porque es lo único que
 * necesita bajarse las 4,128 filas.
 */

const f3 = (v: number) => v.toFixed(3);

/** RMSE en prueba y su intervalo al 95%, de `make evaluate`. */
const MODELOS = [
  { name: "extra_trees", rmse: 0.4063, lo: 0.3844, hi: 0.4271 },
  { name: "random_forest", rmse: 0.4235, lo: 0.4015, hi: 0.4443 },
  { name: "xgboost", rmse: 0.4291, lo: 0.4071, hi: 0.4500 },
  { name: "gradient_boosting", rmse: 0.4514, lo: 0.4292, hi: 0.4726 },
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
  s += `<text x="${L}" y="${H - 4}" fill="var(--muted)">RMSE · menor es mejor</text>`;
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
  // La columna de valores no se explica sola: sin encabezado no se sabe si son
  // RMSE o el cambio en RMSE.
  s += `<text class="hdr" x="${W - R + 14}" y="13">Δ RMSE</text>`;
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
  s += `<text x="${L}" y="${H - 4}" fill="var(--muted)">cambio en RMSE · izquierda es mejora</text>`;
  document.getElementById("aportes")!.innerHTML = s;

  document.getElementById("tblAportes")!.innerHTML =
    `<table><thead><tr><th>Decisión</th><th>Modelo</th><th>Δ RMSE</th><th>Efecto</th></tr></thead><tbody>` +
    APORTES.map((d) => `<tr><td>${d.k}</td><td>${d.m}</td><td>${d.v > 0 ? "+" : ""}${d.v.toFixed(4)}</td>` +
      `<td>${d.v < 0 ? "Mejora" : "Empeora"}</td></tr>`).join("") + `</tbody></table>`;
}

/* ── Tablas ────────────────────────────────────────────────────────────── */
/**
 * El costo de servir sale medido de `make cost`, no escrito a mano. Antes vivía
 * aquí como literal y dos de los cuatro modelos no tenían dato porque nadie los
 * había cronometrado, sin que la tabla lo dijera.
 */
interface Costo {
  modelo: string;
  rmse: number;
  artefacto_mb: number;
  modelo_mb: number;
  rss_mb: number;
  latencia_ms: number;
}

async function dibujarCosto() {
  const { modelos } = (await fetch("/data/ch02-costo.json").then((r) => r.json())) as {
    modelos: Costo[];
  };
  const maxMb = Math.max(...modelos.map((m) => m.artefacto_mb));
  const mb = (v: number) => Math.round(v).toLocaleString("en-US") + " MB";

  document.getElementById("tblCosto")!.innerHTML =
    `<table><thead><tr><th>Modelo</th><th>RMSE</th><th>Artefacto</th>` +
    `<th>Modelo en RAM</th><th>Proceso</th><th>Latencia</th></tr></thead><tbody>` +
    modelos.map((m) =>
      `<tr data-hi="${m.modelo === "xgboost" ? 1 : 0}"><td>${m.modelo}</td><td>${f3(m.rmse)}</td>` +
      `<td><span class="barcell"><i style="width:${Math.max(3, (m.artefacto_mb / maxMb) * 90)}px"></i>` +
      `${m.artefacto_mb.toFixed(1)} MB</span></td>` +
      `<td>${mb(m.modelo_mb)}</td><td>${mb(m.rss_mb)}</td>` +
      `<td>${m.latencia_ms.toFixed(2)} ms</td></tr>`,
    ).join("") + `</tbody></table>`;
}

function dibujarTodos() {
  document.getElementById("tblTodos")!.innerHTML =
    `<table><thead><tr><th>Modelo</th><th>RMSE (validación)</th><th>RMSE (prueba)</th><th>R²</th></tr></thead><tbody>` +
    TODOS.map((m, i) => `<tr data-hi="${i === 0 ? 1 : 0}"><td>${m.name}</td><td>${m.cv ? f3(m.cv) : "—"}</td>` +
      `<td>${f3(m.rmse)}</td><td>${f3(m.r2)}</td></tr>`).join("") + `</tbody></table>`;
}

/* ── Toggles de tabla ──────────────────────────────────────────────────── */
([["tblCiBtn", "tblCi"], ["tblAportesBtn", "tblAportes"]] as const).forEach(([bid, tid]) => {
  const btn = document.getElementById(bid)!;
  const tbl = document.getElementById(tid) as HTMLElement;
  btn.addEventListener("click", () => {
    const abierto = btn.getAttribute("aria-pressed") === "true";
    btn.setAttribute("aria-pressed", abierto ? "false" : "true");
    btn.textContent = abierto ? "Ver tabla" : "Ocultar tabla";
    tbl.hidden = abierto;
  });
});

dibujarIC();
dibujarAportes();
dibujarTodos();
await dibujarCosto();

// Sin repintado al cambiar de tema, a diferencia del mapa. Estas gráficas son
// SVG y sus colores van escritos como `var(--cool)` dentro del atributo, así
// que los resuelve el CSS en cada pintado. El canvas del mapa sí guarda píxeles
// ya resueltos, y por eso allá hace falta.

"""
Qué cuesta llevar el modelo al navegador, medido en KB y en RMSE

El modelo que la demo usa hoy pesa 286 KB comprimidos y da 0.4291 de RMSE.
Cabe en el presupuesto de 300 KB, pero sin margen, así que antes de construir
la interfaz conviene saber si había algo mejor y no se buscó.

Se miden dos ejes independientes:

  **Codificación**: el mismo modelo escrito de otra forma. No cambia una sola
  predicción, solo el número de bytes. JSON anidado, JSON plano, y binario con
  arrays separados por tipo, cada uno con gzip y con brotli, que es lo que
  Vercel sirve de verdad.

  **Modelo**: árboles más cortos o menos árboles. Cambia la predicción, así que
  cada candidato se vuelve a medir contra el test set entero.

El criterio de honestidad viene del capítulo: el intervalo del 95% del campeón
es [0.3844, 0.4271]. Dentro, la demo predice lo mismo que el modelo desplegado.
Fuera, la página tiene que decirlo.

Uso:
    python src/export_study.py
"""
import gzip
import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from preprocessing import PROCESSED_DIR, SKOPS_TRUSTED, build_pipeline

REPORTS_DIR = Path("reports")
OUT = REPORTS_DIR / "export_study.json"
RUN_ID = "ca09f3c690ad43528fbb6d5f65bef174"
IC_CAMPEON = (0.3844, 0.4271)
PRESUPUESTO_KB = 300

# Los del modelo afinado, para que los candidatos solo cambien lo que se estudia.
BASE = dict(subsample=0.8, reg_lambda=2.0, reg_alpha=0.1, learning_rate=0.05,
            colsample_bytree=0.7, random_state=42, n_jobs=2)


def kb_gzip(datos: bytes) -> float:
    return len(gzip.compress(datos, 9)) / 1024


def kb_brotli(datos: bytes) -> float:
    """Brotli mide lo que el navegador descarga de verdad: Vercel lo sirve así.

    Se usa el binario del sistema porque el paquete de Python no está instalado
    y no vale la pena añadir una dependencia para una medición.
    """
    if not shutil.which("brotli"):
        return float("nan")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(datos)
        origen = Path(f.name)
    destino = origen.with_suffix(".br")
    destino.unlink(missing_ok=True)
    subprocess.run(["brotli", "-q", "11", "-o", str(destino), str(origen)], check=True)
    tam = destino.stat().st_size / 1024
    origen.unlink()
    destino.unlink()
    return tam


def compactar(nodo: dict) -> dict:
    if "leaf" in nodo:
        return {"v": round(nodo["leaf"], 6)}
    hijos = {h["nodeid"]: h for h in nodo["children"]}
    return {"f": int(nodo["split"].removeprefix("f")), "u": nodo["split_condition"],
            "i": compactar(hijos[nodo["yes"]]), "d": compactar(hijos[nodo["no"]])}


def aplanar(arbol: dict) -> list:
    """El mismo árbol como lista de nodos: (columna, umbral, izquierda, derecha)."""
    nodos = []

    def visitar(n: dict) -> int:
        idx = len(nodos)
        nodos.append(None)
        if "v" in n:
            nodos[idx] = (-1, n["v"], -1, -1)
        else:
            i, d = visitar(n["i"]), visitar(n["d"])
            nodos[idx] = (n["f"], n["u"], i, d)
        return idx

    visitar(arbol)
    return nodos


def binario(arboles: list) -> bytes:
    """Arrays separados por tipo, que es lo que comprime bien.

    Juntar en cada nodo un entero, un flotante y dos índices mezcla rangos y
    entropías distintas. Separados, la columna de índices queda casi constante
    y el compresor la aprovecha. Los umbrales van en float32 porque es la
    precisión con la que XGBoost compara.
    """
    planos = [aplanar(a) for a in arboles]
    cortes, cols, umbrales, izq, der = [0], [], [], [], []
    for nodos in planos:
        for f, u, i, d in nodos:
            cols.append(f + 1)            # 0 marca hoja, así cabe en un byte
            umbrales.append(u)
            izq.append(max(i, 0))
            der.append(max(d, 0))
        cortes.append(len(cols))
    partes = [
        struct.pack(f"<{len(cortes)}I", *cortes),
        bytes(cols),
        np.asarray(umbrales, dtype="<f4").tobytes(),
        np.asarray(izq, dtype="<u2").tobytes(),
        np.asarray(der, dtype="<u2").tobytes(),
    ]
    return b"".join(partes)


def rmse_de(modelo, X_test, y_test) -> float:
    return float(np.sqrt(mean_squared_error(y_test, modelo.predict(X_test))))


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    import mlflow.artifacts
    import skops.io as sio

    with tempfile.TemporaryDirectory() as tmp:
        ruta = mlflow.artifacts.download_artifacts(
            run_id=RUN_ID, artifact_path="model", dst_path=tmp)
        pipe = sio.load(next(Path(ruta).rglob("*.skops")), trusted=SKOPS_TRUSTED)

    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").values.ravel()
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").values.ravel()

    booster = pipe.named_steps["model"].get_booster()
    arboles = [compactar(json.loads(a)) for a in booster.get_dump(dump_format="json")]
    prep_json = json.dumps({"prep": "..."})   # el preprocesamiento son 73 números

    # ── Eje 1: la misma predicción, escrita de otra forma ────────────────────
    print("[estudio] Codificaciones del mismo modelo (predicción idéntica)")
    anidado = json.dumps(arboles, separators=(",", ":")).encode()
    plano = json.dumps([aplanar(a) for a in arboles], separators=(",", ":")).encode()
    bin_ = binario(arboles)
    codificaciones = []
    for nombre, datos in (("JSON anidado", anidado), ("JSON plano", plano),
                          ("binario por columnas", bin_)):
        fila = {"codificacion": nombre, "kb_crudo": len(datos) / 1024,
                "kb_gzip": kb_gzip(datos), "kb_brotli": kb_brotli(datos)}
        codificaciones.append(fila)
        print(f"  {nombre:22s} {fila['kb_crudo']:8.1f} KB  "
              f"{fila['kb_gzip']:7.1f} gzip  {fila['kb_brotli']:7.1f} brotli")

    # ── Eje 2: modelos más pequeños, que sí cambian la predicción ────────────
    print("\n[estudio] Candidatos, con su RMSE en el test set")
    candidatos = []

    def medir(nombre: str, modelo, arboles_json: list) -> None:
        datos = json.dumps(arboles_json, separators=(",", ":")).encode()
        rmse = rmse_de(modelo, X_test, y_test)
        fila = {"candidato": nombre, "rmse": rmse, "arboles": len(arboles_json),
                "kb_gzip": kb_gzip(datos), "kb_brotli": kb_brotli(datos),
                "dentro_del_ic": IC_CAMPEON[0] <= rmse <= IC_CAMPEON[1]}
        candidatos.append(fila)
        print(f"  {nombre:34s} RMSE {rmse:.4f}  {fila['kb_gzip']:6.1f} gzip  "
              f"{fila['kb_brotli']:6.1f} brotli"
              f"{'  dentro del IC' if fila['dentro_del_ic'] else ''}")

    medir("xgboost_tuned (el actual)", pipe, arboles)

    for profundidad, n in ((7, 150), (6, 300), (5, 300), (4, 300), (4, 500), (3, 500)):
        modelo = XGBRegressor(max_depth=profundidad, n_estimators=n, **BASE)
        p = build_pipeline(modelo, X_train)
        p.fit(X_train, y_train)
        dump = [compactar(json.loads(a))
                for a in p.named_steps["model"].get_booster().get_dump(dump_format="json")]
        medir(f"profundidad {profundidad}, {n} árboles", p, dump)

    mejor = min((c for c in candidatos if c["kb_brotli"] <= PRESUPUESTO_KB),
                key=lambda c: c["rmse"], default=None)
    print(f"\n[estudio] Mejor dentro de {PRESUPUESTO_KB} KB brotli: "
          f"{mejor['candidato']} → {mejor['rmse']:.4f}" if mejor else "\nNinguno cabe")

    OUT.write_text(json.dumps({
        "ic_campeon": IC_CAMPEON, "presupuesto_kb": PRESUPUESTO_KB,
        "codificaciones": codificaciones, "candidatos": candidatos,
        "nota_prep": "El preprocesamiento son 73 números, despreciable frente a los árboles",
    }, indent=2))
    print(f"[estudio] Guardado → {OUT}")


if __name__ == "__main__":
    main()

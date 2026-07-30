"""
Cuánto cuesta servir cada modelo: tamaño, memoria y latencia.

Existe porque las cifras de la tabla "Costo de servir cada modelo" del sitio
estaban escritas a mano, sin script detrás, y dos de los cuatro modelos no
tenían dato. Un número que no se puede volver a medir no vale.

Qué mide, con la definición escrita para que nadie tenga que adivinarla después:

  artefacto_mb   suma de bytes del artefacto en disco. Es lo que se despliega.
  base_mb        RSS del proceso ya con numpy, pandas y sklearn importados,
                 antes de cargar el modelo. El suelo que paga cualquier modelo.
  modelo_mb      RSS después de cargar, menos base_mb. Lo que cuesta el modelo
                 en sí, que es la cifra comparable entre modelos.
  rss_mb         RSS total después de cargar. Lo que hay que provisionar de
                 verdad en un servidor.
  latencia_ms    mediana de 20 predicciones de una sola fila, tras 3 de
                 calentamiento. Individuales a propósito: por lotes sale otra
                 cifra, mucho mejor, y no es la que sufre una API.

Todos los MB son decimales (bytes / 10^6), que es la convención de los
fabricantes de disco y la que ya usaba la tabla del sitio. Mezclarla con MiB
(bytes / 2^20) inventa una diferencia del 4.9% que parece un error de medición
y no lo es.

Cada modelo se mide en su propio subproceso. Con `extra_trees` pasando de 1 GB,
medirlos en el mismo proceso dejaría al siguiente contando memoria que no es
suya, y en una máquina de 16 GB acumularlos es cómo se la tumba.

Uso:
    python src/cost.py --check     # proyecta el pico y no carga nada
    python src/cost.py             # mide y escribe reports/cost.json
    # o: make cost / make cost-check
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import psutil

REPORTS_DIR = Path("reports")
SALIDA = REPORTS_DIR / "cost.json"
PROCESSED_DIR = Path("data/processed")

# La misma medición va a dos sitios: reports/ para comparar entre versiones, y
# web/public/data/ para que la tabla del sitio la lea en vez de repetirla a mano.
# Ese fue el problema que este script existe para arreglar.
WEB = Path(__file__).resolve().parents[2] / "web" / "public" / "data" / "ch02-costo.json"

# Los cuatro de la tabla del sitio: los dos candidatos a desplegar y los dos que
# quedaron cerca. Los cinco restantes no compiten por RMSE, así que su costo no
# cambia ninguna decisión.
MODELOS = ["extra_trees", "random_forest", "xgboost", "gradient_boosting"]

REPETICIONES = 20
CALENTAMIENTO = 3

MB = 1_000_000  # decimal, ver el docstring

# El artefacto en disco no es lo que ocupa cargado: los bosques de sklearn se
# despliegan en objetos con más overhead. Medido sobre extra_trees, la relación
# ronda 1.85; se proyecta con 2.2 para que el aviso llegue antes de tiempo y no
# después. Sirve para abortar, no para reportar.
FACTOR_CARGA = 2.2
LIMITE_RAM = 0.45


def _mejor_run(nombre: str):
    """El run con menor RMSE de ese modelo, y el artefacto que produjo.

    Se busca en vez de pedirse por parámetro para que la tabla del sitio no
    dependa de que alguien copie el run_id correcto del mlflow ui.
    """
    import mlflow

    runs = mlflow.search_runs(
        experiment_names=["california-housing"],
        filter_string="attributes.status = 'FINISHED'",
    )
    # Los candidatos de las búsquedas son runs anidados: compiten en validación,
    # no en prueba, y su métrica no es comparable con la de los padres.
    runs = runs[runs["tags.mlflow.parentRunId"].isna()]
    runs = runs[runs["tags.mlflow.runName"].str.startswith(nombre, na=False)]
    runs = runs.dropna(subset=["metrics.rmse"]).sort_values("metrics.rmse")
    if runs.empty:
        raise SystemExit(f"sin runs terminados para {nombre}")
    mejor = runs.iloc[0]

    modelos = mlflow.search_logged_models(
        experiment_ids=[mejor["experiment_id"]], output_format="pandas"
    )
    propio = modelos[modelos["source_run_id"] == mejor["run_id"]]
    if propio.empty:
        raise SystemExit(f"el run {mejor['run_id']} de {nombre} no dejó modelo")

    return {
        "run_id": mejor["run_id"],
        "run_name": mejor["tags.mlflow.runName"],
        "rmse": float(mejor["metrics.rmse"]),
        "model_id": propio.iloc[0]["model_id"],
        "artifact_location": propio.iloc[0]["artifact_location"],
    }


def _bytes_en(ruta: str) -> int:
    p = Path(ruta.removeprefix("file://"))
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / MB


def medir_uno(nombre: str) -> dict:
    """Mide un modelo. Corre en su propio proceso, nunca junto a otro."""
    import mlflow
    import pandas as pd

    info = _mejor_run(nombre)
    artefacto_mb = _bytes_en(info["artifact_location"]) / MB

    # Una sola fila cruda, con las columnas originales: el pipeline hace el resto.
    fila = pd.read_parquet(PROCESSED_DIR / "X_test.parquet").head(1)

    base_mb = _rss_mb()
    modelo = mlflow.sklearn.load_model(f"models:/{info['model_id']}")
    rss_mb = _rss_mb()

    for _ in range(CALENTAMIENTO):
        modelo.predict(fila)

    tiempos = []
    for _ in range(REPETICIONES):
        t0 = time.perf_counter()
        modelo.predict(fila)
        tiempos.append((time.perf_counter() - t0) * 1000)

    return {
        "modelo": nombre,
        "run_id": info["run_id"],
        "run_name": info["run_name"],
        "rmse": round(info["rmse"], 4),
        "artefacto_mb": round(artefacto_mb, 1),
        "base_mb": round(base_mb, 0),
        "modelo_mb": round(rss_mb - base_mb, 0),
        "rss_mb": round(rss_mb, 0),
        "latencia_ms": round(median(tiempos), 2),
        "latencia_min_ms": round(min(tiempos), 2),
        "latencia_max_ms": round(max(tiempos), 2),
        "repeticiones": REPETICIONES,
    }


def proyectar() -> bool:
    """Proyecta el pico sin cargar nada. Devuelve si cabe."""
    total_gb = psutil.virtual_memory().total / 1024**3  # GiB, solo para el aviso
    techo_mb = psutil.virtual_memory().total * LIMITE_RAM / MB
    print(f"RAM total {total_gb:.1f} GB · techo al {LIMITE_RAM:.0%} = {techo_mb:,.0f} MB")
    print(f"\n{'modelo':>18} {'artefacto':>11} {'pico proyectado':>17}")

    peor = 0.0
    for nombre in MODELOS:
        info = _mejor_run(nombre)
        art = _bytes_en(info["artifact_location"]) / MB
        # Un proceso por modelo, así que el pico es el del modelo más grande,
        # no la suma. De ahí que aquí se mire el máximo.
        pico = art * FACTOR_CARGA + 400  # 400 MB de suelo: intérprete y librerías
        peor = max(peor, pico)
        print(f"{nombre:>18} {art:>8,.1f} MB {pico:>14,.0f} MB")

    print(f"\npico del peor caso: {peor:,.0f} MB de un techo de {techo_mb:,.0f} MB")
    if peor > techo_mb:
        print("NO CABE. Abortado sin cargar nada.")
        return False
    print(f"Cabe, con {techo_mb - peor:,.0f} MB de margen.")
    return True


def medir_todos() -> None:
    if not proyectar():
        raise SystemExit(1)

    print("\nMidiendo, un subproceso por modelo.\n")
    filas = []
    for nombre in MODELOS:
        # `--one` se invoca a sí mismo: al terminar el subproceso el sistema
        # recupera la memoria del modelo entera, sin depender del recolector.
        salida = subprocess.run(
            [sys.executable, __file__, "--one", nombre],
            capture_output=True,
            text=True,
            check=False,
        )
        if salida.returncode != 0:
            print(salida.stderr, file=sys.stderr)
            raise SystemExit(f"falló la medición de {nombre}")
        fila = json.loads(salida.stdout)
        filas.append(fila)
        print(
            f"{fila['modelo']:>18}  {fila['artefacto_mb']:>7,.1f} MB artefacto  "
            f"{fila['modelo_mb']:>6,.0f} MB modelo  {fila['rss_mb']:>6,.0f} MB RSS  "
            f"{fila['latencia_ms']:>6.2f} ms"
        )

    reporte = json.dumps(
        {
            "repeticiones": REPETICIONES,
            "calentamiento": CALENTAMIENTO,
            "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "modelos": filas,
        },
        ensure_ascii=False,
        indent=1,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(reporte, encoding="utf-8")
    print(f"\n✓ {SALIDA}")

    if WEB.parent.exists():
        WEB.write_text(reporte, encoding="utf-8")
        print(f"✓ {WEB}")
    else:
        print(f"[cost] sin {WEB.parent}, no escribo para el sitio")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="proyecta el pico y no carga nada")
    ap.add_argument("--one", metavar="MODELO", help="uso interno: mide uno y escribe JSON")
    args = ap.parse_args()

    if args.one:
        print(json.dumps(medir_uno(args.one), ensure_ascii=False))
    elif args.check:
        raise SystemExit(0 if proyectar() else 1)
    else:
        medir_todos()


if __name__ == "__main__":
    main()

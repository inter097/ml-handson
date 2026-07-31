"""
El modelo, exportado a JSON para que prediga en el navegador

La demo del capítulo necesita predecir en el cliente, así que el modelo tiene
que viajar como JSON. El campeón no puede: `extra_trees_tuned` ocupa 703 MB de
artefacto. XGBoost afinado ocupa 2.4 MB y su RMSE queda a 0.023 del campeón,
así que es el único candidato realista.

Medido antes de escribir la interfaz: los 300 árboles compactados ocupan 186 KB
servidos con brotli, que es lo que responde Vercel, dentro del presupuesto de
300 KB. `export_study.py` comprobó que ninguna otra codificación ni ningún
modelo más pequeño mejora ese punto.

Lo que exporta, que es todo lo que hace falta para reproducir la predicción:

  medianas del imputador · media y escala del escalador
  centros de los barrios y su gamma · categorías del one-hot
  los 300 árboles, con nodos de una letra

Y dos cosas más que la demo necesita para ser honesta:

  - **Paridad**: cinco filas con su predicción de Python. Si el TypeScript no
    da el mismo número, la demo está mintiendo y hay que verlo enseguida.
  - **Medianas por defecto**: la demo enseña cuatro controles, no seis, y las
    dos que oculta se fijan en su mediana. Cuánto cuesta eso está medido aquí,
    no supuesto: esconder AveRooms costaba 0.0868, así que pasó a ser visible.

Uso:
    python src/export_demo.py
"""
import gzip
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from preprocessing import PROCESSED_DIR, SKOPS_TRUSTED

REPORTS_DIR = Path("reports")
SITE_DIR = Path("../web/public/data")
REPORTE = REPORTS_DIR / "demo_export.json"
MODELO = SITE_DIR / "ch02-modelo.json"

RUN_ID = "ca09f3c690ad43528fbb6d5f65bef174"   # xgboost_tuned
IC_CAMPEON = (0.3844, 0.4271)                 # extra_trees_tuned, del reporte de evaluación
PRESUPUESTO_KB = 300

# Lo que la demo pregunta y lo que esconde. El criterio no es la importancia
# sino si alguien sabe contestarlo mirando un barrio.
VISIBLES = ["MedInc", "AveOccup", "AveRooms", "HouseAge"]
OCULTAS = ["AveBedrms", "Population"]


def cargar_pipeline(run_id: str):
    import tempfile

    import mlflow.artifacts
    import skops.io as sio

    with tempfile.TemporaryDirectory() as tmp:
        ruta = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="model", dst_path=tmp
        )
        return sio.load(next(Path(ruta).rglob("*.skops")), trusted=SKOPS_TRUSTED)


def compactar(nodo: dict) -> dict:
    """Nodo con nombres de una letra: f=columna, u=umbral, i/d=hijos, v=hoja.

    Los umbrales van **sin redondear**. Redondearlos a cinco decimales parecía
    inofensivo y desviaba la predicción hasta 0.35: un valor que cae junto a un
    corte se va por la otra rama, y con 300 árboles esos desvíos se acumulan.
    Las hojas sí se redondean a seis decimales, porque ahí el error se suma en
    lugar de propagarse por el árbol.

    La rama de valores ausentes no se exporta: la imputación ocurre antes,
    dentro del preprocesamiento, así que al árbol nunca le llega un NaN.
    """
    if "leaf" in nodo:
        return {"v": round(nodo["leaf"], 6)}
    hijos = {h["nodeid"]: h for h in nodo["children"]}
    return {
        "f": int(nodo["split"].removeprefix("f")),
        "u": nodo["split_condition"],
        "i": compactar(hijos[nodo["yes"]]),
        "d": compactar(hijos[nodo["no"]]),
    }


def tamanos(objeto) -> tuple:
    """Bruto, gzip y brotli, en KB.

    El número que importa es el de brotli: comprobado contra el sitio en
    producción, Vercel responde `content-encoding: br` para los JSON. Medir
    solo gzip daba 286 KB y hacía parecer justo un presupuesto que estaba al
    62%.
    """
    crudo = json.dumps(objeto, separators=(",", ":")).encode()
    br = float("nan")
    if shutil.which("brotli"):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(crudo)
            origen = Path(f.name)
        destino = origen.with_suffix(".br")
        destino.unlink(missing_ok=True)
        subprocess.run(["brotli", "-q", "11", "-o", str(destino), str(origen)], check=True)
        br = destino.stat().st_size / 1024
        origen.unlink()
        destino.unlink()
    return len(crudo) / 1024, len(gzip.compress(crudo, 9)) / 1024, br


def con_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Los tres cocientes de features.py, recalculados igual que allí."""
    df = df.copy()
    df["rooms_per_household"] = df["AveRooms"] / df["AveOccup"]
    df["bedrooms_ratio"] = df["AveBedrms"] / df["AveRooms"]
    df["population_per_household"] = df["Population"] / df["AveOccup"]
    return df


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    pipe = cargar_pipeline(RUN_ID)
    prep = pipe.named_steps["prep"]
    booster = pipe.named_steps["model"].get_booster()

    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").values.ravel()

    rmse_completo = float(np.sqrt(mean_squared_error(y_test, pipe.predict(X_test))))
    print(f"[export] RMSE con todas las variables: {rmse_completo:.4f}")

    # ── Cuánto cuesta esconder tres controles ────────────────────────────────
    # Se sustituyen por su mediana de entrenamiento y se recalculan los
    # cocientes, que es lo que hará la demo cuando nadie abra «avanzado».
    medianas = X_train[OCULTAS].median()
    X_medianas = X_test.drop(columns=[c for c in X_test.columns
                                      if c in ("rooms_per_household", "bedrooms_ratio",
                                               "population_per_household")])
    for col in OCULTAS:
        X_medianas[col] = medianas[col]
    X_medianas = con_ratios(X_medianas)[X_test.columns]
    rmse_medianas = float(np.sqrt(mean_squared_error(y_test, pipe.predict(X_medianas))))
    print(f"[export] RMSE fijando {', '.join(OCULTAS)} en su mediana: "
          f"{rmse_medianas:.4f}  ({rmse_medianas - rmse_completo:+.4f})")

    # ── El preprocesamiento, parámetro a parámetro ───────────────────────────
    num = prep.named_transformers_["num"]
    geo = prep.named_transformers_["geo"]
    cat = prep.named_transformers_["cat"]
    columnas_num = list(prep.transformers_[0][2])

    # Sin redondear, por la misma razón que los umbrales: un error de 1e-5 en
    # una media desplaza la columna estandarizada lo justo para que una fila
    # cruce un corte y el árbol la mande por la otra rama. Son 73 números, así
    # que la precisión completa no pesa nada frente a los 300 árboles.
    spec = {
        "columnas_numericas": columnas_num,
        "medianas_imputador": [float(v) for v in num.named_steps["imputer"].statistics_],
        "media_escalador": [float(v) for v in num.named_steps["scaler"].mean_],
        "escala_escalador": [float(v) for v in num.named_steps["scaler"].scale_],
        "centros_barrios": [[float(c) for c in fila]
                            for fila in geo.kmeans_.cluster_centers_],
        "gamma_barrios": float(geo.gamma),
        "categorias": [str(c) for c in cat.categories_[0]],
        "salida": list(prep.get_feature_names_out()),
    }

    # ── Rangos de los controles ──────────────────────────────────────────────
    # Rango completo donde el máximo es creíble, y percentiles 1 y 99 donde no.
    # AveOccup llega a 502 personas por hogar y AveRooms a 141 habitaciones:
    # son cuarteles y residencias, y un control con ese tope deja el 99% de los
    # casos apelotonado en el primer píxel del deslizador.
    COMPLETAS = {"MedInc", "HouseAge"}
    rangos = {
        col: {
            "min": round(float(X_train[col].min() if col in COMPLETAS
                               else X_train[col].quantile(0.01)), 3),
            "max": round(float(X_train[col].max() if col in COMPLETAS
                               else X_train[col].quantile(0.99)), 3),
            "inicio": round(float(X_train[col].median()), 3),
        }
        for col in VISIBLES
    }

    # ── Paridad: cinco filas con su predicción de Python ─────────────────────
    muestra = X_test.head(5)
    paridad = [
        {"entrada": {k: (None if pd.isna(v) else (float(v) if not isinstance(v, str) else v))
                     for k, v in fila.items()},
         "prediccion": round(float(p), 6)}
        for (_, fila), p in zip(muestra.iterrows(), pipe.predict(muestra))
    ]

    # El punto de partida del que cuelgan todos los árboles. `booster.attr` lo
    # devuelve como None: vive en la configuración, viene entre corchetes, y en
    # regresión es la media del objetivo, no el 0.5 que suele suponerse.
    base_score = float(json.loads(booster.save_config())
                       ["learner"]["learner_model_param"]["base_score"].strip("[]"))
    print(f"[export] base_score: {base_score:.6f}")

    arboles = [compactar(json.loads(a)) for a in booster.get_dump(dump_format="json")]
    modelo = {
        "generado_desde": RUN_ID,
        "modelo": "xgboost_tuned",
        "rmse": round(rmse_completo, 4),
        "ic_campeon": IC_CAMPEON,
        "base_score": base_score,
        "prep": spec,
        "medianas_ocultas": {c: round(float(medianas[c]), 6) for c in OCULTAS},
        "rangos": rangos,
        "penalizacion_ocultas": round(rmse_medianas - rmse_completo, 4),
        "arboles": arboles,
        "paridad": paridad,
    }

    crudo, gz, br = tamanos(modelo)
    MODELO.write_text(json.dumps(modelo, separators=(",", ":")))
    print(f"[export] {len(arboles)} árboles, {len(spec['salida'])} columnas")
    print(f"[export] {crudo:.1f} KB en bruto · {gz:.1f} KB gzip · {br:.1f} KB brotli "
          f"({'cabe' if br <= PRESUPUESTO_KB else 'NO CABE'} en {PRESUPUESTO_KB} KB, "
          f"y brotli es lo que sirve Vercel)")
    print(f"[export] Guardado → {MODELO}")

    REPORTE.write_text(json.dumps({
        "run_id": RUN_ID,
        "rmse_completo": rmse_completo,
        "rmse_con_medianas": rmse_medianas,
        "penalizacion": rmse_medianas - rmse_completo,
        "variables_visibles": VISIBLES,
        "variables_ocultas": OCULTAS,
        "medianas_ocultas": {c: float(medianas[c]) for c in OCULTAS},
        "arboles": len(arboles),
        "kb_crudo": crudo, "kb_gzip": gz, "kb_brotli": br,
        "presupuesto_kb": PRESUPUESTO_KB,
        "ic_campeon": IC_CAMPEON,
        "dentro_del_ic": IC_CAMPEON[0] <= rmse_completo <= IC_CAMPEON[1],
    }, indent=2))
    print(f"[export] Reporte → {REPORTE}")


if __name__ == "__main__":
    main()

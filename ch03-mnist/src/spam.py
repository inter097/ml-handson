"""
Ejercicio 4 — Filtro de spam
Referencia: Géron cap. 3, ejercicio 4 ("Build a spam classifier")

Corpus público de Apache SpamAssassin: correos reales de 2002, con cabeceras,
HTML, adjuntos y codificaciones rotas. No es un dataset limpio — es lo que
llegó a una bandeja de entrada.

El trabajo real está antes del modelo. Un correo no es un vector; hay que
decidir qué se conserva y qué se tira, y cada decisión es una hipótesis sobre
qué distingue el spam:

  Cuerpo en texto plano   se prefiere sobre el HTML cuando el correo trae los
                          dos; si solo hay HTML, se quitan las etiquetas
  Asunto                  se antepone al cuerpo — suele cargar la señal más fuerte
  URLs, correos, números  se reemplazan por marcadores. Que aparezca *una* URL
                          importa; cuál exactamente, no. Sin esto el vocabulario
                          explota con enlaces que salen una sola vez
  Minúsculas              GRATIS y gratis son la misma palabra

Por qué la precisión importa más que la exhaustividad aquí: mandar un correo
legítimo a la carpeta de spam es mucho peor que dejar pasar publicidad. El
umbral se ajusta en esa dirección.

Uso:
    python src/spam.py
    # o: make spam
"""
import email
import email.policy
import re
import tarfile
import urllib.request
from pathlib import Path

import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from baseline import EXPERIMENT, SEED

RAW_DIR = Path("data/raw/spamassassin")
BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
CORPUS = {"ham": "20030228_easy_ham.tar.bz2", "spam": "20030228_spam.tar.bz2"}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

MODELOS = {
    "naive_bayes":         lambda: MultinomialNB(),
    "regresion_logistica": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    "svm_lineal":          lambda: LinearSVC(random_state=SEED),
}

RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
RE_MAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_NUM = re.compile(r"\b\d[\d.,]*\b")
RE_TAG = re.compile(r"<[^>]+>")
RE_ESPACIO = re.compile(r"\s+")


def descargar() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for etiqueta, archivo in CORPUS.items():
        destino = RAW_DIR / etiqueta
        if destino.exists():
            continue
        tgz = RAW_DIR / archivo
        if not tgz.exists():
            print(f"[spam] descargando {archivo}...")
            urllib.request.urlretrieve(f"{BASE_URL}/{archivo}", tgz)
        print(f"[spam] extrayendo {archivo}...")
        with tarfile.open(tgz, "r:bz2") as tar:
            tar.extractall(destino, filter="data")


def texto_de(mensaje) -> str:
    """Extrae el cuerpo legible, prefiriendo texto plano sobre HTML."""
    plano, html = [], []
    for parte in mensaje.walk():
        if parte.is_multipart():
            continue
        tipo = parte.get_content_type()
        if tipo not in ("text/plain", "text/html"):
            continue
        try:
            contenido = parte.get_content()
        except Exception:
            # Codificaciones rotas y adjuntos mal formados abundan en el
            # corpus; se ignoran esas partes en vez de tirar el correo.
            continue
        (plano if tipo == "text/plain" else html).append(contenido)

    if plano:
        return "\n".join(plano)
    return RE_TAG.sub(" ", "\n".join(html))


def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = RE_URL.sub(" URL ", texto)
    texto = RE_MAIL.sub(" CORREO ", texto)
    texto = RE_NUM.sub(" NUMERO ", texto)
    return RE_ESPACIO.sub(" ", texto).strip()


def cargar() -> tuple:
    descargar()
    textos, etiquetas = [], []
    for etiqueta in ("ham", "spam"):
        archivos = sorted((RAW_DIR / etiqueta).rglob("*"))
        archivos = [f for f in archivos if f.is_file() and f.name != "cmds"]
        for f in archivos:
            with open(f, "rb") as fh:
                msg = email.parser.BytesParser(policy=email.policy.default).parse(fh)
            asunto = str(msg.get("Subject") or "")
            # El asunto va primero y por eso pesa: TF-IDF no conoce posiciones,
            # pero anteponerlo lo mete en el mismo saco de palabras.
            textos.append(normalizar(asunto + "\n" + texto_de(msg)))
            etiquetas.append(1 if etiqueta == "spam" else 0)
    return textos, np.array(etiquetas)


def run() -> None:
    textos, y = cargar()
    print(f"\n[spam] {len(textos):,} correos · spam el {y.mean():.1%}")
    print(f"[spam] largo mediano tras normalizar: {int(np.median([len(t) for t in textos])):,} caracteres")

    # min_df=3: una palabra que aparece en menos de tres correos es ruido o
    # una cadena única, no señal generalizable.
    vect = TfidfVectorizer(min_df=3, sublinear_tf=True, strip_accents="unicode")
    print(f"[spam] vocabulario: {len(vect.fit(textos).vocabulary_):,} términos")

    print(f"\n  {'modelo':22s} {'exactitud':>11s} {'precisión':>11s} {'exhaust.':>10s}")

    mlflow.set_experiment(EXPERIMENT)
    resultados = {}
    for nombre, hacer in MODELOS.items():
        pipe = Pipeline([("vect", TfidfVectorizer(min_df=3, sublinear_tf=True,
                                                  strip_accents="unicode")),
                         ("model", hacer())])
        acc = cross_val_score(pipe, textos, y, cv=CV, scoring="accuracy", n_jobs=-1).mean()
        preds = cross_val_predict(pipe, textos, y, cv=CV, n_jobs=-1)
        prec, rec = precision_score(y, preds), recall_score(y, preds)
        resultados[nombre] = (acc, prec, rec, preds, pipe)
        print(f"  {nombre:22s} {acc:11.4f} {prec:11.4f} {rec:10.4f}")

        with mlflow.start_run(run_name=f"spam_{nombre}"):
            mlflow.log_params({"dataset": "spamassassin", "modelo": nombre,
                               "cv": "StratifiedKFold(5, shuffle)", "n": len(textos)})
            mlflow.log_metrics({"exactitud": float(acc), "precision": float(prec),
                                "exhaustividad": float(rec)})

    mejor = max(resultados, key=lambda k: resultados[k][1])   # por precisión
    acc, prec, rec, preds, _ = resultados[mejor]
    (tn, fp), (fn, tp) = confusion_matrix(y, preds)

    print(f"\n[spam] Matriz de confusión — {mejor} (mejor precisión)")
    print("                     predijo legítimo   predijo spam")
    print(f"    legítimo              {tn:8,d}       {fp:8,d}   ← los que duelen")
    print(f"    spam                  {fn:8,d}       {tp:8,d}")

    print(f"\n  {fp:,} correos legítimos irían a la carpeta de spam.")
    print(f"  {fn:,} correos de spam llegarían a la bandeja.")
    print("  Los primeros cuestan mucho más que los segundos, y por eso aquí")
    print("  se elige el modelo por precisión y no por exactitud.")

    _umbral_seguro(textos, y)


def _umbral_seguro(textos, y, objetivo: float = 0.99) -> None:
    """Cuánta exhaustividad cuesta casi no tirar correo legítimo."""
    pipe = Pipeline([("vect", TfidfVectorizer(min_df=3, sublinear_tf=True,
                                              strip_accents="unicode")),
                     ("model", LogisticRegression(max_iter=1000, random_state=SEED))])
    scores = cross_val_predict(pipe, textos, y, cv=CV, method="predict_proba",
                               n_jobs=-1)[:, 1]
    precisiones, exhaustividades, umbrales = precision_recall_curve(y, scores)
    i = int((precisiones >= objetivo).argmax())

    print(f"\n[spam] Para {objetivo:.0%} de precisión (casi ningún legítimo perdido):")
    print(f"  umbral          {umbrales[i]:.3f}")
    print(f"  precisión       {precisiones[i]:.4f}")
    print(f"  exhaustividad   {exhaustividades[i]:.4f}   ← el spam que se deja pasar")

    with mlflow.start_run(run_name="spam_umbral_seguro"):
        mlflow.log_params({"dataset": "spamassassin", "objetivo_precision": objetivo})
        mlflow.log_metrics({"umbral": float(umbrales[i]),
                            "precision": float(precisiones[i]),
                            "exhaustividad": float(exhaustividades[i])})


if __name__ == "__main__":
    run()

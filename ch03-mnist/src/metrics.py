"""
Fase 4 — El compromiso entre precisión y exhaustividad
Referencia: Géron cap. 3, "Precision/Recall Trade-off" y "The ROC Curve"

Un clasificador no decide "sí" o "no": calcula una puntuación y la compara
contra un umbral. Mover ese umbral mueve las dos métricas en direcciones
opuestas — subirlo hace al modelo más exigente, así que se equivoca menos
cuando dice "sí" (más precisión) pero se le escapan más (menos exhaustividad).

No existe un umbral óptimo en abstracto. Depende de qué cueste equivocarse:

  Un filtro de spam quiere precisión alta — mandar un correo importante a la
  basura es peor que dejar pasar publicidad.

  Un detector de tumores quiere exhaustividad alta — revisar de más es un
  susto; pasar uno por alto es otra cosa.

Este módulo genera las tres gráficas del capítulo y busca el umbral que
alcanza una precisión objetivo.

Uso:
    python src/metrics.py --digit 5 --target-precision 0.90
    # o: make metrics
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict

from baseline import CV, EXPERIMENT, SEED, build_pipeline
from features import load_data

REPORTS_DIR = Path("reports")

COOL = "#00937f"   # exhaustividad
WARM = "#c2571f"   # precisión
INK = "#0b100f"
MUTED = "#7e8783"
GRID = "#e1e4de"


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def scores_for(estimator, X, y) -> np.ndarray:
    """Puntuaciones limpias: cada imagen puntuada por un modelo que no la vio."""
    if hasattr(estimator.named_steps["model"], "decision_function"):
        return cross_val_predict(estimator, X, y, cv=CV, method="decision_function", n_jobs=-1)
    # Los bosques no dan decision_function; su probabilidad de la clase
    # positiva cumple el mismo papel de puntuación ordenable.
    proba = cross_val_predict(estimator, X, y, cv=CV, method="predict_proba", n_jobs=-1)
    return proba[:, 1]


def run(digit: int, target_precision: float) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, _, y_train, _ = load_data()
    y_bin = (y_train == digit)

    print(f"\n[metrics] Puntuando con validación cruzada ({CV} pliegues)...")
    sgd = build_pipeline(SGDClassifier(random_state=SEED))
    scores = scores_for(sgd, X_train, y_bin)

    precisions, recalls, thresholds = precision_recall_curve(y_bin, scores)

    # El umbral más bajo que alcanza la precisión objetivo. Se busca el
    # mínimo porque a mayor umbral, mayor precisión pero menor exhaustividad:
    # queremos pagar lo menos posible en exhaustividad.
    idx = int((precisions >= target_precision).argmax())
    umbral = float(thresholds[idx])
    prec_en_umbral = float(precisions[idx])
    rec_en_umbral = float(recalls[idx])

    print(f"\n[metrics] Para alcanzar {target_precision:.0%} de precisión:")
    print(f"  umbral          {umbral:+.2f}")
    print(f"  precisión       {prec_en_umbral:.4f}")
    print(f"  exhaustividad   {rec_en_umbral:.4f}   ← lo que cuesta esa precisión")

    print("\n[metrics] Entrenando un bosque aleatorio para comparar (tarda)...")
    forest = build_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=-1))
    scores_forest = scores_for(forest, X_train, y_bin)

    auc_sgd = roc_auc_score(y_bin, scores)
    auc_forest = roc_auc_score(y_bin, scores_forest)
    print(f"  área bajo ROC — lineal {auc_sgd:.4f} | bosque {auc_forest:.4f}")

    _plot(digit, precisions, recalls, thresholds, umbral, target_precision,
          y_bin, scores, scores_forest, auc_sgd, auc_forest)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"umbral_{digit}"):
        mlflow.log_params({"tarea": f"detectar_{digit}", "cv": CV,
                           "precision_objetivo": target_precision})
        mlflow.log_metrics({
            "umbral": umbral,
            "precision_en_umbral": prec_en_umbral,
            "exhaustividad_en_umbral": rec_en_umbral,
            "auc_lineal": float(auc_sgd),
            "auc_bosque": float(auc_forest),
        })
        for f in ("umbral.png", "curva_pr.png", "curva_roc.png"):
            mlflow.log_artifact(str(REPORTS_DIR / f))
        print(f"[metrics] registrado en MLflow · run {mlflow.active_run().info.run_id[:8]}")


def _plot(digit, precisions, recalls, thresholds, umbral, objetivo,
          y_bin, scores, scores_forest, auc_sgd, auc_forest) -> None:
    # ── 1. Las dos métricas contra el umbral ────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(thresholds, precisions[:-1], color=WARM, lw=2, label="Precisión")
    ax.plot(thresholds, recalls[:-1], color=COOL, lw=2, ls="--", label="Exhaustividad")
    ax.axvline(umbral, color=MUTED, lw=1)
    ax.annotate(f"umbral para {objetivo:.0%} de precisión",
                xy=(umbral, 0.5), xytext=(8, 0), textcoords="offset points",
                color=MUTED, fontsize=9, va="center")
    ax.set_xlabel("Umbral de decisión", color=INK, fontsize=10)
    ax.set_ylabel("Valor de la métrica", color=INK, fontsize=10)
    ax.set_title(f"Subir el umbral sube una métrica y baja la otra — dígito {digit}",
                 color=INK, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 1.02)
    _style(ax)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "umbral.png", dpi=150)
    plt.close(fig)

    # ── 2. Precisión contra exhaustividad ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot(recalls, precisions, color=COOL, lw=2)
    i = int((precisions >= objetivo).argmax())
    ax.plot(recalls[i], precisions[i], "o", color=WARM, ms=8,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"{precisions[i]:.0%} precisión\n{recalls[i]:.0%} exhaustividad",
                xy=(recalls[i], precisions[i]), xytext=(-10, -42),
                textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("Exhaustividad", color=INK, fontsize=10)
    ax.set_ylabel("Precisión", color=INK, fontsize=10)
    ax.set_title("El precio de cada punto de precisión", color=INK,
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    _style(ax)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "curva_pr.png", dpi=150)
    plt.close(fig)

    # ── 3. ROC, comparando dos modelos ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for sc, auc, color, name, ls in [
        (scores, auc_sgd, WARM, "Lineal (SGD)", "-"),
        (scores_forest, auc_forest, COOL, "Bosque aleatorio", "-"),
    ]:
        fpr, tpr, _ = roc_curve(y_bin, sc)
        ax.plot(fpr, tpr, color=color, lw=2, ls=ls, label=f"{name} — área {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Azar")
    ax.set_xlabel("Falsos positivos (proporción)", color=INK, fontsize=10)
    ax.set_ylabel("Verdaderos positivos (proporción)", color=INK, fontsize=10)
    ax.set_title("Cuanto más arriba a la izquierda, mejor", color=INK,
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _style(ax)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "curva_roc.png", dpi=150)
    plt.close(fig)

    print(f"[metrics] gráficas → {REPORTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compromiso precisión/exhaustividad.")
    parser.add_argument("--digit", type=int, default=5, choices=range(10))
    parser.add_argument("--target-precision", type=float, default=0.90)
    args = parser.parse_args()
    run(args.digit, args.target_precision)

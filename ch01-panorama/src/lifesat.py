"""
El único código del capítulo 1
Referencia: Géron cap. 1, "Model-based learning" e "Instance-based learning"

27 países, dos columnas: PIB per cápita y satisfacción con la vida. Es el
dataset más pequeño del libro y existe para enseñar una sola cosa — que hay
dos maneras distintas de generalizar a un caso nuevo:

  Por modelo      Resume los datos en una recta con dos parámetros. Después,
                  los países originales ya no hacen falta.

  Por instancia   No resume nada. Guarda los 27 países y, ante uno nuevo,
                  busca los más parecidos y promedia su satisfacción.

Ambos predicen para un país que no está en los datos. Dan respuestas
parecidas por caminos opuestos.

De paso, el ejemplo ilustra el subajuste: que la satisfacción de un país sea
una recta en función de su riqueza es exactamente un modelo demasiado simple
para la realidad.

Uso:
    python src/lifesat.py
    # o: make lifesat
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

URL = "https://github.com/ageron/data/raw/main/lifesat/lifesat.csv"
RAW = Path("data/raw/lifesat.csv")
REPORTS = Path("reports")

# País ausente del dataset, con su PIB per cápita real. Es el caso nuevo
# sobre el que ambos modelos tienen que pronunciarse.
NUEVO = ("Chipre", 37_655.2)

COOL = "#00937f"
WARM = "#c2571f"
INK = "#0b100f"
MUTED = "#7e8783"
GRID = "#e1e4de"


def cargar() -> pd.DataFrame:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"[lifesat] descargando {URL}")
        pd.read_csv(URL).to_csv(RAW, index=False)
    return pd.read_csv(RAW)


def run() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = cargar()
    X = df[["GDP per capita (USD)"]].values
    y = df["Life satisfaction"].values

    print(f"\n[lifesat] {len(df)} países · PIB de "
          f"{X.min():,.0f} a {X.max():,.0f} USD · satisfacción de {y.min()} a {y.max()}")

    lineal = LinearRegression().fit(X, y)
    vecinos = KNeighborsRegressor(n_neighbors=3).fit(X, y)

    nombre, pib = NUEVO
    p_lineal = float(lineal.predict([[pib]])[0])
    p_vecinos = float(vecinos.predict([[pib]])[0])

    print(f"\n[lifesat] Predicción para {nombre} (PIB {pib:,.0f} USD), que no está en los datos")
    print(f"  por modelo (recta)        {p_lineal:.2f}")
    print(f"  por instancia (3 vecinos) {p_vecinos:.2f}")
    print(f"  diferencia                {abs(p_lineal - p_vecinos):.2f}")

    print(f"\n[lifesat] Lo que guarda cada uno para predecir")
    print(f"  la recta:    2 números  → pendiente {lineal.coef_[0]:.2e}, "
          f"intersección {lineal.intercept_:.2f}")
    print(f"  los vecinos: {len(X)} países completos, y los necesita todos")

    # R² sobre los mismos datos con los que se ajustó. No es una estimación
    # de generalización — con 27 filas no hay conjunto de prueba que valga —,
    # solo indica cuánto de la variación capta la recta.
    r2 = lineal.score(X, y)
    print(f"\n[lifesat] La recta explica el {r2:.0%} de la variación")
    print("  El resto no es ruido: es que la felicidad de un país no es una")
    print("  función de su riqueza. Un modelo demasiado simple para lo que hay")
    print("  — el subajuste que describe el capítulo.")

    _plot(df, X, y, lineal, vecinos, nombre, pib, p_lineal, p_vecinos)


def _plot(df, X, y, lineal, vecinos, nombre, pib, p_lineal, p_vecinos) -> None:
    rejilla = np.linspace(X.min() * 0.9, X.max() * 1.05, 500).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.scatter(X, y, s=42, color=MUTED, alpha=.75, zorder=3, label="Países del dataset")

    ax.plot(rejilla, lineal.predict(rejilla), color=WARM, lw=2,
            label="Por modelo — una recta")
    ax.plot(rejilla, vecinos.predict(rejilla), color=COOL, lw=2, ls="--",
            label="Por instancia — 3 vecinos")

    for pred, color in [(p_lineal, WARM), (p_vecinos, COOL)]:
        ax.plot([pib], [pred], "o", ms=10, color=color,
                markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax.axvline(pib, color=MUTED, lw=1, ls=":", zorder=1)
    ax.annotate(f"{nombre}\nno está en los datos", xy=(pib, min(p_lineal, p_vecinos) - .45),
                ha="center", va="top", fontsize=9, color=INK)

    ax.set_xlabel("PIB per cápita (USD)", fontsize=10, color=INK)
    ax.set_ylabel("Satisfacción con la vida", fontsize=10, color=INK)
    ax.set_title("Dos formas de responder por un país que no viste",
                 fontsize=12, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

    fig.tight_layout()
    fig.savefig(REPORTS / "lifesat.png", dpi=150)
    plt.close(fig)
    print(f"\n[lifesat] gráfica → {REPORTS}/lifesat.png")


if __name__ == "__main__":
    run()

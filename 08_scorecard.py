# -*- coding: utf-8 -*-
"""
08_scorecard.py — El scorecard: de coeficientes a puntos
Proyecto: Give Me Some Credit — Probability of Default

QUE ES UN SCORECARD

Un banco no le dice al analista "la probabilidad de mora es 0.0713".
Le dice "este cliente tiene 618 puntos". El scorecard es esa traduccion:
una tabla donde cada rango de cada variable vale una cantidad fija de
puntos, y el score del cliente es la suma.

POR QUE IMPORTA

Es el entregable caracteristico del credit scoring, y la razon por la que
la regresion logistica sigue viva en banca pese a existir modelos mas
precisos:

  - Se puede auditar: la tabla completa cabe en una hoja.
  - Se puede explicar: al cliente rechazado se le dice exactamente que
    caracteristica le resto puntos.
  - Se puede aplicar a mano, sin ordenador.

LA MATEMATICA

La regresion logistica es lineal en los log-odds:

    ln(odds) = beta_0 + SUM(beta_i * WoE_i)

Un score es simplemente una transformacion lineal de esos log-odds, y se
fija con dos convenciones que elige el banco:

    Score base : ej. 600 puntos corresponden a odds de 50:1
    PDO        : Points to Double the Odds, ej. cada 20 puntos duplican
                 las odds de que el cliente sea bueno

    Factor = PDO / ln(2)
    Offset = Score_base - Factor * ln(Odds_base)

Y los puntos de cada atributo (variable i, bin j):

    puntos_ij = -(WoE_ij * beta_i + beta_0/n) * Factor + Offset/n

donde n es el numero de variables. El termino beta_0/n reparte el
intercepto por igual entre todas, para que la suma cuadre.

Se ejecuta con:  python 08_scorecard.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from src import artifacts
from src import woe as woe_mod

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

AQUI = Path(__file__).parent
PROC = AQUI / "datasets" / "processed" / "woe"
MODELOS = AQUI / "artifacts" / "models"
METRICAS = AQUI / "artifacts" / "metrics"
PLOTS = AQUI / "artifacts" / "plots"
PREPROC = AQUI / "artifacts" / "preprocessors"

TARGET = "SeriousDlqin2yrs"

# --- Convenciones de escala (las fija el banco, no los datos) ----------------
SCORE_BASE = 600      # puntos de referencia
ODDS_BASE = 50        # ...que corresponden a odds de 50:1 (bueno:malo)
PDO = 20              # cada 20 puntos duplican las odds

FACTOR = PDO / np.log(2)
OFFSET = SCORE_BASE - FACTOR * np.log(ODDS_BASE)


def titulo(t):
    print("\n" + "=" * 79)
    print(t)
    print("=" * 79)


def score_desde_pd(p):
    """Convierte una probabilidad de mora en puntos."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    odds_bueno = (1 - p) / p
    return OFFSET + FACTOR * np.log(odds_bueno)


def pd_desde_score(s):
    """El camino inverso: de puntos a probabilidad de mora."""
    odds_bueno = np.exp((np.asarray(s, dtype=float) - OFFSET) / FACTOR)
    return 1.0 / (1.0 + odds_bueno)


# =============================================================================
# PASO 1 — CARGA
# =============================================================================
titulo("PASO 1 — MODELO Y TABLAS WoE")

art_mod = joblib.load(MODELOS / "logistic_v1.joblib")
modelo, COLS = art_mod["modelo"], art_mod["columnas"]
art_pre = artifacts.cargar(PREPROC / "preprocessor_v1.joblib")
tablas = art_pre["tablas_woe"]

beta = pd.Series(modelo.coef_[0], index=COLS)
beta0 = float(modelo.intercept_[0])
n_vars = len(COLS)

print(f"  Modelo     : {art_mod['subconjunto']}")
print(f"  Variables  : {n_vars}")
print(f"  Intercepto : {beta0:.4f}")

print(f"\n  Escala del scorecard (convencion del banco, no de los datos):")
print(f"    {SCORE_BASE} puntos  <->  odds {ODDS_BASE}:1 (bueno:malo)")
print(f"    PDO = {PDO} puntos duplican las odds")
print(f"    Factor = PDO/ln(2) = {FACTOR:.4f}")
print(f"    Offset = {SCORE_BASE} - Factor*ln({ODDS_BASE}) = {OFFSET:.4f}")

print(f"\n  Equivalencia score <-> probabilidad de mora:")
for s in [400, 500, 550, 600, 650, 700, 800]:
    print(f"    {s:>4} puntos  ->  PD = {pd_desde_score(s)*100:6.2f}%")


# =============================================================================
# PASO 2 — CONSTRUIR LA TABLA DE PUNTOS
# =============================================================================
titulo("PASO 2 — LA TABLA DE PUNTOS")

# puntos_ij = -(WoE_ij * beta_i + beta_0/n) * Factor + Offset/n
def puntos(woe, b):
    return -(woe * b + beta0 / n_vars) * FACTOR + OFFSET / n_vars

filas = []
for col in COLS:
    b = beta[col]
    if col.startswith("woe_"):
        original = col[4:]
        t = tablas[original]
        for bin_, r in t.iterrows():
            filas.append({
                "variable": original,
                "rango": str(bin_),
                "n": int(r["n"]),
                "pct_pobl": r["pct_poblacion"],
                "default_pct": r["default_rate_pct"],
                "woe": round(float(r["woe"]), 4),
                "puntos": round(float(puntos(r["woe"], b)), 1),
            })
    else:
        # Los flags no estan bineados: valen 0 o 1 directamente.
        for v in (0, 1):
            filas.append({
                "variable": col, "rango": f"= {v}",
                "n": np.nan, "pct_pobl": np.nan, "default_pct": np.nan,
                "woe": float(v),
                "puntos": round(float(puntos(v, b)), 1),
            })

scorecard = pd.DataFrame(filas)

# Los puntos son relativos: lo que importa es la DIFERENCIA entre bins de
# una misma variable. Centramos cada variable en su caso "normal" para que
# la tabla se lea como "cuanto sumas o restas respecto a lo habitual".
#   - variables bineadas -> el bin con mas poblacion
#   - flags              -> el valor 0 (no activado), que es el caso comun
ref = (scorecard.dropna(subset=["n"])
       .sort_values("n", ascending=False).groupby("variable").head(1)
       .set_index("variable")["puntos"].to_dict())
# CUIDADO: los flags no tienen columna n, asi que quedaban fuera del mapa y
# su "vs_normal" salia siempre 0.0 — incluso estando activados. Su referencia
# tiene que ser explicitamente el valor 0.
for col in COLS:
    if not col.startswith("woe_"):
        ref[col] = float(puntos(0.0, beta[col]))

scorecard["puntos_rel"] = scorecard.apply(
    lambda r: round(r["puntos"] - ref.get(r["variable"], r["puntos"]), 1), axis=1)

for v in scorecard["variable"].unique():
    sub = scorecard[scorecard["variable"] == v]
    print(f"\n  --- {v} ---")
    cols_mostrar = ["rango", "n", "pct_pobl", "default_pct", "woe", "puntos", "puntos_rel"]
    print(sub[cols_mostrar].to_string(index=False, na_rep="-"))

scorecard.to_csv(METRICAS / "scorecard.csv", index=False)
print(f"\n  Guardado: artifacts/metrics/scorecard.csv")


# =============================================================================
# PASO 3 — VERIFICACION: ¿la suma de puntos reproduce el modelo?
# =============================================================================
titulo("PASO 3 — VERIFICACION")

# Esta es la prueba que valida todo el scorecard: sumar los puntos de cada
# caracteristica de una persona debe dar EXACTAMENTE el mismo score que
# transformar la PD del modelo. Si no cuadra, la tabla es decorativa.
val = pd.read_parquet(PROC / "val.parquet")
y_val = val[TARGET]
X_val = val.drop(columns=[TARGET])

mapa = {}
for col in COLS:
    b = beta[col]
    if col.startswith("woe_"):
        mapa[col] = lambda w, b=b: puntos(w, b)
    else:
        mapa[col] = lambda w, b=b: puntos(w, b)

score_sumado = np.zeros(len(X_val))
for col in COLS:
    score_sumado += puntos(X_val[col].to_numpy(float), beta[col])

p_modelo = modelo.predict_proba(X_val[COLS])[:, 1]
score_modelo = score_desde_pd(p_modelo)

dif = np.abs(score_sumado - score_modelo)
print(f"  Diferencia entre 'sumar puntos' y 'transformar la PD del modelo':")
print(f"    maxima : {dif.max():.10f} puntos")
print(f"    media  : {dif.mean():.10f} puntos")
ok = dif.max() < 1e-6
print(f"  [{'OK' if ok else 'ERROR'}] la suma de puntos reproduce el modelo exactamente")


# =============================================================================
# PASO 4 — EJEMPLO: UN CLIENTE CONCRETO
# =============================================================================
titulo("PASO 4 — COMO SE EXPLICA UN RECHAZO")

# Elegimos a la persona con peor score de validacion
i = int(np.argmin(score_sumado))
persona = X_val.iloc[i]

detalle = []
for col in COLS:
    p_col = puntos(float(persona[col]), beta[col])
    nombre = col[4:] if col.startswith("woe_") else col
    detalle.append({"caracteristica": nombre, "puntos": round(float(p_col), 1),
                    "vs_normal": round(float(p_col - ref.get(nombre, p_col)), 1)})
det = pd.DataFrame(detalle).sort_values("vs_normal")

print(f"  Cliente #{X_val.index[i]}  |  score {score_sumado[i]:.0f} puntos  |  "
      f"PD {p_modelo[i]*100:.1f}%  |  ¿cayo en mora? {'SI' if y_val.iloc[i]==1 else 'NO'}")
print(f"\n  Desglose (ordenado de lo que mas resta a lo que mas suma):")
print(det.to_string(index=False))
print(f"\n  SCORE TOTAL: {det['puntos'].sum():.0f} puntos")
print("\n  Esto es lo que la regresion logistica da y los boosters no:")
print("  se le puede decir al cliente exactamente que le resto puntos.")


# =============================================================================
# PASO 5 — DISTRIBUCION Y BANDAS DE RIESGO
# =============================================================================
titulo("PASO 5 — BANDAS DE RIESGO")

# Los bancos agrupan el score en bandas y asignan politica a cada una.
bandas = pd.cut(score_sumado, [-np.inf, 500, 550, 600, 650, 700, np.inf],
                labels=["<500", "500-550", "550-600", "600-650", "650-700", ">700"])
tabla_bandas = pd.DataFrame({"banda": bandas, "y": y_val.to_numpy()}) \
    .groupby("banda", observed=True).agg(n=("y", "size"), default=("y", "mean"))
tabla_bandas["pct_pobl"] = (tabla_bandas["n"] / len(y_val) * 100).round(2)
tabla_bandas["default_pct"] = (tabla_bandas["default"] * 100).round(2)
tabla_bandas = tabla_bandas.drop(columns=["default"])
print(tabla_bandas.to_string())

umbral_pd = 800 / (800 + 6000)
umbral_score = float(score_desde_pd(umbral_pd))
print(f"\n  El umbral de negocio (PD < {umbral_pd*100:.2f}%) equivale a "
      f"{umbral_score:.0f} puntos.")
print(f"  Politica: aprobar por encima de {umbral_score:.0f}, rechazar por debajo.")
print(f"  Se aprobaria al {(score_sumado >= umbral_score).mean()*100:.1f}% de los solicitantes.")

tabla_bandas.to_csv(METRICAS / "scorecard_bandas.csv")


# =============================================================================
# PASO 6 — GRAFICAS
# =============================================================================
titulo("PASO 6 — GRAFICAS")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

ax = axes[0]
ax.hist(score_sumado[y_val == 0], bins=60, alpha=0.7, density=True,
        color="#2b6247", label="no cayo en mora")
ax.hist(score_sumado[y_val == 1], bins=60, alpha=0.7, density=True,
        color="#a8392f", label="si cayo en mora")
ax.axvline(umbral_score, color="black", ls="--", lw=2,
           label=f"corte {umbral_score:.0f}")
ax.set_xlabel("Score"); ax.set_ylabel("Densidad")
ax.set_title("Distribucion del score"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
x = np.arange(len(tabla_bandas))
ax.bar(x, tabla_bandas["default_pct"], color="#a8392f")
ax.set_xticks(x); ax.set_xticklabels(tabla_bandas.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("% default"); ax.set_title("Riesgo por banda de score")
ax.grid(alpha=0.3, axis="y")
for xi, (n, d) in enumerate(zip(tabla_bandas["n"], tabla_bandas["default_pct"])):
    ax.text(xi, d, f"n={n:,}", ha="center", va="bottom", fontsize=7)

ax = axes[2]
top = scorecard.dropna(subset=["n"]).copy()
top["rango_corto"] = top["variable"].str.slice(0, 18) + " " + top["rango"].str.slice(0, 12)
top = top.reindex(top["puntos_rel"].abs().sort_values(ascending=False).index).head(14).iloc[::-1]
colores = ["#2b6247" if v > 0 else "#a8392f" for v in top["puntos_rel"]]
ax.barh(range(len(top)), top["puntos_rel"], color=colores)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top["rango_corto"], fontsize=6.5)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Puntos respecto al bin mas frecuente")
ax.set_title("Atributos que mas mueven el score"); ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(PLOTS / "08_scorecard.png", dpi=110)
plt.close()
print(f"  {(PLOTS / '08_scorecard.png').relative_to(AQUI)}")
print("  artifacts/metrics/scorecard.csv")
print("  artifacts/metrics/scorecard_bandas.csv")

print("\n" + "=" * 79)
print(f"SCORECARD GENERADO — {len(scorecard)} atributos | "
      f"corte en {umbral_score:.0f} puntos" if ok else "ERROR EN LA VERIFICACION")
print("=" * 79)
sys.exit(0 if ok else 1)

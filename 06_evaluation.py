# -*- coding: utf-8 -*-
"""
06_evaluation.py — Comparacion final, eleccion del modelo y submission
Proyecto: Give Me Some Credit — Probability of Default

Tres cosas:
  1. Comparar los tres modelos con criterios MULTIPLES, no solo AUC.
  2. Elegir uno de forma razonada y dejar el razonamiento por escrito.
  3. Reentrenarlo con las 150,000 filas etiquetadas y predecir sobre
     cs-test.csv en el formato de sampleEntry.csv.

POR QUE SE REENTRENA CON TODO AL FINAL
  Hasta ahora apartamos 30,000 filas de validacion para poder medir sin
  enganarnos. Pero una vez tomadas TODAS las decisiones (variables,
  hiperparametros, numero de arboles), esas 30,000 filas ya no tienen que
  seguir escondidas: son datos etiquetados que el modelo final puede
  aprovechar. Mas datos, mejor modelo.
  Lo que NO se puede hacer es volver a mirarlas para decidir algo.

Se ejecuta con:  python 06_evaluation.py
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             precision_recall_curve)

sys.path.insert(0, str(Path(__file__).parent))
from src import preprocessing as pp
from src import woe as woe_mod
from src import artifacts
from src.evaluation import (ks_statistic, gini, error_calibracion,
                            umbral_optimo, beneficio, validar_submission)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

AQUI = Path(__file__).parent
DATA = AQUI.parent.parent / "data"
PROC = AQUI / "datasets" / "processed"
MODELOS = AQUI / "artifacts" / "models"
METRICAS = AQUI / "artifacts" / "metrics"
PLOTS = AQUI / "artifacts" / "plots"
SUBMISSIONS = AQUI / "artifacts" / "submissions"
SUBMISSIONS.mkdir(parents=True, exist_ok=True)

TARGET = "SeriousDlqin2yrs"

# Supuestos economicos (los mismos de 07_business_evaluation.py)
MARGEN, PERDIDA = 800, 6000
UMBRAL_NEGOCIO = MARGEN / (MARGEN + PERDIDA)


def titulo(t):
    print("\n" + "=" * 79)
    print(t)
    print("=" * 79)


# =============================================================================
# PASO 1 — CARGA
# =============================================================================
titulo("PASO 1 — MODELOS Y DATOS")

RAMA = {"logistic_v1": "woe", "xgboost_v1": "trees", "lightgbm_v1": "trees"}
NOMBRE = {"logistic_v1": "LogReg (WoE)", "xgboost_v1": "XGBoost", "lightgbm_v1": "LightGBM"}

modelos = {}
for stem, rama in RAMA.items():
    f = MODELOS / f"{stem}.joblib"
    if f.exists():
        modelos[stem] = joblib.load(f)
        print(f"  {NOMBRE[stem]:<14} rama={rama:<6} "
              f"{len(modelos[stem]['columnas'])} variables")

if not modelos:
    sys.exit("No hay modelos entrenados. Ejecuta 03, 04 y 05 primero.")

val = {r: pd.read_parquet(PROC / r / "val.parquet") for r in set(RAMA.values())}
y_val = val["trees"][TARGET]

pred = {}
for stem, art in modelos.items():
    X = val[RAMA[stem]].drop(columns=[TARGET])
    pred[stem] = art["modelo"].predict_proba(X[art["columnas"]])[:, 1]

# --- Ensemble: promedio de las probabilidades -------------------------------
# Truco clasico: si los modelos se equivocan en cosas distintas, su promedio
# se equivoca menos. Cuesta nada probarlo.
if len(pred) == 3:
    pred["ensemble"] = np.mean(list(pred.values()), axis=0)
    NOMBRE["ensemble"] = "Ensemble (promedio)"

print(f"\n  Validacion: {len(y_val):,} personas | {int(y_val.sum()):,} morosos")


# =============================================================================
# PASO 2 — COMPARACION MULTI-CRITERIO
# =============================================================================
titulo("PASO 2 — COMPARACION EN VALIDACION")

cv_tabla = {}
f_cv = METRICAS / "comparacion_modelos_cv.csv"
if f_cv.exists():
    cv_tabla = pd.read_csv(f_cv).set_index("modelo").to_dict("index")

filas = []
for stem, p in pred.items():
    a = roc_auc_score(y_val, p)
    ks, _ = ks_statistic(y_val, p)
    cvi = cv_tabla.get(NOMBRE[stem], {})
    filas.append({
        "modelo": NOMBRE[stem],
        "AUC_val": round(a, 5),
        "Gini": round(gini(a), 4),
        "KS": round(ks, 5),
        "AP": round(average_precision_score(y_val, p), 4),
        "calib_pp": round(error_calibracion(y_val, p), 3),
        "AUC_cv": cvi.get("auc_cv_media", np.nan),
        "cv_desv": cvi.get("auc_cv_desv", np.nan),
        "beneficio_$": beneficio(p, y_val, UMBRAL_NEGOCIO, MARGEN, PERDIDA)["beneficio"],
    })

comp = pd.DataFrame(filas).sort_values("AUC_val", ascending=False).reset_index(drop=True)
print(comp.to_string(index=False))

print("\n  Gini = 2*AUC - 1. Es la metrica que usa la banca por costumbre;")
print("  contiene exactamente la misma informacion que el AUC.")


# =============================================================================
# PASO 3 — LA DECISION NO ES SOLO EL AUC
# =============================================================================
titulo("PASO 3 — CRITERIOS DE DECISION")

criterios = pd.DataFrame([
    {"criterio": "Poder predictivo (AUC)",   "LogReg": "0.85948", "XGBoost": "0.86633", "LightGBM": "0.86588"},
    {"criterio": "¿Diferencia significativa?", "LogReg": "-", "XGBoost": "gana a LogReg", "LightGBM": "gana a LogReg"},
    {"criterio": "Calibracion (error pp)",   "LogReg": "0.53", "XGBoost": "0.34", "LightGBM": "0.39"},
    {"criterio": "Interpretabilidad",        "LogReg": "TOTAL (scorecard)", "XGBoost": "baja", "LightGBM": "baja"},
    {"criterio": "Explicable a un regulador","LogReg": "si", "XGBoost": "requiere SHAP", "LightGBM": "requiere SHAP"},
    {"criterio": "Velocidad de entrenamiento","LogReg": "instantanea", "XGBoost": "1.7s", "LightGBM": "0.4s"},
    {"criterio": "Complejidad de despliegue", "LogReg": "minima", "XGBoost": "media", "LightGBM": "media"},
    {"criterio": "Riesgo de sobreajuste",    "LogReg": "muy bajo", "XGBoost": "medio", "LightGBM": "medio-alto"},
])
print(criterios.to_string(index=False))

print("\n  XGBoost y LightGBM son estadisticamente INDISTINGUIBLES entre si")
print("  (+0.00045, dentro del ruido). La eleccion entre ellos no puede")
print("  basarse en el AUC: hay que usar los demas criterios.")


# =============================================================================
# PASO 4 — ELECCION DEL MODELO FINAL
# =============================================================================
titulo("PASO 4 — MODELO ELEGIDO")

# Criterio: entre los que empatan en poder predictivo, gana el mejor
# calibrado. Las probabilidades calibradas son lo que hace funcionar la
# regla de negocio umbral = M/(M+L); sin ellas el modelo ordena bien pero
# no permite decidir cuanto prestar.
candidatos = comp[comp["modelo"] != "Ensemble (promedio)"]
mejor_auc = candidatos["AUC_val"].max()
empatados = candidatos[candidatos["AUC_val"] >= mejor_auc - 0.002]
elegido_nombre = empatados.sort_values("calib_pp").iloc[0]["modelo"]
ELEGIDO = [k for k, v in NOMBRE.items() if v == elegido_nombre][0]

print(f"  ELEGIDO: {elegido_nombre}")
print(f"\n  Razonamiento:")
print(f"   1. XGBoost y LightGBM superan a LogReg de forma consistente")
print(f"      (+0.0069 y +0.0064, ambas mayores que el ruido).")
print(f"   2. Entre ellos la diferencia es +0.00045: ruido puro.")
print(f"   3. Desempate por CALIBRACION, porque de ella depende que la regla")
print(f"      umbral = M/(M+L) funcione: {elegido_nombre} tiene el menor error.")
print(f"\n  NOTA PARA UN CONTEXTO REAL: si el modelo tuviera que pasar por un")
print(f"  comite de riesgo, la regresion logistica seguiria siendo defendible.")
print(f"  Cuesta 0.007 de AUC (~$5 por solicitud) y a cambio da un scorecard")
print(f"  que explica cada rechazo. Esa decision es de negocio, no tecnica.")


# =============================================================================
# PASO 4B — EXPLICABILIDAD DEL MODELO ELEGIDO (SHAP)
# =============================================================================
titulo("PASO 4B — ¿POR QUE RECHAZA A QUIEN RECHAZA?")

# El problema de elegir un booster: la regresion logistica da un scorecard
# donde cada rango vale unos puntos fijos, y con eso se le explica a un
# cliente rechazado exactamente que le resto. Un modelo de 105 arboles no
# tiene esa tabla.
#
# SHAP lo resuelve: reparte la prediccion de CADA persona entre sus
# variables, de forma que las contribuciones suman exactamente el
# resultado del modelo. Es lo mas parecido a un scorecard que puede tener
# un booster — con una diferencia clave: en el scorecard los puntos son
# los MISMOS para todos; en SHAP cada persona tiene su propio desglose.
try:
    import shap
    _hay_shap = True
except ImportError:
    _hay_shap = False
    print("  shap no esta instalado (pip install shap). Se omite este paso.")

if _hay_shap:
    # TreeSHAP solo aplica a ensambles de arboles; se explica el booster con
    # mejor AUC de validacion. Para la regresion logistica el equivalente es
    # el scorecard (08_scorecard.py), que ya da atribucion exacta por atributo.
    stem_shap = comp[comp["modelo"].isin(["XGBoost", "LightGBM"])].iloc[0]["modelo"]
    stem_shap = [k for k, v in NOMBRE.items() if v == stem_shap][0]
    art_s = modelos[stem_shap]
    Xs = val["trees"].drop(columns=[TARGET])[art_s["columnas"]]

    explainer = shap.TreeExplainer(art_s["modelo"])
    muestra = Xs.sample(3000, random_state=42)
    shap_vals = explainer.shap_values(muestra)

    print(f"  Modelo explicado: {NOMBRE[stem_shap]}  ({len(muestra):,} personas)")

    # --- Importancia global: media del |SHAP| ---------------------------------
    imp = pd.DataFrame({
        "variable": art_s["columnas"],
        "impacto_medio": np.abs(shap_vals).mean(axis=0),
    }).sort_values("impacto_medio", ascending=False).reset_index(drop=True)
    imp["pct"] = (imp["impacto_medio"] / imp["impacto_medio"].sum() * 100).round(2)

    print("\n  Impacto medio de cada variable sobre la prediccion:")
    print(imp.head(10).to_string(index=False))
    print("\n  A diferencia de feature_importances_, esto SI es comparable entre")
    print("  modelos: mide cuanto mueve cada variable la prediccion, no cuantas")
    print("  veces la uso el algoritmo para partir.")

    # --- Explicacion individual ----------------------------------------------
    p_s = art_s["modelo"].predict_proba(Xs)[:, 1]
    i_peor = int(np.argmax(p_s))
    fila = Xs.iloc[[i_peor]]
    sv = explainer.shap_values(fila)[0]

    det = pd.DataFrame({
        "variable": art_s["columnas"],
        "valor": fila.iloc[0].values,
        "contribucion": sv,
    })
    det = det.reindex(det["contribucion"].abs().sort_values(ascending=False).index)

    print(f"\n  --- EXPLICACION DE UN RECHAZO CONCRETO ---")
    print(f"  Cliente #{Xs.index[i_peor]}  |  PD = {p_s[i_peor]*100:.1f}%  |  "
          f"¿cayo en mora? {'SI' if y_val.iloc[i_peor]==1 else 'NO'}")
    print(f"  Base (log-odds medio de la poblacion): {explainer.expected_value:+.4f}")
    print("\n  " + det.head(8).to_string(index=False).replace("\n", "\n  "))
    print(f"\n  Suma de contribuciones + base = {explainer.expected_value + sv.sum():+.4f}")
    print(f"  (en log-odds; equivale a PD = "
          f"{1/(1+np.exp(-(explainer.expected_value + sv.sum())))*100:.1f}%)")
    print("\n  Contribucion positiva = empuja hacia MORA. Negativa = hacia pagar.")
    print("  Esto es lo que se le entrega a un regulador que pregunte por que")
    print("  se rechazo a esta persona concreta.")

    # --- Grafica --------------------------------------------------------------
    fig, axes_s = plt.subplots(1, 2, figsize=(15, 5.5))
    t = imp.head(12).iloc[::-1]
    axes_s[0].barh(range(len(t)), t["impacto_medio"], color="#7a3a6a")
    axes_s[0].set_yticks(range(len(t)))
    axes_s[0].set_yticklabels(t["variable"], fontsize=7)
    axes_s[0].set_xlabel("Impacto medio |SHAP| sobre los log-odds")
    axes_s[0].set_title(f"Importancia global — {NOMBRE[stem_shap]}")
    axes_s[0].grid(alpha=0.3, axis="x")

    d2 = det.head(10).iloc[::-1]
    cols_b = ["#a8392f" if v > 0 else "#2b6247" for v in d2["contribucion"]]
    axes_s[1].barh(range(len(d2)), d2["contribucion"], color=cols_b)
    axes_s[1].set_yticks(range(len(d2)))
    axes_s[1].set_yticklabels([f"{v}={x:g}" for v, x in zip(d2["variable"], d2["valor"])],
                              fontsize=7)
    axes_s[1].axvline(0, color="black", lw=0.8)
    axes_s[1].set_xlabel("Contribucion a los log-odds  (rojo = mas riesgo)")
    axes_s[1].set_title(f"Por que se rechaza al cliente #{Xs.index[i_peor]}")
    axes_s[1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(PLOTS / "06_shap_explicabilidad.png", dpi=110)
    plt.close()
    imp.to_csv(METRICAS / "shap_importancia.csv", index=False)
    print(f"\n  Guardado: artifacts/plots/06_shap_explicabilidad.png")
    print(f"            artifacts/metrics/shap_importancia.csv")


# =============================================================================
# PASO 5 — REENTRENAR CON LAS 150,000 FILAS
# =============================================================================
titulo("PASO 5 — REENTRENAMIENTO CON TODOS LOS DATOS ETIQUETADOS")

art = modelos[ELEGIDO]
rama = RAMA[ELEGIDO]
COLS = art["columnas"]

print(f"  Hasta ahora el modelo se entreno con 120,000 filas (u 96,000 con")
print(f"  early stopping). Ya no hace falta esconder la validacion: las")
print(f"  decisiones estan tomadas.")

test_raw = pd.read_csv(DATA / "cs-test.csv", index_col=0)
test_raw.index.name = "row_id"

if rama == "trees":
    tr = pd.read_parquet(PROC / "trees" / "train.parquet")
    va = pd.read_parquet(PROC / "trees" / "val.parquet")
    todo = pd.concat([tr, va])
    X_todo, y_todo = todo[COLS], todo[TARGET]
    X_test = pd.read_parquet(PROC / "trees" / "test.parquet")[COLS]
else:
    # En la rama WoE hay que RE-APRENDER las tablas con las 150,000 filas.
    # Los cortes de los bins son los mismos; lo que cambia es el WoE de cada
    # bin, ahora calculado con mas datos.
    crudo = pd.read_csv(DATA / "cs-training.csv", index_col=0)
    y_todo = crudo[TARGET]
    Xc = pp.crear_features(pp.limpiar(crudo.drop(columns=[TARGET])))
    tablas = woe_mod.aprender_woe(Xc, y_todo)
    X_todo = woe_mod.aplicar_woe(Xc, tablas)[COLS]
    Tc = pp.crear_features(pp.limpiar(test_raw.drop(columns=[TARGET])))
    X_test = woe_mod.aplicar_woe(Tc, tablas)[COLS]
    print("  (rama WoE: tablas re-aprendidas con las 150,000 filas)")

modelo_final = art["modelo"].__class__(**art["modelo"].get_params())

# El numero de arboles ya se fijo con early stopping; al reentrenar con todo
# no hay set de vigilancia, asi que se usa ese numero como constante.
if "n_arboles" in art:
    n = art["n_arboles"]
    modelo_final.set_params(n_estimators=n)
    for p in ("early_stopping_rounds",):
        if p in modelo_final.get_params():
            modelo_final.set_params(**{p: None})
    print(f"  Arboles fijados en {n} (el valor que eligio el early stopping)")

modelo_final.fit(X_todo, y_todo)
print(f"  Reentrenado con {len(X_todo):,} filas ({y_todo.mean()*100:.2f}% de morosos)")

# --- Congelar el preprocesador que corresponde a ESTE modelo ----------------
# El artefacto de 02_preprocessing.py se aprendio con 120,000 filas. Si el
# modelo desplegado se entreno con 150,000, ambos tienen que ir emparejados:
# un modelo y un preprocesador de "epocas" distintas producen puntuaciones
# que nadie puede reproducir despues.
art_pre = artifacts.cargar(AQUI / "artifacts" / "preprocessors" / "preprocessor_v1.joblib")
art_final = artifacts.construir_artefacto(
    tablas_woe=(tablas if rama == "woe" else art_pre["tablas_woe"]),
    columnas_trees=(COLS if rama == "trees" else art_pre["columnas_trees"]),
    columnas_woe=(COLS if rama == "woe" else art_pre["columnas_woe"]),
    n_filas_train=len(X_todo),
    tasa_default=float(y_todo.mean()),
    semilla=art_pre["metadatos"]["semilla"],
)
ruta_pre = artifacts.guardar(
    art_final, AQUI / "artifacts" / "preprocessors" / "preprocessor_final.joblib")
print(f"  Preprocesador emparejado guardado: {ruta_pre.name} "
      f"({len(X_todo):,} filas)")
if rama == "trees":
    print("  (rama 'trees': la limpieza es por reglas, no aprende del numero")
    print("   de filas; las tablas WoE se conservan solo por trazabilidad)")

# Comprobacion de cordura: sobre validacion (que ahora forma parte del train)
# el AUC tiene que ser >= al de antes. No es una medida de rendimiento real.
Xv = val[rama].drop(columns=[TARGET])[COLS] if rama == "trees" else X_todo.loc[val[rama].index]
print(f"  AUC in-sample sobre las filas de validacion: "
      f"{roc_auc_score(y_val, modelo_final.predict_proba(Xv)[:, 1]):.5f}")
print("  (in-sample: NO es una estimacion de rendimiento, solo cordura)")


# =============================================================================
# PASO 6 — PREDICCION SOBRE cs-test.csv
# =============================================================================
titulo("PASO 6 — SUBMISSION")

p_test = modelo_final.predict_proba(X_test)[:, 1]

sub = pd.DataFrame({"Id": np.arange(1, len(p_test) + 1), "Probability": p_test})
ruta = SUBMISSIONS / f"submission_{ELEGIDO}.csv"
sub.to_csv(ruta, index=False)

muestra = pd.read_csv(DATA / "sampleEntry.csv")
print(f"  Formato esperado (sampleEntry.csv): {list(muestra.columns)}, {len(muestra):,} filas")
print(f"  Generado                          : {list(sub.columns)}, {len(sub):,} filas")
print(f"\n{sub.head().to_string(index=False)}")

print("\n  --- VERIFICACIONES ---")
ok = True
def check(c, m):
    global ok
    print(f"  [{'OK' if c else 'ERROR'}] {m}")
    ok &= bool(c)

# El contrato del submission esta en src/evaluation.py y tiene tests propios.
for descripcion, correcto in validar_submission(sub, len(muestra)):
    check(correcto, descripcion)

print(f"\n  PD media predicha en test : {p_test.mean()*100:.2f}%")
print(f"  PD media real en train     : {y_todo.mean()*100:.2f}%")
dif = abs(p_test.mean() - y_todo.mean()) * 100
check(dif < 1.0, f"la PD media difiere solo {dif:.2f} pp del train "
                 "(train y test vienen de la misma poblacion)")
print(f"\n  Guardado: {ruta.relative_to(AQUI)}")

# Submissions de los otros modelos, para poder compararlos en Kaggle
for stem, a in modelos.items():
    if stem == ELEGIDO:
        continue
    if RAMA[stem] != "trees":
        continue
    Xt = pd.read_parquet(PROC / "trees" / "test.parquet")[a["columnas"]]
    pt = a["modelo"].predict_proba(Xt)[:, 1]
    pd.DataFrame({"Id": np.arange(1, len(pt) + 1), "Probability": pt}).to_csv(
        SUBMISSIONS / f"submission_{stem}.csv", index=False)
    print(f"  Guardado tambien: submission_{stem}.csv")


# =============================================================================
# PASO 7 — GRAFICAS FINALES
# =============================================================================
titulo("PASO 7 — GRAFICAS")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
colores = {"logistic_v1": "#3a4f7a", "xgboost_v1": "#7a3a6a",
           "lightgbm_v1": "#1f7a5a", "ensemble": "#b08428"}

ax = axes[0, 0]
for stem, p in pred.items():
    f, t, _ = roc_curve(y_val, p)
    ax.plot(f, t, lw=2 if stem == ELEGIDO else 1.3, color=colores[stem],
            alpha=1.0 if stem == ELEGIDO else 0.75,
            label=f"{NOMBRE[stem]} ({roc_auc_score(y_val, p):.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos")
ax.set_title("Curvas ROC"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

ax = axes[0, 1]
for stem, p in pred.items():
    pr, rc, _ = precision_recall_curve(y_val, p)
    ax.plot(rc, pr, lw=2 if stem == ELEGIDO else 1.3, color=colores[stem],
            alpha=1.0 if stem == ELEGIDO else 0.75, label=NOMBRE[stem])
ax.axhline(y_val.mean(), ls="--", color="gray", lw=1, label="azar")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Curvas Precision-Recall"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
for stem, p in pred.items():
    d = pd.DataFrame({"p": p, "y": y_val.to_numpy()})
    d["dec"] = pd.qcut(d.p, 10, labels=False)
    g = d.groupby("dec").agg(pp_=("p", "mean"), pr=("y", "mean")) * 100
    ax.plot(g.pp_, g.pr, "o-", color=colores[stem], lw=1.6, ms=4, label=NOMBRE[stem])
lim = 40
ax.plot([0, lim], [0, lim], "--", color="gray", lw=1, label="calibracion perfecta")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("PD predicha (%)"); ax.set_ylabel("PD observada (%)")
ax.set_title("Calibracion: cuanto mas pegado a la diagonal, mejor")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.hist(p_test, bins=80, color="#1f7a5a", alpha=0.85)
ax.axvline(UMBRAL_NEGOCIO, color="#a8392f", ls="--", lw=2,
           label=f"umbral de negocio {UMBRAL_NEGOCIO:.4f}")
ax.set_yscale("log")
ax.set_xlabel("PD predicha"); ax.set_ylabel("Personas (escala log)")
ax.set_title(f"Distribucion de las {len(p_test):,} predicciones de test")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
rechazados = (p_test >= UMBRAL_NEGOCIO).sum()
ax.text(0.55, 0.75, f"se rechazaria a\n{rechazados:,} ({rechazados/len(p_test)*100:.1f}%)",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="#a8392f"))

plt.tight_layout()
plt.savefig(PLOTS / "06_comparacion_final.png", dpi=110)
plt.close()
print(f"  {(PLOTS / '06_comparacion_final.png').relative_to(AQUI)}")

comp.to_csv(METRICAS / "comparacion_final.csv", index=False)
print(f"  artifacts/metrics/comparacion_final.csv")

print("\n" + "=" * 79)
print(f"MODELO FINAL: {elegido_nombre}   |   submission con {len(sub):,} predicciones"
      if ok else "HAY ERRORES — REVISAR ARRIBA")
print("=" * 79)
sys.exit(0 if ok else 1)

# -*- coding: utf-8 -*-
"""
04_train_xgboost.py — XGBoost sobre los datos crudos
Proyecto: Give Me Some Credit — Probability of Default

Usa la rama 'trees' del preprocesamiento: valores CRUDOS, NaN nativos y
flags. Nada de WoE, nada de escalado, nada de imputacion.

POR QUE ESTE TRATAMIENTO ES DISTINTO AL DE LA REGRESION LOGISTICA

  1. Sin capear outliers. Un arbol solo mira el ORDEN de los valores: un
     corte en "RevUtil > 1.5" separa las mismas filas tanto si el maximo
     es 2 como si es 50,708. Capear no aporta y elimina cortes posibles.

  2. Sin imputar NaN. XGBoost aprende en cada division hacia que lado
     mandar los faltantes. Y aqui el faltante ES predictivo (5.57% de
     default frente a 6.96%), asi que imputar destruiria senal.

  3. Con todas las variables derivadas. Los arboles no sufren
     multicolinealidad: en cada division eligen UNA variable, no reparten
     peso entre varias. (Lo verificamos empiricamente en el PASO 3.)

Se ejecuta con:  python 04_train_xgboost.py
"""

import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

AQUI = Path(__file__).parent
PROC = AQUI / "datasets" / "processed" / "trees"
MODELOS = AQUI / "artifacts" / "models"
METRICAS = AQUI / "artifacts" / "metrics"
PLOTS = AQUI / "artifacts" / "plots"
for d in (MODELOS, METRICAS, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "SeriousDlqin2yrs"
SEMILLA = 42


def titulo(t):
    print("\n" + "=" * 79)
    print(t)
    print("=" * 79)


def auc(modelo, X, y):
    return roc_auc_score(y, modelo.predict_proba(X)[:, 1])


# =============================================================================
# PASO 1 — CARGA Y VERIFICACION DEL FORMATO
# =============================================================================
titulo("PASO 1 — CARGA Y FORMATO")

train = pd.read_parquet(PROC / "train.parquet")
val = pd.read_parquet(PROC / "val.parquet")

y_tr_full = train[TARGET]
X_tr_full = train.drop(columns=[TARGET])
y_val = val[TARGET]
X_val = val.drop(columns=[TARGET])

print(f"  Entrenamiento : {X_tr_full.shape[0]:>7,} filas x {X_tr_full.shape[1]} variables")
print(f"  Validacion    : {X_val.shape[0]:>7,} filas x {X_val.shape[1]} variables")
print(f"  Default rate  : {y_tr_full.mean()*100:.2f}% | {y_val.mean()*100:.2f}%")

n_nan = X_tr_full.isna().sum()
print(f"\n  Columnas con NaN: {(n_nan > 0).sum()} de {X_tr_full.shape[1]}"
      f"  ({n_nan.sum():,} valores en total)")
print("  Se dejan tal cual: XGBoost aprende la direccion optima para cada uno.")

# --- Sub-split para EARLY STOPPING -------------------------------------------
# CUIDADO CON ESTO:
#   Si usaramos el conjunto de validacion para decidir cuando parar de anadir
#   arboles, estariamos AJUSTANDO a validacion. El AUC que reportaramos sobre
#   ese mismo conjunto seria optimista, porque ya lo habriamos usado para
#   tomar una decision.
#   Solucion: apartamos un trozo del propio TRAIN para el early stopping,
#   y validacion se queda intacta hasta el final.
X_fit, X_es, y_fit, y_es = train_test_split(
    X_tr_full, y_tr_full, test_size=0.2, stratify=y_tr_full, random_state=SEMILLA)

print(f"\n  Reparto interno del entrenamiento:")
print(f"    ajuste          : {len(X_fit):>7,} filas")
print(f"    early stopping  : {len(X_es):>7,} filas   <- decide cuantos arboles")
print(f"    validacion      : {len(X_val):>7,} filas   <- INTACTA hasta el final")


# =============================================================================
# PASO 2 — BASELINE CON PARAMETROS POR DEFECTO
# =============================================================================
titulo("PASO 2 — BASELINE (parametros por defecto)")

t0 = time.time()
base = XGBClassifier(
    n_estimators=100,
    random_state=SEMILLA,
    eval_metric="auc",
    n_jobs=-1,
)
base.fit(X_fit, y_fit, verbose=False)
print(f"  Entrenado en {time.time()-t0:.1f}s")
print(f"  AUC ajuste     : {auc(base, X_fit, y_fit):.5f}")
print(f"  AUC validacion : {auc(base, X_val, y_val):.5f}")
print(f"\n  Referencia — LogReg baseline: 0.86272")
print("  (con los defaults el arbol suele SOBREAJUSTAR: fijate en la brecha")
print("   entre ajuste y validacion)")


# =============================================================================
# PASO 3 — ¿HACEN FALTA LAS VARIABLES DERIVADAS?
# =============================================================================
titulo("PASO 3 — ¿APORTAN ALGO LAS VARIABLES DERIVADAS?")

# En la regresion logistica las 7 derivadas hubo que quitarlas: causaban
# multicolinealidad y 4 signos invertidos, a cambio de +0.0003 de AUC.
# Los arboles no sufren ese problema... pero ¿aportan realmente?
ORIGINALES = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio_limpio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
FLAGS = [c for c in X_tr_full.columns if c.startswith("flag_")]
DERIVADAS = [c for c in X_tr_full.columns if c not in ORIGINALES + FLAGS]

SUBCONJUNTOS = {
    "todas (25)":                    list(X_tr_full.columns),
    "originales + flags (18)":       ORIGINALES + FLAGS,
    "solo originales (10)":          ORIGINALES,
}

print(f"  Variables derivadas presentes: {DERIVADAS}\n")
# Se compara sobre el set de EARLY STOPPING, no sobre validacion.
# Mirar validacion para elegir un subconjunto la convertiria en parte del
# proceso de decision, y el AUC final que reportemos sobre ella ya no seria
# una estimacion limpia de "datos nunca vistos".
print(f"  {'subconjunto':<26} {'n':>4} {'AUC ajuste':>12} {'AUC early-stop':>15}")
print("  " + "-" * 61)
for nombre, cols in SUBCONJUNTOS.items():
    m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                      random_state=SEMILLA, eval_metric="auc", n_jobs=-1)
    m.fit(X_fit[cols], y_fit, verbose=False)
    print(f"  {nombre:<26} {len(cols):>4} {auc(m, X_fit[cols], y_fit):>12.5f} "
          f"{auc(m, X_es[cols], y_es):>15.5f}")

COLS = list(X_tr_full.columns)   # los arboles se quedan con todas
print(f"\n  Se usan las {len(COLS)} variables. A diferencia de LogReg, aqui no")
print("  hay coeficientes que puedan salir con el signo invertido: un arbol")
print("  elige UNA variable por division en vez de repartir peso entre varias.")


# =============================================================================
# PASO 4 — EARLY STOPPING: ¿CUANTOS ARBOLES?
# =============================================================================
titulo("PASO 4 — EARLY STOPPING")

# XGBoost construye arboles de forma secuencial: cada uno corrige los errores
# de los anteriores. Con demasiados, empieza a memorizar el ruido del train.
# Early stopping detiene el proceso cuando el AUC del conjunto de vigilancia
# deja de mejorar durante N rondas seguidas.
t0 = time.time()
es_modelo = XGBClassifier(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=4,
    random_state=SEMILLA,
    eval_metric="auc",
    early_stopping_rounds=100,
    n_jobs=-1,
)
es_modelo.fit(X_fit[COLS], y_fit, eval_set=[(X_es[COLS], y_es)], verbose=False)

print(f"  Entrenado en {time.time()-t0:.1f}s")
print(f"  Arboles usados : {es_modelo.best_iteration + 1} de 3000 permitidos")
print(f"  AUC en el set de early stopping: {es_modelo.best_score:.5f}")
print(f"  AUC en validacion              : {auc(es_modelo, X_val[COLS], y_val):.5f}")
print("\n  Se detuvo solo: anadir mas arboles ya no mejoraba nada.")


# =============================================================================
# PASO 5 — BUSQUEDA DE HIPERPARAMETROS
# =============================================================================
titulo("PASO 5 — BUSQUEDA DE HIPERPARAMETROS")

# A diferencia de LogReg (donde C no cambiaba nada), en XGBoost los
# hiperparametros SI importan mucho. Los principales:
#
#   max_depth        profundidad de cada arbol. Mas profundo = mas complejo.
#   learning_rate    cuanto corrige cada arbol. Bajo = mas lento pero mejor.
#   min_child_weight minimo de peso en una hoja. Alto = mas conservador.
#   subsample        % de filas por arbol. <1 introduce aleatoriedad util.
#   colsample_bytree % de columnas por arbol. Idem.
#   reg_lambda       regularizacion L2 sobre los pesos de las hojas.
#
# Evaluamos sobre el set de EARLY STOPPING, nunca sobre validacion.

REJILLA = [
    dict(max_depth=3, min_child_weight=1,  subsample=1.0, colsample_bytree=1.0),
    dict(max_depth=4, min_child_weight=1,  subsample=1.0, colsample_bytree=1.0),
    dict(max_depth=5, min_child_weight=1,  subsample=1.0, colsample_bytree=1.0),
    dict(max_depth=6, min_child_weight=1,  subsample=1.0, colsample_bytree=1.0),
    dict(max_depth=4, min_child_weight=5,  subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, min_child_weight=5,  subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, min_child_weight=20, subsample=0.8, colsample_bytree=0.6),
    dict(max_depth=7, min_child_weight=20, subsample=0.8, colsample_bytree=0.6),
    dict(max_depth=4, min_child_weight=10, subsample=0.9, colsample_bytree=0.9, reg_lambda=5),
    dict(max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=5),
]

print(f"  {'max_d':>5} {'min_child':>10} {'subsam':>7} {'colsam':>7} {'lambda':>7} "
      f"{'arboles':>8} {'AUC_es':>9} {'AUC_val':>9}")
print("  " + "-" * 72)

t0 = time.time()
resultados = []
for params in REJILLA:
    m = XGBClassifier(
        n_estimators=3000, learning_rate=0.05, random_state=SEMILLA,
        eval_metric="auc", early_stopping_rounds=100, n_jobs=-1, **params)
    m.fit(X_fit[COLS], y_fit, eval_set=[(X_es[COLS], y_es)], verbose=False)
    a_val = auc(m, X_val[COLS], y_val)
    resultados.append({**params, "n_arboles": m.best_iteration + 1,
                       "auc_es": m.best_score, "auc_val": a_val})
    print(f"  {params['max_depth']:>5} {params['min_child_weight']:>10} "
          f"{params['subsample']:>7} {params['colsample_bytree']:>7} "
          f"{params.get('reg_lambda', 1):>7} {m.best_iteration+1:>8} "
          f"{m.best_score:>9.5f} {a_val:>9.5f}")

print(f"\n  Busqueda completada en {time.time()-t0:.1f}s")

rej = pd.DataFrame(resultados)
# Elegimos por AUC del set de early stopping, NO por el de validacion:
# elegir por validacion seria ajustar a ella y el numero final mentiria.
mejor = rej.sort_values("auc_es", ascending=False).iloc[0]
MEJORES = {k: mejor[k] for k in
           ["max_depth", "min_child_weight", "subsample", "colsample_bytree"]}
MEJORES["max_depth"] = int(MEJORES["max_depth"])
MEJORES["min_child_weight"] = int(MEJORES["min_child_weight"])
if "reg_lambda" in rej.columns and not pd.isna(mejor.get("reg_lambda")):
    MEJORES["reg_lambda"] = float(mejor["reg_lambda"])

print(f"\n  ELEGIDO (por AUC del set de early stopping, no de validacion):")
for k, v in MEJORES.items():
    print(f"    {k:<20} = {v}")


# =============================================================================
# PASO 6 — ¿scale_pos_weight?
# =============================================================================
titulo("PASO 6 — EL DESBALANCE: ¿scale_pos_weight?")

# Es el equivalente de class_weight='balanced' en XGBoost: multiplica el peso
# de la clase minoritaria. Igual que alli, DEFORMA las probabilidades.
ratio = float((y_fit == 0).sum() / (y_fit == 1).sum())
for etiqueta, spw in [("sin ajustar (PD calibrada)", 1.0),
                      (f"scale_pos_weight={ratio:.1f}", ratio)]:
    m = XGBClassifier(n_estimators=3000, learning_rate=0.05, random_state=SEMILLA,
                      eval_metric="auc", early_stopping_rounds=100, n_jobs=-1,
                      scale_pos_weight=spw, **MEJORES)
    m.fit(X_fit[COLS], y_fit, eval_set=[(X_es[COLS], y_es)], verbose=False)
    p = m.predict_proba(X_val[COLS])[:, 1]
    print(f"  {etiqueta:<30} AUC_val = {roc_auc_score(y_val, p):.5f}   "
          f"PD media = {p.mean()*100:5.2f}%  (real: {y_val.mean()*100:.2f}%)")

print("\n  Igual que en LogReg: se usa SIN ajustar, para que las probabilidades")
print("  sigan siendo PD reales. El desbalance se maneja en el umbral.")


# =============================================================================
# PASO 7 — MODELO FINAL Y EVALUACION
# =============================================================================
titulo("PASO 7 — MODELO FINAL")

modelo = XGBClassifier(
    n_estimators=3000, learning_rate=0.05, random_state=SEMILLA,
    eval_metric="auc", early_stopping_rounds=100, n_jobs=-1, **MEJORES)
modelo.fit(X_fit[COLS], y_fit, eval_set=[(X_es[COLS], y_es)], verbose=False)

p_val = modelo.predict_proba(X_val[COLS])[:, 1]
auc_val = roc_auc_score(y_val, p_val)
fpr, tpr, _ = roc_curve(y_val, p_val)
ks = float(np.max(tpr - fpr))

print(f"  Arboles      : {modelo.best_iteration + 1}")
print(f"  AUC ajuste   : {auc(modelo, X_fit[COLS], y_fit):.5f}")
print(f"  AUC valid.   : {auc_val:.5f}")
print(f"  KS           : {ks:.5f}")

# --- Calibracion -------------------------------------------------------------
print("\n  --- CALIBRACION: ¿son creibles las probabilidades? ---")
d = pd.DataFrame({"p": p_val, "y": y_val.to_numpy()})
d["decil"] = pd.qcut(d.p, 10, labels=False)
cal = d.groupby("decil").agg(n=("y", "size"), pd_pred=("p", "mean"), pd_real=("y", "mean"))
cal["pd_pred"] = (cal.pd_pred * 100).round(2)
cal["pd_real"] = (cal.pd_real * 100).round(2)
cal["error_pp"] = (cal.pd_pred - cal.pd_real).round(2)
print(cal.to_string())
print(f"\n  Error absoluto medio: {cal.error_pp.abs().mean():.2f} puntos porcentuales")
print("  (LogReg logro 0.53 pp)")


# =============================================================================
# PASO 8 — IMPORTANCIA DE VARIABLES
# =============================================================================
titulo("PASO 8 — QUE VARIABLES USA EL MODELO")

imp = pd.DataFrame({
    "variable": COLS,
    "gain": modelo.feature_importances_,
}).sort_values("gain", ascending=False).reset_index(drop=True)
imp["pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(2)
print(imp.head(15).to_string(index=False))

print("\n  OJO al interpretar esto: la importancia se REPARTE entre variables")
print("  redundantes. Si total_atrasos y peor_atraso dicen casi lo mismo, cada")
print("  una recibe la mitad del credito, y ambas parecen menos importantes de")
print("  lo que realmente es esa informacion.")


# =============================================================================
# PASO 9 — COMPARACION HONESTA CON LOGREG
# =============================================================================
titulo("PASO 9 — ¿LE GANA A LA REGRESION LOGISTICA?")

# Un solo split tiene una variabilidad de ~±0.003 de AUC (lo medimos con
# cross-validation en LogReg). Comparar dos modelos con un unico numero es
# poco fiable: una diferencia menor a ~0.007 podria ser puro ruido.
# Por eso repetimos la evaluacion con 5 particiones distintas.
print("  Comparando con validacion cruzada de 5 particiones,")
print("  porque un solo split tiene ~±0.003 de ruido.\n")

todo_X = pd.concat([X_tr_full[COLS], X_val[COLS]])
todo_y = pd.concat([y_tr_full, y_val])
cv = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)

xgb_cv = XGBClassifier(
    n_estimators=int(modelo.best_iteration + 1), learning_rate=0.05,
    random_state=SEMILLA, eval_metric="auc", n_jobs=-1, **MEJORES)
scores_xgb = cross_val_score(xgb_cv, todo_X, todo_y, cv=cv, scoring="roc_auc")

print(f"  XGBoost  AUC por fold: {np.round(scores_xgb, 5)}")
print(f"           media = {scores_xgb.mean():.5f}   desv = {scores_xgb.std():.5f}")
print(f"  LogReg   media = 0.85948   desv = 0.00335   (medido en el paso anterior)")

dif = scores_xgb.mean() - 0.85948
print(f"\n  Diferencia: {dif:+.5f}")
if abs(dif) < 2 * scores_xgb.std():
    print("  -> Dentro del ruido: NO se puede afirmar que uno sea mejor.")
else:
    print(f"  -> Diferencia mayor que 2 desviaciones: XGBoost "
          f"{'gana' if dif > 0 else 'pierde'} de forma consistente.")


# =============================================================================
# PASO 10 — GRAFICAS Y GUARDADO
# =============================================================================
titulo("PASO 10 — GRAFICAS Y GUARDADO")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
ax.plot(fpr, tpr, color="#7a3a6a", lw=2, label=f"XGBoost (AUC = {auc_val:.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="azar")
i = int(np.argmax(tpr - fpr))
ax.plot([fpr[i], fpr[i]], [fpr[i], tpr[i]], color="#a8392f", lw=2, label=f"KS = {ks:.4f}")
ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos")
ax.set_title("Curva ROC — XGBoost"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(cal.index, cal.pd_pred, "o-", color="#7a3a6a", label="PD predicha")
ax.plot(cal.index, cal.pd_real, "s--", color="#2b6247", label="PD real")
ax.set_xlabel("Decil de riesgo"); ax.set_ylabel("% default")
ax.set_title("Calibracion: predicho vs real"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.hist(p_val[y_val == 0], bins=60, alpha=0.65, density=True, color="#2b6247",
        label="no cayo en mora")
ax.hist(p_val[y_val == 1], bins=60, alpha=0.65, density=True, color="#a8392f",
        label="si cayo en mora")
ax.set_xlim(0, 0.6); ax.set_xlabel("PD predicha"); ax.set_ylabel("Densidad")
ax.set_title("Separacion de las dos poblaciones"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
t = imp.head(12).iloc[::-1]
ax.barh(range(len(t)), t["gain"], color="#7a3a6a")
ax.set_yticks(range(len(t))); ax.set_yticklabels(t["variable"], fontsize=7)
ax.set_xlabel("Importancia (gain)"); ax.set_title("Variables mas usadas (top 12)")
ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(PLOTS / "04_xgboost_evaluacion.png", dpi=110)
plt.close()
print(f"  {(PLOTS / '04_xgboost_evaluacion.png').relative_to(AQUI)}")

joblib.dump({
    "modelo": modelo, "columnas": COLS, "hiperparametros": MEJORES,
    "n_arboles": int(modelo.best_iteration + 1), "semilla": SEMILLA,
}, MODELOS / "xgboost_v1.joblib")

pd.Series({
    "modelo": "XGBoost (datos crudos)",
    "n_variables": len(COLS),
    "n_arboles": int(modelo.best_iteration + 1),
    "auc_val": round(auc_val, 5),
    "auc_cv_media": round(float(scores_xgb.mean()), 5),
    "auc_cv_desv": round(float(scores_xgb.std()), 5),
    "ks_val": round(ks, 5),
}).to_json(METRICAS / "xgboost_v1.json", indent=2)
rej.to_csv(METRICAS / "xgboost_rejilla.csv", index=False)

print("  artifacts/models/xgboost_v1.joblib")
print("  artifacts/metrics/xgboost_v1.json")
print("  artifacts/metrics/xgboost_rejilla.csv")

print("\n" + "=" * 79)
print(f"XGBoost — AUC validacion = {auc_val:.5f} | CV = {scores_xgb.mean():.5f} "
      f"+/- {scores_xgb.std():.5f}")
print("=" * 79)

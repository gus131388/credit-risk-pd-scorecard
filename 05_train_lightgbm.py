# -*- coding: utf-8 -*-
r"""
05_train_lightgbm.py — LightGBM sobre los datos crudos
Proyecto: Give Me Some Credit — Probability of Default

Usa la MISMA rama 'trees' que XGBoost: valores crudos, NaN nativos y flags.

EN QUE SE DIFERENCIA DE XGBoost

Los dos son gradient boosting: construyen arboles en secuencia, y cada uno
corrige los errores acumulados de los anteriores. La diferencia esta en COMO
hacen crecer cada arbol.

  XGBoost crece por NIVELES (level-wise):
      completa un nivel entero antes de bajar al siguiente.
                    o
                  /   \
                 o     o          <- nivel 1 completo
                / \   / \
               o   o o   o        <- nivel 2 completo
      Arboles equilibrados. El parametro que manda es max_depth.

  LightGBM crece por HOJAS (leaf-wise):
      en cada paso parte la hoja que MAS reduce el error, este donde este.
                    o
                  /   \
                 o     o
                / \
               o   o
              / \
             o   o                <- arbol desequilibrado a proposito
      Consigue mas precision con menos hojas, pero puede sobreajustar con
      facilidad. El parametro que manda es num_leaves, no max_depth.

Consecuencia practica: no se puede copiar la configuracion de XGBoost.
Un max_depth=5 en XGBoost permite hasta 2^5 = 32 hojas equilibradas;
en LightGBM hay que fijar num_leaves de forma explicita.

Se ejecuta con:  python 05_train_lightgbm.py
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
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

AQUI = Path(__file__).parent
PROC = AQUI / "datasets" / "processed"
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
# PASO 1 — CARGA
# =============================================================================
titulo("PASO 1 — CARGA")

train = pd.read_parquet(PROC / "trees" / "train.parquet")
val = pd.read_parquet(PROC / "trees" / "val.parquet")

y_tr_full = train[TARGET]
X_tr_full = train.drop(columns=[TARGET])
y_val = val[TARGET]
X_val = val.drop(columns=[TARGET])
COLS = list(X_tr_full.columns)

# Mismo reparto y misma semilla que 04_train_xgboost.py -> comparacion justa.
X_fit, X_es, y_fit, y_es = train_test_split(
    X_tr_full, y_tr_full, test_size=0.2, stratify=y_tr_full, random_state=SEMILLA)

print(f"  Variables      : {len(COLS)}  (rama 'trees', igual que XGBoost)")
print(f"  ajuste         : {len(X_fit):>7,} filas")
print(f"  early stopping : {len(X_es):>7,} filas")
print(f"  validacion     : {len(X_val):>7,} filas   <- INTACTA hasta el final")
print(f"\n  Mismo split y misma semilla que 04_train_xgboost.py:")
print("  cualquier diferencia de AUC vendra del algoritmo, no del reparto.")


# =============================================================================
# PASO 2 — BASELINE
# =============================================================================
titulo("PASO 2 — BASELINE (parametros por defecto)")

t0 = time.time()
base = LGBMClassifier(n_estimators=100, random_state=SEMILLA, n_jobs=-1, verbose=-1)
base.fit(X_fit, y_fit)
t_lgb = time.time() - t0

t0 = time.time()
base_xgb = XGBClassifier(n_estimators=100, random_state=SEMILLA,
                         eval_metric="auc", n_jobs=-1)
base_xgb.fit(X_fit, y_fit, verbose=False)
t_xgb = time.time() - t0

print(f"  {'':<12} {'tiempo':>9} {'AUC ajuste':>12} {'AUC val':>10}")
print("  " + "-" * 46)
print(f"  {'LightGBM':<12} {t_lgb:>8.1f}s {auc(base, X_fit, y_fit):>12.5f} "
      f"{auc(base, X_val, y_val):>10.5f}")
print(f"  {'XGBoost':<12} {t_xgb:>8.1f}s {auc(base_xgb, X_fit, y_fit):>12.5f} "
      f"{auc(base_xgb, X_val, y_val):>10.5f}")
print(f"\n  num_leaves por defecto en LightGBM: 31")
print("  (fijate en la brecha ajuste/validacion: por defecto tambien sobreajusta)")


# =============================================================================
# PASO 3 — EARLY STOPPING
# =============================================================================
titulo("PASO 3 — EARLY STOPPING")

t0 = time.time()
es = LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=31,
                    random_state=SEMILLA, n_jobs=-1, verbose=-1)
es.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
       callbacks=[lgb.early_stopping(100, verbose=False)])

print(f"  Entrenado en {time.time()-t0:.1f}s")
print(f"  Arboles usados : {es.best_iteration_} de 3000 permitidos")
print(f"  AUC early-stop : {es.best_score_['valid_0']['auc']:.5f}")
print(f"  AUC validacion : {auc(es, X_val, y_val):.5f}")
print("\n  Referencia XGBoost: 105 arboles, AUC validacion 0.86872")


# =============================================================================
# PASO 4 — BUSQUEDA DE HIPERPARAMETROS
# =============================================================================
titulo("PASO 4 — BUSQUEDA DE HIPERPARAMETROS")

# Los parametros clave de LightGBM y su equivalente aproximado en XGBoost:
#
#   num_leaves         (no existe en XGBoost) el mas importante aqui.
#                      Controla la complejidad directamente. Mas hojas =
#                      mas capacidad de memorizar.
#   min_child_samples  ~ min_child_weight. Minimo de filas por hoja.
#   feature_fraction   ~ colsample_bytree. % de columnas por arbol.
#   bagging_fraction   ~ subsample. % de filas por arbol.
#   reg_lambda         ~ reg_lambda. Regularizacion L2.
#
# Se evalua sobre el set de EARLY STOPPING. Validacion no participa.

REJILLA = [
    dict(num_leaves=15,  min_child_samples=20),
    dict(num_leaves=31,  min_child_samples=20),
    dict(num_leaves=63,  min_child_samples=20),
    dict(num_leaves=127, min_child_samples=20),
    dict(num_leaves=31,  min_child_samples=100),
    dict(num_leaves=63,  min_child_samples=100),
    dict(num_leaves=63,  min_child_samples=200, feature_fraction=0.6, bagging_fraction=0.8),
    dict(num_leaves=31,  min_child_samples=100, feature_fraction=0.6, bagging_fraction=0.8),
    dict(num_leaves=15,  min_child_samples=50,  feature_fraction=0.8, bagging_fraction=0.8),
    dict(num_leaves=63,  min_child_samples=100, feature_fraction=0.6, bagging_fraction=0.8, reg_lambda=5),
    dict(num_leaves=127, min_child_samples=200, feature_fraction=0.6, bagging_fraction=0.8, reg_lambda=10),
]

print(f"  {'leaves':>7} {'min_child':>10} {'feat_f':>7} {'bag_f':>6} {'lambda':>7} "
      f"{'arboles':>8} {'AUC_es':>9} {'AUC_val':>9}")
print("  " + "-" * 72)

t0 = time.time()
resultados = []
for params in REJILLA:
    p = dict(num_leaves=31, min_child_samples=20, feature_fraction=1.0,
             bagging_fraction=1.0, reg_lambda=0.0)
    p.update(params)
    m = LGBMClassifier(n_estimators=3000, learning_rate=0.05, random_state=SEMILLA,
                       n_jobs=-1, verbose=-1, bagging_freq=1, **p)
    m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    a_es = m.best_score_["valid_0"]["auc"]
    a_val = auc(m, X_val, y_val)
    resultados.append({**p, "n_arboles": m.best_iteration_, "auc_es": a_es, "auc_val": a_val})
    print(f"  {p['num_leaves']:>7} {p['min_child_samples']:>10} "
          f"{p['feature_fraction']:>7} {p['bagging_fraction']:>6} {p['reg_lambda']:>7} "
          f"{m.best_iteration_:>8} {a_es:>9.5f} {a_val:>9.5f}")

print(f"\n  Busqueda completada en {time.time()-t0:.1f}s")

rej = pd.DataFrame(resultados)
mejor = rej.sort_values("auc_es", ascending=False).iloc[0]
MEJORES = dict(
    num_leaves=int(mejor["num_leaves"]),
    min_child_samples=int(mejor["min_child_samples"]),
    feature_fraction=float(mejor["feature_fraction"]),
    bagging_fraction=float(mejor["bagging_fraction"]),
    reg_lambda=float(mejor["reg_lambda"]),
)
print("\n  ELEGIDO (por AUC del set de early stopping):")
for k, v in MEJORES.items():
    print(f"    {k:<20} = {v}")


# =============================================================================
# PASO 5 — MODELO FINAL
# =============================================================================
titulo("PASO 5 — MODELO FINAL")

modelo = LGBMClassifier(n_estimators=3000, learning_rate=0.05, random_state=SEMILLA,
                        n_jobs=-1, verbose=-1, bagging_freq=1, **MEJORES)
modelo.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], eval_metric="auc",
           callbacks=[lgb.early_stopping(100, verbose=False)])

p_val = modelo.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, p_val)
fpr, tpr, _ = roc_curve(y_val, p_val)
ks = float(np.max(tpr - fpr))

print(f"  Arboles      : {modelo.best_iteration_}")
print(f"  AUC ajuste   : {auc(modelo, X_fit, y_fit):.5f}")
print(f"  AUC valid.   : {auc_val:.5f}")
print(f"  KS           : {ks:.5f}")
print(f"  PD media     : {p_val.mean()*100:.2f}%   (real: {y_val.mean()*100:.2f}%)")

print("\n  --- CALIBRACION ---")
d = pd.DataFrame({"p": p_val, "y": y_val.to_numpy()})
d["decil"] = pd.qcut(d.p, 10, labels=False)
cal = d.groupby("decil").agg(n=("y", "size"), pd_pred=("p", "mean"), pd_real=("y", "mean"))
cal["pd_pred"] = (cal.pd_pred * 100).round(2)
cal["pd_real"] = (cal.pd_real * 100).round(2)
cal["error_pp"] = (cal.pd_pred - cal.pd_real).round(2)
print(cal.to_string())
err_cal = cal.error_pp.abs().mean()
print(f"\n  Error absoluto medio: {err_cal:.2f} pp   "
      f"(LogReg 0.53 | XGBoost 0.34)")


# =============================================================================
# PASO 6 — IMPORTANCIA DE VARIABLES
# =============================================================================
titulo("PASO 6 — IMPORTANCIA DE VARIABLES")

imp = pd.DataFrame({"variable": COLS, "gain": modelo.booster_.feature_importance("gain")})
imp["pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(2)
imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
print(imp.head(12).to_string(index=False))
print("\n  Recordatorio: la importancia se REPARTE entre variables redundantes.")
print("  Un 30% no significa 'sin esto el modelo cae'.")


# =============================================================================
# PASO 7 — COMPARACION DE LOS TRES MODELOS
# =============================================================================
titulo("PASO 7 — LOS TRES MODELOS, COMPARADOS CON VALIDACION CRUZADA")

# Un solo split tiene ~±0.003 de ruido: comparar tres modelos con un unico
# numero cada uno no permite concluir nada. Repetimos con 5 particiones
# IDENTICAS para los tres (misma semilla, mismas filas en cada fold).
print("  5 particiones identicas para los tres modelos.\n")

cv = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)

trees_X = pd.concat([X_tr_full, X_val])
trees_y = pd.concat([y_tr_full, y_val])

woe_tr = pd.read_parquet(PROC / "woe" / "train.parquet")
woe_va = pd.read_parquet(PROC / "woe" / "val.parquet")
woe_all = pd.concat([woe_tr, woe_va])
art_log = joblib.load(MODELOS / "logistic_v1.joblib")
woe_X = woe_all[art_log["columnas"]]
woe_y = woe_all[TARGET]

art_xgb = joblib.load(MODELOS / "xgboost_v1.joblib")

comparativa = {}
t0 = time.time()

comparativa["LogReg (WoE)"] = cross_val_score(
    LogisticRegression(max_iter=5000, random_state=SEMILLA),
    woe_X, woe_y, cv=cv, scoring="roc_auc")

comparativa["XGBoost"] = cross_val_score(
    XGBClassifier(n_estimators=art_xgb["n_arboles"], learning_rate=0.05,
                  random_state=SEMILLA, eval_metric="auc", n_jobs=-1,
                  **art_xgb["hiperparametros"]),
    trees_X, trees_y, cv=cv, scoring="roc_auc")

comparativa["LightGBM"] = cross_val_score(
    LGBMClassifier(n_estimators=modelo.best_iteration_, learning_rate=0.05,
                   random_state=SEMILLA, n_jobs=-1, verbose=-1, bagging_freq=1,
                   **MEJORES),
    trees_X, trees_y, cv=cv, scoring="roc_auc")

print(f"  (calculado en {time.time()-t0:.1f}s)\n")
print(f"  {'modelo':<16} {'media':>9} {'desv':>9} {'min':>9} {'max':>9}")
print("  " + "-" * 56)
tabla = []
for n, s in comparativa.items():
    print(f"  {n:<16} {s.mean():>9.5f} {s.std():>9.5f} {s.min():>9.5f} {s.max():>9.5f}")
    tabla.append({"modelo": n, "auc_cv_media": round(float(s.mean()), 5),
                  "auc_cv_desv": round(float(s.std()), 5)})

mejor_nombre = max(comparativa, key=lambda k: comparativa[k].mean())
print(f"\n  MEJOR: {mejor_nombre} ({comparativa[mejor_nombre].mean():.5f})")

print("\n  ¿Son diferencias reales o ruido?")
nombres = list(comparativa)
for i in range(len(nombres)):
    for j in range(i + 1, len(nombres)):
        a, b = comparativa[nombres[i]], comparativa[nombres[j]]
        dif = a.mean() - b.mean()
        # Diferencias emparejadas: los folds son los mismos para ambos
        d_folds = a - b
        umbral = 2 * d_folds.std() / np.sqrt(len(d_folds))
        veredicto = "REAL" if abs(dif) > umbral else "dentro del ruido"
        print(f"    {nombres[i]:<16} vs {nombres[j]:<16} {dif:+.5f}   {veredicto}")


# =============================================================================
# PASO 8 — GRAFICAS Y GUARDADO
# =============================================================================
titulo("PASO 8 — GRAFICAS Y GUARDADO")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
for nombre, art, X_, color in [
        ("LogReg", art_log, woe_va[art_log["columnas"]], "#3a4f7a"),
        ("XGBoost", art_xgb, X_val[art_xgb["columnas"]], "#7a3a6a")]:
    pp = art["modelo"].predict_proba(X_)[:, 1]
    f, t, _ = roc_curve(y_val, pp)
    ax.plot(f, t, lw=1.6, color=color, alpha=0.85,
            label=f"{nombre} ({roc_auc_score(y_val, pp):.4f})")
ax.plot(fpr, tpr, lw=2.2, color="#1f7a5a", label=f"LightGBM ({auc_val:.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="azar")
ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos")
ax.set_title("Curvas ROC — los tres modelos"); ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(cal.index, cal.pd_pred, "o-", color="#1f7a5a", label="PD predicha")
ax.plot(cal.index, cal.pd_real, "s--", color="#a8392f", label="PD real")
ax.set_xlabel("Decil de riesgo"); ax.set_ylabel("% default")
ax.set_title(f"Calibracion LightGBM (error {err_cal:.2f} pp)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
pos = np.arange(len(comparativa))
medias = [comparativa[n].mean() for n in comparativa]
errs = [comparativa[n].std() for n in comparativa]
ax.bar(pos, medias, yerr=errs, capsize=6,
       color=["#3a4f7a", "#7a3a6a", "#1f7a5a"])
ax.set_xticks(pos); ax.set_xticklabels(list(comparativa), fontsize=9)
ax.set_ylim(min(medias) - 0.012, max(medias) + 0.008)
ax.set_ylabel("ROC-AUC (validacion cruzada)")
ax.set_title("Comparacion con barras de error (5 folds)")
ax.grid(alpha=0.3, axis="y")
for x, m in zip(pos, medias):
    ax.text(x, m, f"{m:.5f}", ha="center", va="bottom", fontsize=8)

ax = axes[1, 1]
t = imp.head(12).iloc[::-1]
ax.barh(range(len(t)), t["gain"], color="#1f7a5a")
ax.set_yticks(range(len(t))); ax.set_yticklabels(t["variable"], fontsize=7)
ax.set_xlabel("Importancia (gain)"); ax.set_title("Variables mas usadas — LightGBM")
ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(PLOTS / "05_lightgbm_evaluacion.png", dpi=110)
plt.close()
print(f"  {(PLOTS / '05_lightgbm_evaluacion.png').relative_to(AQUI)}")

joblib.dump({"modelo": modelo, "columnas": COLS, "hiperparametros": MEJORES,
             "n_arboles": int(modelo.best_iteration_), "semilla": SEMILLA},
            MODELOS / "lightgbm_v1.joblib")

pd.Series({
    "modelo": "LightGBM (datos crudos)",
    "n_variables": len(COLS),
    "n_arboles": int(modelo.best_iteration_),
    "auc_val": round(auc_val, 5),
    "auc_cv_media": round(float(comparativa["LightGBM"].mean()), 5),
    "auc_cv_desv": round(float(comparativa["LightGBM"].std()), 5),
    "ks_val": round(ks, 5),
    "error_calibracion_pp": round(float(err_cal), 3),
}).to_json(METRICAS / "lightgbm_v1.json", indent=2)
pd.DataFrame(tabla).to_csv(METRICAS / "comparacion_modelos_cv.csv", index=False)
rej.to_csv(METRICAS / "lightgbm_rejilla.csv", index=False)

print("  artifacts/models/lightgbm_v1.joblib")
print("  artifacts/metrics/lightgbm_v1.json")
print("  artifacts/metrics/comparacion_modelos_cv.csv")

print("\n" + "=" * 79)
print(f"LightGBM — AUC validacion = {auc_val:.5f} | "
      f"CV = {comparativa['LightGBM'].mean():.5f} +/- {comparativa['LightGBM'].std():.5f}")
print("=" * 79)

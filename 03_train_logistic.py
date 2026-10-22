# -*- coding: utf-8 -*-
"""
03_train_logistic.py — Regresion logistica sobre WoE (modelo baseline)
Proyecto: Give Me Some Credit — Probability of Default

Este es el BASELINE: el numero de referencia contra el que compararemos
XGBoost y LightGBM. Sin baseline, decir que un modelo "es bueno" no
significa nada.

Se ejecuta con:  python 03_train_logistic.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, average_precision_score)

sys.path.insert(0, str(Path(__file__).parent))
from src.evaluation import (ks_statistic, gini, tabla_calibracion,
                            error_calibracion, signos_incoherentes)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

AQUI = Path(__file__).parent
PROC = AQUI / "datasets" / "processed" / "woe"
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


# =============================================================================
# PASO 1 — CARGA
# =============================================================================
titulo("PASO 1 — CARGA DE LOS DATASETS WoE")

train = pd.read_parquet(PROC / "train.parquet")
val = pd.read_parquet(PROC / "val.parquet")

y_tr = train[TARGET]
X_tr = train.drop(columns=[TARGET])
y_val = val[TARGET]
X_val = val.drop(columns=[TARGET])

print(f"  Entrenamiento : {X_tr.shape[0]:>7,} filas x {X_tr.shape[1]} variables")
print(f"  Validacion    : {X_val.shape[0]:>7,} filas x {X_val.shape[1]} variables")
print(f"  Tasa de default: {y_tr.mean()*100:.2f}% (train) | {y_val.mean()*100:.2f}% (val)")
print("\n  Nota: no hace falta escalar. El WoE ya deja todas las variables")
print("  en la misma escala (log-odds), que es justo la que usa LogReg.")


# =============================================================================
# PASO 2 — DIAGNOSTICO DE MULTICOLINEALIDAD (VIF)
# =============================================================================
titulo("PASO 2 — MULTICOLINEALIDAD: VARIANCE INFLATION FACTOR")

# VIF mide cuanto se "infla" la incertidumbre del coeficiente de una variable
# por culpa de las demas. Se calcula regresando cada variable contra TODAS
# las otras: si las demas la predicen bien (R2 alto), es redundante.
#
#     VIF_i = 1 / (1 - R2_i)
#
#     VIF = 1    -> independiente de las demas
#     VIF > 5    -> empieza a preocupar
#     VIF > 10   -> redundancia seria
#
# Lo calculamos SOLO sobre train: es un diagnostico del modelo, no del dato.

def calcular_vif(X):
    filas = []
    valores = X.to_numpy()
    for i, col in enumerate(X.columns):
        otras = np.delete(valores, i, axis=1)
        objetivo = valores[:, i]
        r2 = LinearRegression().fit(otras, objetivo).score(otras, objetivo)
        # r2 = 1 exacto significa redundancia perfecta -> VIF infinito
        vif = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
        filas.append({"variable": col, "R2": round(r2, 4), "VIF": round(vif, 2)})
    return pd.DataFrame(filas).sort_values("VIF", ascending=False).reset_index(drop=True)

vif_todas = calcular_vif(X_tr)
print("  Con TODAS las variables:\n")
print(vif_todas.to_string(index=False))

problematicas = vif_todas[vif_todas["VIF"] > 5]
print(f"\n  Variables con VIF > 5: {len(problematicas)}")


# =============================================================================
# PASO 3 — COMPARAR SUBCONJUNTOS DE VARIABLES
# =============================================================================
titulo("PASO 3 — QUE SUBCONJUNTO USAR")

# Las 10 variables ORIGINALES del dataset (ya en escala WoE).
ORIGINALES = [
    "woe_RevolvingUtilizationOfUnsecuredLines",
    "woe_age",
    "woe_NumberOfTime30-59DaysPastDueNotWorse",
    "woe_DebtRatio_limpio",
    "woe_MonthlyIncome",
    "woe_NumberOfOpenCreditLinesAndLoans",
    "woe_NumberOfTimes90DaysLate",
    "woe_NumberRealEstateLoansOrLines",
    "woe_NumberOfTime60-89DaysPastDueNotWorse",
    "woe_NumberOfDependents",
]
FLAGS = [c for c in X_tr.columns if c.startswith("flag_")]

# Las features DERIVADAS que construimos en crear_features(). Todas son
# funciones de las originales, asi que compiten con sus propias variables
# padre: ingreso_por_persona con MonthlyIncome y NumberOfDependents,
# prop_inmobiliaria con RealEstateLoans y OpenCreditLines, etc.
DERIVADAS = [c for c in X_tr.columns if c not in ORIGINALES + FLAGS]

def sin(cols):
    return [c for c in X_tr.columns if c not in cols]

# Indicadores de CALIDAD DE DATO. Cada uno marca exactamente las mismas filas
# que van al bin FALTANTE de su variable WoE correspondiente -> colinealidad
# perfecta. Es la causa raiz del hallazgo H1 de la auditoria.
FLAGS_CALIDAD = [
    "flag_income_faltante", "flag_income_cero", "flag_dependents_faltante",
    "flag_registro_invalido", "flag_revutil_invalido", "flag_edad_invalida",
]

CANDIDATOS = {
    "A: todas":                             list(X_tr.columns),
    "B: sin flag_algun, con total_atrasos": sin(["woe_peor_atraso", "flag_algun_atraso"]),
    "C: sin flag_algun, con peor_atraso":   sin(["woe_total_atrasos", "flag_algun_atraso"]),
    "D: sin derivadas de morosidad":        sin(["woe_total_atrasos", "woe_peor_atraso",
                                                 "flag_algun_atraso"]),
    "E: originales + todos los flags":      ORIGINALES + FLAGS,
    "F: sin flags de calidad de dato":      sin(FLAGS_CALIDAD),
    "G: originales + flag_algun_atraso":    ORIGINALES + ["flag_algun_atraso"],
    "H: solo las 10 originales":            ORIGINALES,
}

# --- Set de SELECCION, apartado del propio entrenamiento --------------------
# POR QUE NO USAMOS VALIDACION PARA ELEGIR:
#   Si comparamos subconjuntos mirando el AUC de validacion y nos quedamos con
#   el mejor, validacion pasa a formar parte de la DECISION. El numero que
#   reportemos luego sobre ella ya no seria una estimacion limpia de "datos
#   nunca vistos": estaria inflado por haberla usado para elegir.
#   Solucion (la misma que usa 04_train_xgboost.py): apartamos un trozo del
#   train para decidir, y validacion queda intacta hasta el final.
X_fit, X_sel, y_fit, y_sel = train_test_split(
    X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=SEMILLA)

print(f"\n  Reparto interno del entrenamiento:")
print(f"    ajuste     : {len(X_fit):>7,} filas")
print(f"    seleccion  : {len(X_sel):>7,} filas   <- decide el subconjunto")
print(f"    validacion : {len(X_val):>7,} filas   <- INTACTA hasta el final\n")


def entrenar(cols, y_train=y_tr, balanceado=False, X_train=None):
    """Entrena sobre X_tr completo por defecto; X_train permite usar el sub-split."""
    if X_train is None:
        X_train, y_train = X_tr, y_tr
    modelo = LogisticRegression(
        max_iter=2000,
        random_state=SEMILLA,
        class_weight="balanced" if balanceado else None,
    )
    modelo.fit(X_train[cols], y_train)
    return modelo

resultados = []
for nombre, cols in CANDIDATOS.items():
    # Se entrena sobre el sub-split de ajuste y se evalua sobre el de seleccion.
    # Validacion NO participa en esta decision.
    m = entrenar(cols, y_train=y_fit, X_train=X_fit)
    auc_fit = roc_auc_score(y_fit, m.predict_proba(X_fit[cols])[:, 1])
    auc_sel = roc_auc_score(y_sel, m.predict_proba(X_sel[cols])[:, 1])
    vif_max = calcular_vif(X_fit[cols])["VIF"].max()
    # Comprobacion COMPLETA: variables woe_ contra la convencion de signo, e
    # indicadores contra su direccion univariada medida en los datos.
    inc = signos_incoherentes(m.coef_[0], cols, X_fit, y_fit)
    resultados.append({
        "subconjunto": nombre,
        "n_vars": len(cols),
        "AUC_ajuste": round(auc_fit, 5),
        "AUC_seleccion": round(auc_sel, 5),
        "VIF_max": round(vif_max, 1),
        "incoherentes": len(inc),
        "_detalle": inc,
    })

comparacion = pd.DataFrame(resultados).drop(columns=["_detalle"])
print(comparacion.to_string(index=False))

for r in resultados:
    if r["_detalle"]:
        print(f"    {r['subconjunto']}: {r['_detalle']}")

print("\n  Lectura:")
print("   - AUC_seleccion es el criterio (validacion no se toca todavia)")
print("   - VIF_max alto     = coeficientes inestables aunque el AUC sea bueno")
print("   - incoherentes > 0 = hay coeficientes que CONTRADICEN la direccion")
print("                        univariada de su variable. En un scorecard eso")
print("                        significa dar puntos a favor a un grupo de alto")
print("                        riesgo: indefendible ante un regulador.")

# CRITERIO DE SELECCION
#   Para un scorecard, un coeficiente que contradice la evidencia univariada
#   es un DEFECTO, no un detalle: hay que poder explicarle a un supervisor
#   (y al cliente rechazado) por que el modelo decidio lo que decidio.
#   Por eso exigimos PRIMERO coherencia total, y solo entre los candidatos
#   validos maximizamos AUC.
#
#   HALLAZGO H1 (auditoria): la version anterior solo revisaba las variables
#   woe_ y eximia a los indicadores. Eso dejo pasar tres indicadores con el
#   signo al reves; el peor, flag_registro_invalido, identifica al grupo con
#   54.67% de eventos (vs 6.60% de la poblacion) y su coeficiente negativo le
#   asignaba PUNTOS POSITIVOS en el scorecard.
#   Causa raiz: cada indicador de calidad marca exactamente las mismas filas
#   que van al bin FALTANTE de su variable WoE -> colinealidad perfecta. En un
#   scorecard WoE los indicadores de faltante son redundantes por construccion.
validos = comparacion[comparacion["incoherentes"] == 0]
if validos.empty:
    print("\n  AVISO: ningun subconjunto logra coherencia total.")
    validos = comparacion
mejor = validos.sort_values(["AUC_seleccion", "VIF_max"], ascending=[False, True]).iloc[0]

NOMBRE_ELEGIDO = mejor["subconjunto"]
COLS = CANDIDATOS[NOMBRE_ELEGIDO]

mejor_auc_bruto = comparacion["AUC_seleccion"].max()
print(f"\n  ELEGIDO -> {NOMBRE_ELEGIDO}  ({len(COLS)} variables)")
print(f"  Coste de exigir signos coherentes: "
      f"{mejor_auc_bruto - mejor['AUC_seleccion']:+.5f} de AUC "
      f"({(mejor_auc_bruto - mejor['AUC_seleccion'])/mejor_auc_bruto*100:.3f}%)")
print(f"  Variables derivadas descartadas: {len([d for d in DERIVADAS if d not in COLS])} de {len(DERIVADAS)}")
print("\n  El modelo final se reentrena con el train COMPLETO (ajuste + seleccion).")


# =============================================================================
# PASO 4 — ENTRENAR EL MODELO FINAL
# =============================================================================
titulo("PASO 4 — MODELO FINAL")

# ¿Usar class_weight='balanced'?
#   Con 6.68% de eventos, "balanced" hace que el modelo preste mas atencion
#   a la clase minoritaria. Pero DEFORMA las probabilidades: dejan de ser
#   una PD real y pasan a ser puntuaciones relativas.
#   Para un scorecard queremos PD calibrada, asi que comparamos ambas.
for etiqueta, bal in [("sin balancear (PD calibrada)", False),
                      ("class_weight='balanced'", True)]:
    m = entrenar(COLS, balanceado=bal)
    p_val = m.predict_proba(X_val[COLS])[:, 1]
    print(f"  {etiqueta:<32} AUC_val = {roc_auc_score(y_val, p_val):.5f}   "
          f"PD media predicha = {p_val.mean()*100:5.2f}%  "
          f"(real: {y_val.mean()*100:.2f}%)")

modelo = entrenar(COLS, balanceado=False)
p_tr = modelo.predict_proba(X_tr[COLS])[:, 1]
p_val = modelo.predict_proba(X_val[COLS])[:, 1]

print("\n  Se usa el modelo SIN balancear: sus probabilidades son PD reales,")
print("  que es lo que necesita un scorecard. El desbalance se maneja")
print("  eligiendo el umbral, no deformando el modelo.")


# =============================================================================
# PASO 5 — EVALUACION
# =============================================================================
titulo("PASO 5 — EVALUACION EN VALIDACION")

auc_tr = roc_auc_score(y_tr, p_tr)
auc_val = roc_auc_score(y_val, p_val)

fpr, tpr, umbrales_roc = roc_curve(y_val, p_val)
ks = np.max(tpr - fpr)
ks_umbral = umbrales_roc[np.argmax(tpr - fpr)]
ap = average_precision_score(y_val, p_val)

print("  --- ROC-AUC ---")
print(f"    Train      : {auc_tr:.5f}")
print(f"    Validacion : {auc_val:.5f}")
print(f"    Diferencia : {auc_tr-auc_val:+.5f}  (si fuera grande -> sobreajuste)")
print("\n    Interpretacion: tomando al azar un moroso y un no-moroso,")
print(f"    el modelo le da mas riesgo al moroso el {auc_val*100:.1f}% de las veces.")

print("\n  --- KS (Kolmogorov-Smirnov) ---")
print(f"    KS = {ks:.4f}  en umbral {ks_umbral:.4f}")
print("    Referencia bancaria: >0.30 aceptable | >0.40 bueno")

print("\n  --- Average Precision (area bajo la curva precision-recall) ---")
print(f"    AP = {ap:.4f}   (una linea base al azar daria {y_val.mean():.4f})")

# --- El problema del umbral 0.5 ---------------------------------------------
print("\n  --- POR QUE EL UMBRAL 0.5 NO SIRVE AQUI ---")
pred_05 = (p_val >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, pred_05).ravel()
print(f"    Con umbral 0.5 el modelo marca como moroso a {pred_05.sum():,} de {len(y_val):,} personas")
print(f"    Detecta {tp:,} de los {int(y_val.sum()):,} morosos reales "
      f"-> recall = {tp/(tp+fn)*100:.1f}%")
print(f"    Accuracy = {(tn+tp)/len(y_val)*100:.2f}%  ...pero un modelo que diga")
print(f"    'nadie cae en mora' logra {(1-y_val.mean())*100:.2f}% sin detectar a NADIE.")

# --- Barrido de umbrales -----------------------------------------------------
print("\n  --- EL TRADE-OFF PRECISION / RECALL ---")
print(f"    {'umbral':>7} | {'marcados':>9} | {'precision':>9} | {'recall':>7} | {'F1':>6}")
print("    " + "-" * 52)
for u in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50]:
    pred = (p_val >= u).astype(int)
    if pred.sum() == 0:
        continue
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"    {u:>7.2f} | {pred.sum():>9,} | {prec*100:>8.1f}% | {rec*100:>6.1f}% | {f1:>6.3f}")

print("\n    Cada fila es una decision de negocio distinta. Bajar el umbral")
print("    detecta mas morosos (recall sube) pero rechaza mas clientes buenos")
print("    (precision baja). No existe el umbral 'correcto' en abstracto:")
print("    depende de cuanto cuesta cada error.")

# --- Matriz de confusion en un umbral operativo ------------------------------
UMBRAL_OP = 0.10
pred = (p_val >= UMBRAL_OP).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
print(f"\n  --- MATRIZ DE CONFUSION (umbral {UMBRAL_OP}) ---")
print(f"                          modelo: NO cae    modelo: SI cae")
print(f"    realidad: no cayo     {tn:>10,} VN    {fp:>10,} FP")
print(f"    realidad: si cayo     {fn:>10,} FN    {tp:>10,} VP")
print(f"\n    De {tp+fn:,} morosos reales, detectamos {tp:,} ({tp/(tp+fn)*100:.1f}%)")
print(f"    De {tp+fp:,} personas marcadas, {tp:,} eran morosas de verdad "
      f"({tp/(tp+fp)*100:.1f}%)")
print(f"    Rechazamos por error a {fp:,} clientes buenos")


# =============================================================================
# PASO 6 — COEFICIENTES
# =============================================================================
titulo("PASO 6 — COEFICIENTES DEL MODELO")

coefs = pd.DataFrame({
    "variable": COLS,
    "coeficiente": modelo.coef_[0],
}).sort_values("coeficiente", key=abs, ascending=False).reset_index(drop=True)

# En variables WoE el signo esperado es NEGATIVO: WoE alto = seguro,
# asi que deberia BAJAR la probabilidad de default.
coefs["signo_esperado"] = np.where(
    coefs["variable"].str.startswith("woe_"), "negativo", "libre")
coefs["ok"] = np.where(
    (coefs["signo_esperado"] == "negativo") & (coefs["coeficiente"] > 0),
    "<-- SIGNO INVERTIDO", "")

print(f"  Intercepto: {modelo.intercept_[0]:.4f}")
print()
print(coefs.round(4).to_string(index=False))

invertidos = (coefs["ok"] != "").sum()
print(f"\n  Coeficientes con signo invertido: {invertidos}")
if invertidos:
    print("  (sintoma de multicolinealidad residual: dos variables se reparten")
    print("   el efecto y una acaba compensando a la otra)")



# =============================================================================
# PASO 7 — DIAGNOSTICOS: ¿ESTA EL MODELO REALMENTE BIEN?
# =============================================================================
titulo("PASO 7 — DIAGNOSTICOS")

# --- 7a. ¿Convergio el optimizador? -----------------------------------------
# Si llegara al tope de iteraciones, no habria encontrado el optimo y los
# coeficientes no serian de fiar.
print(f"  [1] CONVERGENCIA")
print(f"      Iteraciones usadas: {modelo.n_iter_[0]} de max_iter=2000")
print(f"      -> {'convergio' if modelo.n_iter_[0] < 2000 else 'NO CONVERGIO — revisar'}")

# --- 7b. Calibracion --------------------------------------------------------
# El AUC solo mide si el modelo ORDENA bien. La calibracion pregunta si las
# probabilidades son creibles. De ella depende que la regla de negocio
# umbral = M/(M+L) funcione: sin PD reales, el modelo ordena pero no dice
# cuanto prestar.
print(f"\n  [2] CALIBRACION — si dice '8% de riesgo', ¿falla el 8% de ese grupo?")
cal = tabla_calibracion(y_val, p_val)
print(cal.to_string())
err_cal = error_calibracion(y_val, p_val)
print(f"\n      Error absoluto medio: {err_cal:.2f} puntos porcentuales")

# --- 7c. ¿Cuanta incertidumbre tiene el AUC? --------------------------------
# Un solo split puede salir afortunado. La validacion cruzada repite la
# medicion con particiones distintas y revela la variabilidad real.
print(f"\n  [3] INCERTIDUMBRE DEL AUC (validacion cruzada, 5 particiones)")
todo_X = pd.concat([X_tr[COLS], X_val[COLS]])
todo_y = pd.concat([y_tr, y_val])
cv = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
scores = cross_val_score(LogisticRegression(max_iter=5000, random_state=SEMILLA),
                         todo_X, todo_y, cv=cv, scoring="roc_auc")
print(f"      AUC por fold: {np.round(scores, 5)}")
print(f"      media {scores.mean():.5f}  desv {scores.std():.5f}")
print(f"      rango ~95%: [{scores.mean()-2*scores.std():.5f}, "
      f"{scores.mean()+2*scores.std():.5f}]")
print(f"\n      El AUC de validacion ({auc_val:.5f}) cae dentro del rango: "
      f"{'si' if abs(auc_val-scores.mean()) <= 2*scores.std() else 'NO'}")
print("      -> Comparar modelos con un solo split no es fiable: una diferencia")
print("         menor que ~2 desviaciones puede ser puro ruido.")

# --- 7d. ¿Importa la regularizacion? ----------------------------------------
# C es EL hiperparametro clasico de la regresion logistica. Merece la pena
# comprobar si de verdad cambia algo antes de perder tiempo ajustandolo.
print(f"\n  [4] ¿IMPORTA EL PARAMETRO C (regularizacion)?")
print(f"      {'C':>10} {'AUC_val':>10} {'|coef| max':>11} {'signos mal':>11}")
print("      " + "-" * 46)
for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    mm = LogisticRegression(C=C, max_iter=5000, random_state=SEMILLA).fit(X_tr[COLS], y_tr)
    a = roc_auc_score(y_val, mm.predict_proba(X_val[COLS])[:, 1])
    marca = "  <- el usado" if C == 1.0 else ""
    print(f"      {C:>10} {a:>10.5f} {np.abs(mm.coef_[0]).max():>11.3f} "
          f"{len(signos_incoherentes(mm.coef_[0], COLS, X_tr, y_tr)):>11}{marca}")
print("\n      Con 120,000 filas y 18 variables (unas 6,600 observaciones por")
print("      parametro) los datos dominan y no hay margen para sobreajustar:")
print("      la regularizacion no cambia nada. Saber QUE NO ajustar ahorra")
print("      tanto tiempo como saber que ajustar.")


# =============================================================================
# PASO 8 — GRAFICAS
# =============================================================================
titulo("PASO 8 — GRAFICAS")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# --- Curva ROC ---------------------------------------------------------------
ax = axes[0, 0]
ax.plot(fpr, tpr, color="#3a4f7a", lw=2, label=f"LogReg (AUC = {auc_val:.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="azar (AUC = 0.5)")
i_ks = np.argmax(tpr - fpr)
ax.plot([fpr[i_ks], fpr[i_ks]], [fpr[i_ks], tpr[i_ks]], color="#a8392f", lw=2,
        label=f"KS = {ks:.4f}")
ax.set_xlabel("Falsos positivos (buenos rechazados)")
ax.set_ylabel("Verdaderos positivos (morosos detectados)")
ax.set_title("Curva ROC")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

# --- Curva Precision-Recall --------------------------------------------------
ax = axes[0, 1]
prec, rec, _ = precision_recall_curve(y_val, p_val)
ax.plot(rec, prec, color="#2b6247", lw=2, label=f"AP = {ap:.4f}")
ax.axhline(y_val.mean(), ls="--", color="gray", lw=1,
           label=f"azar = {y_val.mean():.4f}")
ax.set_xlabel("Recall (morosos detectados)")
ax.set_ylabel("Precision (aciertos entre los marcados)")
ax.set_title("Curva Precision-Recall")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Separacion de las dos poblaciones ---------------------------------------
ax = axes[1, 0]
ax.hist(p_val[y_val == 0], bins=60, alpha=0.65, density=True,
        color="#2b6247", label="no cayo en mora")
ax.hist(p_val[y_val == 1], bins=60, alpha=0.65, density=True,
        color="#a8392f", label="si cayo en mora")
ax.set_xlim(0, 0.6)
ax.set_xlabel("Probabilidad predicha (PD)")
ax.set_ylabel("Densidad")
ax.set_title("¿Separa el modelo a las dos poblaciones?")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Coeficientes ------------------------------------------------------------
ax = axes[1, 1]
c = coefs.head(12).iloc[::-1]
colores = ["#a8392f" if v > 0 else "#3a4f7a" for v in c["coeficiente"]]
ax.barh(range(len(c)), c["coeficiente"], color=colores)
ax.set_yticks(range(len(c)))
ax.set_yticklabels([v.replace("woe_", "") for v in c["variable"]], fontsize=7)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Coeficiente")
ax.set_title("Peso de cada variable (top 12)")
ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(PLOTS / "03_logistic_evaluacion.png", dpi=110)
plt.close()
print(f"  Guardado: {(PLOTS / '03_logistic_evaluacion.png').relative_to(AQUI)}")


# =============================================================================
# PASO 8 — GUARDAR
# =============================================================================
titulo("PASO 8 — GUARDAR MODELO Y METRICAS")

joblib.dump({
    "modelo": modelo,
    "columnas": COLS,
    "subconjunto": NOMBRE_ELEGIDO,
    "semilla": SEMILLA,
}, MODELOS / "logistic_v1.joblib")

metricas = {
    "modelo": "LogisticRegression (WoE)",
    "subconjunto": NOMBRE_ELEGIDO,
    "n_variables": len(COLS),
    "auc_train": round(auc_tr, 5),
    "auc_val": round(auc_val, 5),
    "ks_val": round(float(ks), 5),
    "average_precision_val": round(float(ap), 5),
    "auc_cv_media": round(float(scores.mean()), 5),
    "auc_cv_desv": round(float(scores.std()), 5),
    "error_calibracion_pp": round(float(err_cal), 3),
    "gini_val": round(float(gini(auc_val)), 4),
}
pd.Series(metricas).to_json(METRICAS / "logistic_v1.json", indent=2)
comparacion.to_csv(METRICAS / "logistic_comparacion_subconjuntos.csv", index=False)

print("  artifacts/models/logistic_v1.joblib")
print("  artifacts/metrics/logistic_v1.json")
print("  artifacts/metrics/logistic_comparacion_subconjuntos.csv")

print("\n" + "=" * 79)
print(f"BASELINE ESTABLECIDO — ROC-AUC = {auc_val:.5f}")
print("=" * 79)

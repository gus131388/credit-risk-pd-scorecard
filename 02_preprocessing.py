# -*- coding: utf-8 -*-
"""
02_preprocessing.py — Orquestador del preprocesamiento
Proyecto: Give Me Some Credit — Probability of Default

Este script no contiene logica de transformacion: solo coordina.
Carga datos, hace UN solo split, llama a las funciones de src/, guarda
los resultados y reporta estadisticas.

Produce DOS versiones del dataset a partir del MISMO split, porque cada
familia de modelos necesita un tratamiento distinto:

    raw --> split unico --+-- rama arboles ---> datasets/processed/trees/
                          |
                          +-- rama WoE -------> datasets/processed/woe/

Ambas ramas contienen a las MISMAS personas en train, validacion y test,
para que la comparacion entre modelos sea justa.

Se ejecuta con:  python 02_preprocessing.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from src import preprocessing as pp
from src import woe as woe_mod
from src import artifacts
from src import evaluation as ev_mod

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

# --- Rutas -------------------------------------------------------------------
AQUI = Path(__file__).parent
DATA = AQUI.parent.parent / "data"
PROC = AQUI / "datasets" / "processed"
ARTEFACTOS = AQUI / "artifacts" / "preprocessors"

for carpeta in [PROC / "trees", PROC / "woe", ARTEFACTOS]:
    carpeta.mkdir(parents=True, exist_ok=True)

# --- Configuracion -----------------------------------------------------------
SEMILLA = 42            # fija la aleatoriedad: el resultado es reproducible
PCT_VALIDACION = 0.20
ID = "row_id"           # nombre explicito del identificador de cada persona


def titulo(t):
    print("\n" + "=" * 79)
    print(t)
    print("=" * 79)


# =============================================================================
# PASO 1 — CARGA
# =============================================================================
titulo("PASO 1 — CARGA DE DATOS")

train_raw = pd.read_csv(DATA / "cs-training.csv", index_col=0)
test_raw = pd.read_csv(DATA / "cs-test.csv", index_col=0)

# El indice del CSV es el identificador de cada persona. Le ponemos nombre
# para que sobreviva a los parquet de forma explicita y podamos comprobar
# despues que ninguna fila se perdio ni se mezclo entre particiones.
train_raw.index.name = ID
test_raw.index.name = ID

print(f"  cs-training.csv : {train_raw.shape[0]:>7,} filas")
print(f"  cs-test.csv     : {test_raw.shape[0]:>7,} filas")
print(f"  Identificador   : '{ID}' (indice original del CSV)")


# =============================================================================
# PASO 2 — SPLIT UNICO TRAIN / VALIDACION
# =============================================================================
titulo("PASO 2 — SPLIT UNICO TRAIN / VALIDACION")

# ¿Por que dividir AQUI?
#   El criterio no es "antes de limpiar", sino antes de cualquier operacion
#   que ESTIME PARAMETROS a partir de los datos. La linea divisoria es si la
#   transformacion tiene estado aprendido:
#
#     SIN estado -> da igual el lado del split. El resultado de una fila
#       depende solo de esa fila: renombrar columnas, age < 18 -> NaN,
#       marcar 96/98 como centinela, ratios entre columnas.
#
#     CON estado -> obligatoriamente solo con X_tr. Medianas, percentiles,
#       cortes de bins, tablas WoE, escaladores, imputadores, y la seleccion
#       de variables por asociacion con el target.
#
#   Ejemplo del mecanismo: si se imputara MonthlyIncome con la mediana de las
#   150,000 filas, las 30,000 de validacion habrian ayudado a definir el valor
#   con el que luego se las evalua. En una mediana la fuga es minima, pero es
#   la MISMA estructura logica que en un KNNImputer o un target encoding,
#   donde si infla el AUC de forma apreciable.
#
#   Aqui limpiar() y crear_features() son reglas fijas (sin estado), asi que
#   podrian ir a cualquier lado. Se aplican despues del split por higiene: la
#   que de verdad OBLIGA a este orden es aprender_woe(), en el PASO 5.
#
# ¿Por que UN SOLO split para las dos ramas?
#   Si hicieramos un split para arboles y otro para WoE, cada modelo se
#   evaluaria con personas distintas y la comparacion no significaria nada.
#
# stratify=y -> mantiene la proporcion de defaults (6.68%) en ambas partes.

y = train_raw[pp.TARGET]
X = train_raw.drop(columns=[pp.TARGET])

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=PCT_VALIDACION, stratify=y, random_state=SEMILLA,
)

print(f"  Entrenamiento : {len(X_tr):>7,} filas   default = {y_tr.mean()*100:.2f}%")
print(f"  Validacion    : {len(X_val):>7,} filas   default = {y_val.mean()*100:.2f}%")
print(f"  Test (Kaggle) : {len(test_raw):>7,} filas   default = desconocido")
print("\n  Las tasas coinciden -> el estratificado funciono.")


# =============================================================================
# PASO 3 — LIMPIEZA Y FEATURE ENGINEERING (comun a las dos ramas)
# =============================================================================
titulo("PASO 3 — LIMPIEZA Y FEATURE ENGINEERING")

def procesar(df):
    """Reglas fijas: no aprenden del target, se aplican igual a los tres."""
    return pp.crear_features(pp.limpiar(df))

X_tr_c = procesar(X_tr)
X_val_c = procesar(X_val)
X_test_c = procesar(test_raw.drop(columns=[pp.TARGET]))

nuevas = [c for c in X_tr_c.columns if c not in X_tr.columns]
print(f"  Columnas: {X_tr.shape[1]} -> {X_tr_c.shape[1]}  ({len(nuevas)} nuevas)")

print("\n  Efecto de la limpieza:")
print(f"    Centinelas 96/98          : {X_tr_c['flag_registro_invalido'].sum():>6,}")
print(f"    RevUtil invalidados       : {X_tr_c['flag_revutil_invalido'].sum():>6,}")
print(f"    Ingresos cero/simbolicos  : {X_tr_c['flag_income_cero'].sum():>6,}")
print(f"    DebtRatio sin denominador : {X_tr_c['deuda_bruta_proxy'].notna().sum():>6,}")
print(f"    Edades invalidas          : {X_tr_c['flag_edad_invalida'].sum():>6,}")

# --- ¿Cada indicador separa riesgo? -----------------------------------------
# Triaje previo al modelado: un indicador que no distingue no aporta, y si
# ademas tiene muy pocas filas puede hacer dano (separacion perfecta en la
# regresion logistica, coeficientes inestables).
#
# Comparar las dos tasas puntuales NO basta: con 172 filas una diferencia de
# medio punto es indistinguible del azar. Por eso se acompana del intervalo de
# Wilson: si el IC de la tasa "encendida" CONTIENE la tasa del complemento, no
# hay diferencia demostrable.
sep = ev_mod.tabla_separacion(X_tr_c, y_tr, woe_mod.columnas_flag(X_tr_c))

print(f"\n  ¿Cada indicador separa riesgo?   (tasa global = {y_tr.mean()*100:.2f}%)")
print("  Primero los que separan; dentro de cada bloque, por distancia.\n")
print(f"    {'indicador':<26} {'n':>7} {'eventos':>8} {'si=1':>8} {'si=0':>7} "
      f"{'factor':>8}   {'IC 95% de si=1':>16}   veredicto")
print("    " + "-" * 100)
for r in sep.itertuples():
    ic = f"[{r.ic_inf:5.2f}, {r.ic_sup:6.2f}]"
    veredicto = f"SEPARA ({r.direccion})" if r.separa else "no separa"
    print(f"    {r.indicador:<26} {r.n:>7,} {r.eventos:>8,} {r.si_1:>7.2f}% "
          f"{r.si_0:>6.2f}% {r.factor:>7.1f}x   {ic:>16}   {veredicto}")

n_no = int((~sep["separa"]).sum())
print(f"\n    {len(sep) - n_no} de {len(sep)} indicadores separan riesgo de forma demostrable.")
if n_no:
    inutiles = ", ".join(sep.loc[~sep["separa"], "indicador"])
    print(f"    Sin senal (su IC contiene la tasa del complemento): {inutiles}")
    print("    No se eliminan: los arboles los ignoran sin coste. El modelo")
    print("    logistico si los excluye (usa solo las 10 variables originales).")


# =============================================================================
# PASO 4 — RAMA A: ARBOLES (XGBoost / LightGBM)
# =============================================================================
titulo("PASO 4 — RAMA A: ARBOLES")

tr_trees = pp.preparar_arboles(X_tr_c)
val_trees = pp.preparar_arboles(X_val_c)
test_trees = pp.preparar_arboles(X_test_c)

print(f"  {tr_trees.shape[1]} variables, NaN conservados a proposito.")
print("\n  NaN por columna (los arboles los usan como informacion):")
nans = tr_trees.isna().sum()
for col, n in nans[nans > 0].sort_values(ascending=False).items():
    print(f"    {col:<40} {n:>7,}  ({n/len(tr_trees)*100:5.2f}%)")

tr_trees.assign(**{pp.TARGET: y_tr}).to_parquet(PROC / "trees" / "train.parquet")
val_trees.assign(**{pp.TARGET: y_val}).to_parquet(PROC / "trees" / "val.parquet")
test_trees.to_parquet(PROC / "trees" / "test.parquet")
print(f"\n  Guardado en {PROC.relative_to(AQUI)}/trees/")


# =============================================================================
# PASO 5 — RAMA B: WoE (Logistic Regression)
# =============================================================================
titulo("PASO 5 — RAMA B: WoE")

# APRENDER: solo con train, porque mira el target.
tablas_woe = woe_mod.aprender_woe(X_tr_c, y_tr)

iv = woe_mod.calcular_iv(tablas_woe)
print("  INFORMATION VALUE (poder predictivo de cada variable):\n")
print(iv.to_string(index=False))
print("\n  Escala: <0.02 inutil | 0.02-0.10 debil | 0.10-0.30 medio")
print("          0.30-0.50 fuerte | >0.50 sospechoso (investigar)")

print("\n\n  TABLA WoE DE LA VARIABLE MAS PREDICTIVA")
print("  (WoE positivo = mas seguro que la media | negativo = mas riesgoso)")
top = iv.iloc[0]["variable"]
print(f"\n  --- {top}  (IV = {iv.iloc[0]['IV']}) ---")
print(tablas_woe[top][["n", "pct_poblacion", "default_rate_pct", "woe", "iv_parcial"]]
      .round(4).to_string())

# APLICAR: a los tres, usando SOLO lo aprendido en train.
tr_woe = woe_mod.aplicar_woe(X_tr_c, tablas_woe)
val_woe = woe_mod.aplicar_woe(X_val_c, tablas_woe)
test_woe = woe_mod.aplicar_woe(X_test_c, tablas_woe)

print(f"\n\n  {tr_woe.shape[1]} variables | NaN restantes: {tr_woe.isna().sum().sum()}")
print("  (WoE no deja faltantes: 'FALTANTE' es un bin mas con su propio riesgo)")

tr_woe.assign(**{pp.TARGET: y_tr}).to_parquet(PROC / "woe" / "train.parquet")
val_woe.assign(**{pp.TARGET: y_val}).to_parquet(PROC / "woe" / "val.parquet")
test_woe.to_parquet(PROC / "woe" / "test.parquet")
print(f"\n  Guardado en {PROC.relative_to(AQUI)}/woe/")


# =============================================================================
# PASO 6 — PERSISTIR EL TRANSFORMADOR
# =============================================================================
titulo("PASO 6 — PERSISTIR EL TRANSFORMADOR")

# Guardar los .parquet no basta: son el RESULTADO de la transformacion,
# no la transformacion. Para puntuar un cliente nuevo manana hay que
# aplicarle los MISMOS cortes y los MISMOS valores WoE de hoy.
artefacto = artifacts.construir_artefacto(
    tablas_woe=tablas_woe,
    columnas_trees=tr_trees.columns,
    columnas_woe=tr_woe.columns,
    n_filas_train=len(X_tr),
    tasa_default=y_tr.mean(),
    semilla=SEMILLA,
)
ruta = artifacts.guardar(artefacto, ARTEFACTOS / "preprocessor_v1.joblib")
print(artifacts.resumen(artefacto))
print(f"\n  Guardado en {ruta.relative_to(AQUI)}")


# =============================================================================
# PASO 7 — VERIFICACIONES
# =============================================================================
titulo("PASO 7 — VERIFICACIONES")

ok = True

def check(condicion, mensaje):
    global ok
    print(f"  [{'OK' if condicion else 'ERROR'}] {mensaje}")
    ok &= bool(condicion)

# --- Integridad del split ---------------------------------------------------
check(len(set(X_tr.index) & set(X_val.index)) == 0,
      "train y validacion no comparten ninguna persona")
check(len(X_tr) + len(X_val) == len(train_raw),
      f"no se perdio ninguna fila ({len(X_tr)+len(X_val):,} = {len(train_raw):,})")

# --- Las dos ramas deben cubrir a las MISMAS personas -----------------------
check(tr_trees.index.equals(tr_woe.index), "trees y woe: mismas personas en train")
check(val_trees.index.equals(val_woe.index), "trees y woe: mismas personas en validacion")
check(test_trees.index.equals(test_woe.index), "trees y woe: mismas personas en test")

# --- Contrato de columnas ---------------------------------------------------
for nombre, a, b, c in [("trees", tr_trees, val_trees, test_trees),
                        ("woe", tr_woe, val_woe, test_woe)]:
    check(list(a.columns) == list(b.columns) == list(c.columns),
          f"{nombre}: mismas columnas y mismo orden en train/val/test")

# --- En test hay que predecir TODAS las filas -------------------------------
check(len(test_trees) == len(test_raw), f"trees/test conserva las {len(test_raw):,} filas")
check(len(test_woe) == len(test_raw), f"woe/test conserva las {len(test_raw):,} filas")

# --- El dataset WoE debe ser apto para LogReg -------------------------------
check(tr_woe.isna().sum().sum() == 0, "woe: sin NaN")
check(np.isfinite(tr_woe.to_numpy()).all(), "woe: sin infinitos")

# --- Anti-fuga --------------------------------------------------------------
check(pp.TARGET not in tr_trees.columns, "trees: el target no esta entre las features")
check(pp.TARGET not in tr_woe.columns, "woe: el target no esta entre las features")

# --- El artefacto guardado debe reproducir el resultado ---------------------
# Esta es la prueba de fuego: cargamos el artefacto desde disco y lo
# aplicamos a datos crudos como si fueramos produccion. Debe dar lo mismo.
recargado = artifacts.cargar(ruta)
muestra = test_raw.drop(columns=[pp.TARGET]).head(500)
repro = artifacts.transformar_datos_nuevos(muestra, recargado, rama="woe")
check(repro.equals(test_woe.head(500)),
      "el artefacto recargado reproduce exactamente la transformacion")

print("\n" + "=" * 79)
print("PREPROCESAMIENTO COMPLETADO" if ok else "HAY ERRORES — REVISAR ARRIBA")
print("=" * 79)
sys.exit(0 if ok else 1)

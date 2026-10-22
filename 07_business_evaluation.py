# -*- coding: utf-8 -*-
"""
07_business_evaluation.py — Traduccion del modelo a dolares
Proyecto: Give Me Some Credit — Probability of Default

POR QUE ESTE SCRIPT EXISTE

Un ROC-AUC de 0.863 no le dice nada a un comite de credito. La pregunta
real es: "¿cuanto dinero nos hace ganar este modelo, y donde ponemos el
punto de corte?"

Aqui traducimos las probabilidades del modelo a beneficio esperado, y
demostramos algo que sorprende: un modelo con buen AUC puede ser
ECONOMICAMENTE INUTIL si el negocio no lo necesita.

Se ejecuta con:  python 07_business_evaluation.py
Evalua automaticamente todos los modelos que encuentre en artifacts/models/
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
from src.evaluation import umbral_optimo, beneficio, beneficio_sin_modelo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.width", 200)

AQUI = Path(__file__).parent
PROC = AQUI / "datasets" / "processed"
MODELOS = AQUI / "artifacts" / "models"
METRICAS = AQUI / "artifacts" / "metrics"
PLOTS = AQUI / "artifacts" / "plots"
for d in (METRICAS, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "SeriousDlqin2yrs"

# =============================================================================
# SUPUESTOS ECONOMICOS
# =============================================================================
# Simplificados a proposito: todos los prestamos del mismo importe, sin valor
# temporal del dinero ni recuperaciones parciales. La ESTRUCTURA del
# razonamiento es la que se usa en la practica; los numeros exactos los
# aportaria el area de negocio.
PRESTAMO = 10_000     # importe medio
MARGEN = 800          # ganancia si el cliente paga (8% sobre el principal)
LGD = 0.60            # Loss Given Default: fraccion del principal que se pierde
PERDIDA = PRESTAMO * LGD


def titulo(t):
    print("\n" + "=" * 79)
    print(t)
    print("=" * 79)


# =============================================================================
# PASO 1 — CARGA
# =============================================================================
titulo("PASO 1 — DATOS Y MODELOS")

modelos = {}
for f in sorted(MODELOS.glob("*.joblib")):
    art = joblib.load(f)
    rama = "woe" if "logistic" in f.stem else "trees"
    modelos[f.stem] = {"art": art, "rama": rama}
    print(f"  {f.stem:<22} rama={rama}  ({len(art['columnas'])} variables)")

if not modelos:
    sys.exit("No hay modelos en artifacts/models/. Ejecuta antes 03_train_logistic.py")

datos = {}
for rama in {m["rama"] for m in modelos.values()}:
    va = pd.read_parquet(PROC / rama / "val.parquet")
    datos[rama] = (va.drop(columns=[TARGET]), va[TARGET].to_numpy())

y = datos[next(iter(datos))][1]
print(f"\n  Validacion: {len(y):,} personas | {int(y.sum()):,} morosos ({y.mean()*100:.2f}%)")

predicciones = {}
for nombre, m in modelos.items():
    X, _ = datos[m["rama"]]
    predicciones[nombre] = m["art"]["modelo"].predict_proba(X[m["art"]["columnas"]])[:, 1]


# =============================================================================
# PASO 2 — SUPUESTOS Y UMBRAL OPTIMO
# =============================================================================
titulo("PASO 2 — SUPUESTOS ECONOMICOS")

u_opt = umbral_optimo(MARGEN, PERDIDA)
print(f"  Prestamo medio           : ${PRESTAMO:,}")
print(f"  Ganancia si paga         : ${MARGEN:,}")
print(f"  Perdida si cae en mora   : ${PERDIDA:,.0f}   (LGD {LGD*100:.0f}%)")
print(f"  Ratio perdida / ganancia : {PERDIDA/MARGEN:.1f}x")
print(f"\n  UMBRAL OPTIMO = M / (M + L) = {MARGEN} / {MARGEN+PERDIDA:.0f} = {u_opt:.4f}"
      f"  ({u_opt*100:.2f}%)")
print("  -> aprobar a todo el que tenga PD por debajo de ese valor")


# =============================================================================
# PASO 3 — BENEFICIO POR UMBRAL
# =============================================================================
titulo("PASO 3 — BENEFICIO SEGUN EL UMBRAL DE APROBACION")

# Politicas de referencia, sin modelo
ben_todos = beneficio_sin_modelo(y, MARGEN, PERDIDA)
print(f"  {'Aprobar a TODOS (sin modelo)':<38} ${ben_todos:>14,.0f}")
print(f"  {'Aprobar a NADIE':<38} ${0:>14,.0f}")

for nombre, p in predicciones.items():
    print(f"\n  --- {nombre} ---")
    print(f"  {'umbral':>8} | {'aprobados':>10} | {'buenos':>8} | {'morosos':>8} | {'beneficio':>15}")
    print("  " + "-" * 64)
    filas = []
    for u in [0.02, 0.04, 0.06, 0.08, u_opt, 0.15, 0.20, 0.30, 0.50]:
        r = beneficio(p, y, u, MARGEN, PERDIDA)
        filas.append(r)
        marca = "  <- optimo teorico" if abs(u - u_opt) < 1e-9 else ""
        print(f"  {u:>8.4f} | {r['aprobados']:>10,} | {r['buenos']:>8,} | "
              f"{r['morosos']:>8,} | ${r['beneficio']:>14,.0f}{marca}")

    # --- ¿Merece la pena buscar el umbral en los datos? --------------------
    # Buscar el mejor umbral barriendo el MISMO conjunto donde luego se
    # reporta el beneficio es hacer trampa: ese umbral esta ajustado a las
    # particularidades de esas filas concretas.
    # Para comprobarlo honestamente partimos validacion en dos mitades:
    # buscamos el optimo en A y lo aplicamos en B, sin haberla mirado.
    mitad = len(y) // 2
    pA, yA = p[:mitad], y[:mitad]
    pB, yB = p[mitad:], y[mitad:]

    rejilla = np.linspace(0.005, 0.6, 300)
    u_emp_A = max(rejilla, key=lambda u: beneficio(pA, yA, u, MARGEN, PERDIDA)["beneficio"])

    ben_emp_B = beneficio(pB, yB, u_emp_A, MARGEN, PERDIDA)["beneficio"]
    ben_teo_B = beneficio(pB, yB, u_opt, MARGEN, PERDIDA)["beneficio"]

    print(f"\n  ¿Conviene BUSCAR el umbral en los datos o usar la formula?")
    print(f"    Umbral empirico hallado en la mitad A : {u_emp_A:.4f}")
    print(f"    Umbral teorico M/(M+L)                : {u_opt:.4f}")
    print(f"    Aplicados a la mitad B (nunca vista):")
    print(f"      empirico -> ${ben_emp_B:>12,.0f}")
    print(f"      teorico  -> ${ben_teo_B:>12,.0f}   "
          f"{'(gana la formula)' if ben_teo_B >= ben_emp_B else '(gana el empirico)'}")
    print("    -> El umbral se DERIVA de la economia, no se aprende de los datos.")

    # El beneficio que se reporta usa SIEMPRE el umbral teorico: no depende
    # de haber mirado ningun conjunto de evaluacion.
    valor = beneficio(p, y, u_opt, MARGEN, PERDIDA)["beneficio"] - ben_todos
    print(f"\n  VALOR DEL MODELO (al umbral teorico): ${valor:,.0f} "
          f"sobre {len(y):,} solicitudes")
    print(f"                    = ${valor/len(y):,.2f} por solicitud evaluada")


# =============================================================================
# PASO 4 — ¿CUANDO NO VALE LA PENA EL MODELO?
# =============================================================================
titulo("PASO 4 — SENSIBILIDAD: ¿CUANDO COMPENSA USAR EL MODELO?")

print("  Un modelo con buen AUC puede ser INUTIL si el negocio no lo necesita.")
print("  Aqui variamos el ratio perdida/ganancia y vemos que pasa.\n")

nombre_principal = next(iter(predicciones))
p = predicciones[nombre_principal]

print(f"  {'ratio L/M':>10} {'umbral opt':>11} {'sin modelo':>16} {'con modelo':>16} {'ganancia':>15}")
print("  " + "-" * 74)
sens = []
for ratio in [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20]:
    L = MARGEN * ratio
    u = umbral_optimo(MARGEN, L)
    sin_m = beneficio_sin_modelo(y, MARGEN, L)
    con_m = beneficio(p, y, u, MARGEN, L)["beneficio"]
    con_m = max(con_m, 0)     # siempre existe la opcion de no prestar a nadie
    sens.append({"ratio_L_M": ratio, "umbral_optimo": round(u, 4),
                 "sin_modelo": sin_m, "con_modelo": con_m,
                 "ganancia": con_m - sin_m})
    print(f"  {ratio:>10.1f} {u:>11.4f} ${sin_m:>15,.0f} ${con_m:>15,.0f} ${con_m-sin_m:>14,.0f}")

sens = pd.DataFrame(sens)
print("\n  DOS CONCLUSIONES:")
print("   1. Si las perdidas son BARATAS (ratio < 1), lo optimo es prestarle a")
print("      todo el mundo: el modelo solo estorba rechazando gente que si pagaba.")
print("   2. Cuanto mas CARO el default, mas vale el modelo. A partir de ~15x el")
print("      negocio pierde dinero sin modelo y lo gana con el: ahi el modelo no")
print("      es una mejora, es la diferencia entre existir o no.")
print("\n   En credito al consumo el ratio tipico esta entre 5x y 15x.")


# =============================================================================
# PASO 5 — GRAFICAS
# =============================================================================
titulo("PASO 5 — GRAFICAS")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

# --- Beneficio vs umbral ----------------------------------------------------
ax = axes[0]
us = np.linspace(0.005, 0.6, 300)
for nombre, pred in predicciones.items():
    bs = [beneficio(pred, y, u, MARGEN, PERDIDA)["beneficio"] / 1e6 for u in us]
    ax.plot(us, bs, lw=2, label=nombre)
ax.axhline(ben_todos / 1e6, ls="--", color="gray", lw=1.5,
           label=f"sin modelo (${ben_todos/1e6:.2f}M)")
ax.axvline(u_opt, ls=":", color="#a8392f", lw=2,
           label=f"umbral optimo = {u_opt:.4f}")
ax.set_xlabel("Umbral de aprobacion (PD maxima aceptada)")
ax.set_ylabel("Beneficio (millones de $)")
ax.set_title(f"Beneficio segun el umbral\n(margen ${MARGEN}, perdida ${PERDIDA:,.0f})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Sensibilidad al ratio ---------------------------------------------------
ax = axes[1]
x = np.arange(len(sens))
ancho = 0.38
ax.bar(x - ancho/2, sens["sin_modelo"] / 1e6, ancho, label="sin modelo", color="#8a8f9c")
ax.bar(x + ancho/2, sens["con_modelo"] / 1e6, ancho, label="con modelo", color="#2b6247")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{r:g}x" for r in sens["ratio_L_M"]], fontsize=8)
ax.set_xlabel("Ratio perdida / ganancia")
ax.set_ylabel("Beneficio (millones de $)")
ax.set_title("¿Cuando compensa el modelo?")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(PLOTS / "07_evaluacion_negocio.png", dpi=110)
plt.close()
print(f"  Guardado: {(PLOTS / '07_evaluacion_negocio.png').relative_to(AQUI)}")

sens.to_csv(METRICAS / "business_sensibilidad.csv", index=False)
print(f"  Guardado: artifacts/metrics/business_sensibilidad.csv")

print("\n" + "=" * 79)
print("EVALUACION DE NEGOCIO COMPLETADA")
print("=" * 79)

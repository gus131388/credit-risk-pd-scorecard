# -*- coding: utf-8 -*-
"""
evaluation.py — Metricas de evaluacion y traduccion a negocio
Proyecto: Give Me Some Credit — Probability of Default

POR QUE ESTE MODULO EXISTE

Estas funciones vivian sueltas dentro de 03, 06 y 07. Eso tenia dos
problemas: se duplicaban entre scripts (y podian desincronizarse), y no
se podian probar — un test no puede importar un script sin ejecutarlo
entero.

Al extraerlas aqui, la logica que decide si un modelo es aceptable queda
cubierta por tests.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


# =============================================================================
# INFERENCIA SOBRE PROPORCIONES
# =============================================================================

def ic_wilson(p, n, z=1.96):
    """
    Intervalo de confianza de Wilson para una proporcion.

               p + z^2/(2n)  +/-  z*sqrt( p(1-p)/n + z^2/(4n^2) )
        IC =  ------------------------------------------------------
                              1 + z^2/n

    Se usa en vez de la aproximacion normal (p +/- 1.96*EE) porque con eventos
    raros (aqui 6.68%) la normal produce intervalos sesgados y hasta limites
    negativos; Wilson no.

    Acepta escalares o arrays. Con n = 0 devuelve NaN: el intervalo de un grupo
    vacio no existe.

    Devuelve (inferior, superior) en la misma escala que p (0-1).
    """
    n = np.where(np.asarray(n, dtype=float) > 0, n, np.nan)
    p = np.asarray(p, dtype=float)
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margen = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / (1 + z**2 / n)
    return centro - margen, centro + margen


def tabla_separacion(X, y, columnas, z=1.96):
    """
    Para cada indicador binario, contrasta la tasa de evento de las filas que
    lo tienen encendido contra la de todas las demas.

    La pregunta que responde: ¿saber que este indicador vale 1 cambia lo que
    espero del cliente? Comparar solo las dos tasas puntuales engana, porque un
    indicador con 172 filas puede mostrar una diferencia aparente que es puro
    ruido de muestreo. Por eso se acompana del intervalo de Wilson.

    Criterio: si el IC de la tasa con el indicador encendido CONTIENE la tasa
    del grupo complementario, no hay diferencia demostrable.

    Orden del resultado: primero los que separan, y dentro de cada bloque por
    distancia entre los dos grupos. Ordenar SOLO por distancia enganaria: un
    indicador con una sola fila puede mostrar 6.68 puntos de diferencia y no
    significar nada, y aparecer arriba del todo.
    """
    y = pd.Series(np.asarray(y), index=X.index)
    filas = []

    for col in columnas:
        m = X[col] == 1
        n_1 = int(m.sum())
        if n_1 == 0 or n_1 == len(X):
            continue

        ev = int(y[m].sum())
        p_1 = ev / n_1
        p_0 = float(y[~m].mean())
        lo, hi = ic_wilson(p_1, n_1, z)

        filas.append({
            "indicador": col,
            "n": n_1,
            "eventos": ev,
            "si_1": p_1 * 100,
            "si_0": p_0 * 100,
            "factor": p_1 / p_0 if p_0 > 0 else np.nan,
            "ic_inf": float(lo) * 100,
            "ic_sup": float(hi) * 100,
            "separa": not (lo <= p_0 <= hi),
            "direccion": "+" if p_1 > p_0 else "-",
        })

    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    return (tabla.assign(_d=(tabla["si_1"] - tabla["si_0"]).abs())
                 .sort_values(["separa", "_d"], ascending=[False, False])
                 .drop(columns="_d")
                 .reset_index(drop=True))


# =============================================================================
# METRICAS DE DISCRIMINACION
# =============================================================================

def ks_statistic(y_real, p_pred):
    """
    KS (Kolmogorov-Smirnov): maxima separacion entre la distribucion
    acumulada de buenos y la de malos.

    Es la metrica clasica de los scorecards bancarios. Equivale al maximo
    de (TPR - FPR) a lo largo de la curva ROC.

    Referencia: > 0.30 aceptable | > 0.40 bueno
    """
    fpr, tpr, umbrales = roc_curve(y_real, p_pred)
    i = int(np.argmax(tpr - fpr))
    return float(tpr[i] - fpr[i]), float(umbrales[i])


def gini(auc):
    """Gini = 2*AUC - 1. La banca lo usa por costumbre; misma informacion."""
    return 2.0 * auc - 1.0


# =============================================================================
# CALIBRACION
# =============================================================================
# El AUC solo mide si el modelo ORDENA bien. La calibracion pregunta otra
# cosa: cuando dice "8% de riesgo", ¿de verdad falla el 8% de ese grupo?
#
# Sin calibracion, la regla de negocio umbral = M/(M+L) deja de funcionar:
# el modelo ordenaria bien pero no diria cuanto prestar.

def tabla_calibracion(y_real, p_pred, n_grupos=10):
    """
    Parte las predicciones en grupos de igual tamano y compara la PD
    predicha media con la PD observada en cada grupo.
    """
    d = pd.DataFrame({"p": np.asarray(p_pred), "y": np.asarray(y_real)})
    d["grupo"] = pd.qcut(d["p"], n_grupos, labels=False, duplicates="drop")
    g = d.groupby("grupo").agg(n=("y", "size"), pd_pred=("p", "mean"),
                               pd_real=("y", "mean"))
    g["pd_pred"] = (g["pd_pred"] * 100).round(2)
    g["pd_real"] = (g["pd_real"] * 100).round(2)
    g["error_pp"] = (g["pd_pred"] - g["pd_real"]).round(2)
    return g


def error_calibracion(y_real, p_pred, n_grupos=10):
    """Error absoluto medio de calibracion, en puntos porcentuales."""
    return float(tabla_calibracion(y_real, p_pred, n_grupos)["error_pp"].abs().mean())


# =============================================================================
# COHERENCIA DEL SCORECARD
# =============================================================================

def signos_invertidos(coeficientes, columnas, prefijo_woe="woe_"):
    """
    Devuelve las variables WoE cuyo coeficiente tiene el signo equivocado.

    En variables WoE el signo esperado es NEGATIVO: un WoE alto significa
    "mas seguro que la media", asi que debe BAJAR la probabilidad de default.
    Un coeficiente positivo dice literalmente "mas atrasos -> menos riesgo",
    lo cual es absurdo y destruye la credibilidad del scorecard.

    LIMITACION CONOCIDA: solo revisa columnas con prefijo woe_. Los
    indicadores quedan exentos porque su direccion no esta fijada a priori.
    Esa exencion oculto un defecto real (ver signos_incoherentes).
    Se conserva por compatibilidad; para validar un scorecard usa
    signos_incoherentes(), que es estrictamente mas fuerte.
    """
    c = pd.Series(np.asarray(coeficientes), index=list(columnas))
    return [x for x in columnas if x.startswith(prefijo_woe) and c[x] > 0]


def direccion_univariada(X, y, columna):
    """
    Direccion en que la variable mueve el riesgo, medida sin modelo.

    Devuelve +1 si el grupo con la variable activa tiene MAS eventos que el
    resto, -1 si tiene menos, y 0 si el grupo esta vacio.

    Para indicadores 0/1 compara ambos grupos. Para variables continuas usa
    la correlacion con el objetivo.
    """
    x = X[columna]
    if set(pd.unique(x.dropna())) <= {0, 1}:
        m = x == 1
        if m.sum() == 0 or (~m).sum() == 0:
            return 0
        return 1 if y[m].mean() > y[~m].mean() else -1
    r = np.corrcoef(x.to_numpy(float), np.asarray(y, dtype=float))[0, 1]
    if not np.isfinite(r) or r == 0:
        return 0
    return int(np.sign(r))


def signos_incoherentes(coeficientes, columnas, X_train, y_train,
                        prefijo_woe="woe_"):
    """
    Devuelve las variables cuyo coeficiente CONTRADICE su comportamiento
    univariado. Es la comprobacion completa que debe pasar un scorecard.

    POR QUE EXISTE (hallazgo H1 de la auditoria)

    signos_invertidos() solo mira las variables woe_, y eximia por completo a
    los indicadores. Eso permitio que llegaran al modelo final tres
    indicadores con el signo al reves; el peor de ellos,
    flag_registro_invalido, identifica al grupo con 54.67% de eventos frente
    al 6.60% de la poblacion, y su coeficiente negativo le asignaba PUNTOS
    POSITIVOS en el scorecard.

    La prediccion agregada seguia siendo correcta: el bin FALTANTE del WoE ya
    cargaba el riesgo, y el indicador actuaba como correccion de ese doble
    conteo. Lo que estaba mal era la ATRIBUCION, que es la unica razon de ser
    de un scorecard.

    Criterio aplicado:
      - variables woe_*  -> se espera coeficiente NEGATIVO (convencion WoE)
      - indicadores 0/1  -> se espera el MISMO signo que su direccion
                            univariada medida en los datos
    """
    c = pd.Series(np.asarray(coeficientes), index=list(columnas))
    y_train = pd.Series(np.asarray(y_train), index=X_train.index)

    problemas = []
    for col in columnas:
        esperado = -1 if col.startswith(prefijo_woe) else direccion_univariada(
            X_train, y_train, col)
        if esperado == 0:
            continue
        if np.sign(c[col]) != esperado:
            problemas.append(col)
    return problemas


# =============================================================================
# TRADUCCION A NEGOCIO
# =============================================================================

def umbral_optimo(margen, perdida):
    """
    Umbral de aprobacion optimo, derivado del valor esperado.

        Prestar a alguien con probabilidad de mora p:
            Valor esperado = (1 - p) * MARGEN  -  p * PERDIDA

        Conviene mientras sea positivo:
            (1 - p) * M  >  p * L
                     M   >  p * (M + L)
                     p   <  M / (M + L)

    El umbral NO es una eleccion estadistica: lo dicta la economia del
    negocio. Por eso no hay que "aprenderlo" de los datos.
    """
    if margen <= 0 or perdida < 0:
        raise ValueError("margen debe ser > 0 y perdida >= 0")
    return margen / (margen + perdida)


def beneficio(p_pred, y_real, umbral, margen, perdida):
    """Beneficio de aprobar a todo el que tenga PD por debajo del umbral."""
    p_pred = np.asarray(p_pred)
    y_real = np.asarray(y_real)
    aprobados = p_pred < umbral
    buenos = int(((y_real == 0) & aprobados).sum())
    malos = int(((y_real == 1) & aprobados).sum())
    return {
        "umbral": float(umbral),
        "aprobados": int(aprobados.sum()),
        "buenos": buenos,
        "morosos": malos,
        "beneficio": buenos * margen - malos * perdida,
    }


def beneficio_sin_modelo(y_real, margen, perdida):
    """Referencia: aprobar a todo el mundo."""
    y_real = np.asarray(y_real)
    return int((y_real == 0).sum()) * margen - int((y_real == 1).sum()) * perdida


# =============================================================================
# VALIDACION DEL SUBMISSION
# =============================================================================

def validar_submission(sub, n_esperado, columnas=("Id", "Probability")):
    """
    Comprueba que el fichero de entrega cumple el contrato.

    Devuelve una lista de (descripcion, ok). Vacia de problemas = todo bien.
    El error mas caro aqui es entregar menos filas de las pedidas: hay que
    dar una prediccion para TODAS las personas del test.
    """
    cols = list(columnas)
    p = sub[cols[1]] if cols[1] in sub.columns else pd.Series(dtype=float)
    return [
        (f"filas: {len(sub):,} = {n_esperado:,}", len(sub) == n_esperado),
        ("columnas y nombres correctos", list(sub.columns) == cols),
        ("sin valores faltantes", bool(p.notna().all()) and len(p) > 0),
        ("probabilidades dentro de [0, 1]", bool(p.between(0, 1).all()) and len(p) > 0),
        ("ids unicos", bool(sub[cols[0]].is_unique) if cols[0] in sub.columns else False),
        ("ids de 1 a N", bool(cols[0] in sub.columns
                              and sub[cols[0]].min() == 1
                              and sub[cols[0]].max() == len(sub))),
    ]

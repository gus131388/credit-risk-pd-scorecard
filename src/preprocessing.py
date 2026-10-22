# -*- coding: utf-8 -*-
"""
preprocessing.py — Limpieza y feature engineering
Proyecto: Give Me Some Credit — Probability of Default

Este modulo NO se ejecuta directamente: contiene las FUNCIONES que usan
los scripts 02_preprocessing.py y los de modelado.

Por que un modulo aparte y no celdas sueltas:
  Cada transformacion debe aplicarse EXACTAMENTE igual a train, validacion y test.
  Si el codigo estuviera copiado en varios sitios, tarde o temprano se
  desincroniza y se introduce un bug silencioso (o una fuga de datos).

AMBITO DE LA EVIDENCIA
  Todas las cifras citadas en los comentarios de este modulo proceden del EDA
  ejecutado UNICAMENTE sobre las 120,000 filas de entrenamiento (01_eda.py hace
  el split antes de calcular nada). Las 30,000 de validacion no intervinieron
  ni en el diagnostico ni en la eleccion de los umbrales.

  Ademas, los CORTES (100, 2.0, 1.0) se eligieron por la forma de la
  distribucion de X — colas, bimodalidad, ausencia de denominador — y no
  optimizando ninguna metrica. Las tasas de default que aparecen abajo son
  VERIFICACION POSTERIOR de que el corte no destruia senal, no el criterio
  con el que se eligio.

REGLA DE ORO CONTRA DATA LEAKAGE
  Toda estadistica (medianas, tablas WoE, umbrales aprendidos) se calcula
  UNICAMENTE con el conjunto de entrenamiento, y luego se APLICA a los demas.
  Por eso el patron es siempre:
        params = aprender_xxx(train)     <- mira el target, solo en train
        df_out = aplicar_xxx(df, params) <- no mira el target, sirve para todos
  (Es el mismo patron fit/transform de scikit-learn, escrito de forma explicita
   para que se vea que se aprende y de donde.)
"""

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURACION — los umbrales que decidimos a partir del EDA
# =============================================================================

TARGET = "SeriousDlqin2yrs"

# Codigos centinela: el sistema de origen los usa para decir "dato no disponible".
# En el EDA vimos que 214 filas tienen 96 o 98 en las TRES columnas a la vez,
# y que su default rate es 54.67% (vs 6.60% del resto).
SENTINELAS = [96, 98]

COLS_MORA = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# RevolvingUtilization deberia ser un ratio 0-1, pero superarlo es LEGITIMO
# (sobregiros, lineas excedidas). Evidencia del EDA, banda por banda:
#     1.0 - 1.2  -> 38.23% default   (n=1,797)  senal real
#     1.2 - 1.5  -> 46.83% default   (n=  363)  senal real
#     1.5 - 2    -> 45.21% default   (n=  188)  senal real
#     2   - 100  -> 30.83% default   (n=  120)  senal real, pero n pequeno
#     100 - 1000 ->  3.03% default   (n=   33)  RUIDO
#     > 1000     ->  7.19% default   (n=  139)  RUIDO
# Por encima de 100 el riesgo se DESPLOMA por debajo de la media global
# (6.68%): deja de comportarse como deuda. Ademas el histograma log-log
# (05_colas_completas.png) muestra dos modas separadas por un valle vacio,
# y 100 cae justo en ese valle.
# OJO: capear en 1 habria invalidado 2,640 filas con 37.50% de default,
# es decir el grupo mas predictivo del dataset.
REVUTIL_UMBRAL_INVALIDO = 100.0

# DebtRatio ya NO usa umbral de magnitud.
#
# Historico: hubo un DEBTRATIO_UMBRAL_INVALIDO = 2.0, elegido porque el regimen
# de "ratio real" crece de forma monotona hasta ahi (6.05 -> 9.88 -> 13.12% en
# las bandas 0-0.5, 0.5-1, 1-2) y despues se rompe. Era un PROXY: lo que de
# verdad distingue un ratio de un importe es si existia denominador.
# La seccion 5e del EDA lo demostro y el umbral se sustituyo por el criterio
# directo (ver paso 1d de limpiar()). Se elimina la constante para que no quede
# un parametro muerto que alguien pueda creer activo.

# Edad minima plausible para un solicitante de credito.
EDAD_MINIMA = 18

# Un ingreso de 0 o 1 al mes para alguien con lineas de credito activas no es
# un ingreso: es "no reportado" codificado como cero. Evidencia del EDA:
#     MonthlyIncome == 0     -> n= 1,322  default 4.16%
#     0 < MonthlyIncome <= 1 -> n=   498  default 2.41%
#     MonthlyIncome  > 100   -> n=94,456  default 7.02%
#     MonthlyIncome FALTANTE -> n=23,675  default 5.57%
# Los ceros se comportan como los faltantes (riesgo POR DEBAJO de la media),
# no como gente pobre (el decil de menor ingreso tiene 8.35% de default).
# Dejarlos como numero los pondria en el extremo pobre de la distribucion,
# donde el modelo espera riesgo ALTO: señal invertida.
INGRESO_MINIMO_PLAUSIBLE = 1.0


# =============================================================================
# PASO 1 — LIMPIEZA POR REGLAS
# =============================================================================
# Estas transformaciones NO aprenden nada de los datos: son reglas fijas que
# decidimos nosotros. Por eso pueden aplicarse a cualquier conjunto sin riesgo
# de fuga: no dependen del target ni de estadisticas del train.

def limpiar(df):
    """
    Aplica la limpieza por reglas y crea los flags de calidad de dato.

    Filosofia: NUNCA borramos filas. En test hay que predecir las 101,503
    personas, asi que un pipeline que dependa de eliminar filas se rompe
    en produccion. Los valores corruptos se convierten en NaN y se marcan
    con un flag, para no perder la informacion de que estaban corruptos.
    """
    df = df.copy()

    # --- 1a. Flags de valores faltantes ------------------------------------
    # En el EDA descubrimos que "el dato falta" es predictivo por si mismo,
    # y que hay una gradacion monotona:
    #     nada falta        -> 6.96% default  (n=96,325)
    #     solo falta Income -> 5.75%           (n=20,547)
    #     faltan ambos      -> 4.41%           (n= 3,128)
    # Por eso usamos DOS flags separados: uno solo perderia esa gradacion.
    df["flag_income_faltante"] = df["MonthlyIncome"].isna().astype(int)
    df["flag_dependents_faltante"] = df["NumberOfDependents"].isna().astype(int)

    # Ingreso cero/simbolico: mismo fenomeno que el faltante, pero codificado
    # como numero. Flag aparte porque su nivel de riesgo NO es identico al de
    # los NaN (4.16% vs 5.57%), y un flag unico perderia esa distincion.
    income_cero = df["MonthlyIncome"] <= INGRESO_MINIMO_PLAUSIBLE
    df["flag_income_cero"] = income_cero.fillna(False).astype(int)
    df["MonthlyIncome"] = df["MonthlyIncome"].where(~income_cero.fillna(False), np.nan)

    # --- 1b. Codigos centinela 96/98 en las variables de morosidad ---------
    # El flag es MUY informativo (54.67% de default), asi que se crea antes
    # de destruir la evidencia.
    es_centinela = df[COLS_MORA].isin(SENTINELAS).any(axis=1)
    df["flag_registro_invalido"] = es_centinela.astype(int)

    # Ahora si: los 96/98 pasan a NaN. Dejarlos como numero haria que la
    # regresion logistica interprete "98 atrasos" como 98 veces la senal
    # de "1 atraso", lo cual no tiene ningun sentido.
    for col in COLS_MORA:
        df[col] = df[col].where(~df[col].isin(SENTINELAS), np.nan)

    # --- 1c. RevolvingUtilization absurdo -----------------------------------
    revutil_malo = df["RevolvingUtilizationOfUnsecuredLines"] > REVUTIL_UMBRAL_INVALIDO
    df["flag_revutil_invalido"] = revutil_malo.astype(int)
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].where(~revutil_malo, np.nan)

    # --- 1d. DebtRatio: separar el ratio del importe ------------------------
    # DebtRatio = deuda_mensual / ingreso_mensual. El criterio para saber si el
    # campo contiene un ratio o un importe NO es su magnitud: es si el sistema
    # de origen TENIA denominador. Sin ingreso no hay division posible, y lo
    # que queda almacenado es el numerador en bruto.
    #
    # Antes se usaba el proxy "DebtRatio > 2". La seccion 5e del EDA mostro que
    # ese proxy confunde dos poblaciones distintas:
    #
    #   955 filas CON ingreso valido y DebtRatio > 2 -> el proxy las descartaba
    #     como dato corrupto, pero son ratios REALES de gente con ingreso bajo:
    #       DR 2-5    n=700  ingreso ~1,300  deuda ~3,678  default 9.43%
    #       DR 5-10   n=152  ingreso ~  639  deuda ~4,189  default 7.24%
    #       DR 10-100 n= 92  ingreso ~  400  deuda ~5,194  default 8.70%
    #     Todas POR ENCIMA de la tasa global (6.68%): senal genuina que se
    #     estaba tirando a deuda_bruta_proxy.
    #
    #   1,732 filas SIN denominador y DebtRatio <= 2 -> el proxy las trataba
    #     como ratio valido. Inofensivo: el 82.4% vale exactamente 0.
    #
    # Nota: 'sin_denominador' equivale exactamente a la union de los dos flags
    # de ingreso. Por eso ya NO se crea flag_debtratio_invalido: seria una
    # combinacion lineal de columnas existentes, es decir colinealidad
    # perfecta, que es el mecanismo que produjo el hallazgo H1.
    sin_denominador = (df["flag_income_faltante"] | df["flag_income_cero"]).astype(bool)

    # El importe en bruto, solo donde no habia con que dividir.
    #
    # El sufijo _proxy es deliberado. Que este campo contenga deuda mensual en
    # dolares es una INFERENCIA nuestra, no un dato del diccionario de Kaggle:
    # se dedujo de que su distribucion coincide en orden de magnitud con la
    # deuda real de las filas normales (DebtRatio x MonthlyIncome). La hipotesis
    # esta bien respaldada, pero no certificada, y el nombre debe decirlo — si
    # se llamara 'deuda_bruta' a secas, cualquiera que leyera el codigo mas
    # adelante la trataria como un hecho.
    df["deuda_bruta_proxy"] = df["DebtRatio"].where(sin_denominador, np.nan)

    # Y el ratio de verdad, solo donde el denominador existia.
    df["DebtRatio_limpio"] = df["DebtRatio"].where(~sin_denominador, np.nan)

    # --- 1e. Edad implausible ------------------------------------------------
    edad_mala = df["age"] < EDAD_MINIMA
    df["flag_edad_invalida"] = edad_mala.astype(int)
    df["age"] = df["age"].where(~edad_mala, np.nan)

    return df


# =============================================================================
# PASO 2 — FEATURE ENGINEERING
# =============================================================================
# Variables nuevas construidas con logica de negocio, no combinaciones al azar.
# Cada una responde a una pregunta que un analista de riesgo se haria.

def crear_features(df):
    """Crea variables derivadas con sentido de negocio."""
    df = df.copy()

    # --- 2a. Resumen del historial de morosidad ------------------------------
    # ¿Cuantos episodios de atraso acumula en total? (NaN de centinelas se
    # ignora con min_count=1, que devuelve NaN si TODO es NaN)
    df["total_atrasos"] = df[COLS_MORA].sum(axis=1, min_count=1)

    # ¿Tiene algun atraso? El EDA mostro que el salto 0->1 es el mas grande
    # de toda la variable, asi que esta pregunta binaria carga mucha senal.
    df["flag_algun_atraso"] = (df["total_atrasos"] > 0).astype(int)

    # ¿Cual fue el atraso MAS GRAVE? La gravedad importa mas que la cantidad:
    # el salto 0->1 en 90+ dias multiplica el riesgo x7.3, mientras que en
    # 30-59 dias solo lo multiplica x3.7.
    #   0 = sin atrasos | 1 = solo 30-59 | 2 = llego a 60-89 | 3 = llego a 90+
    df["peor_atraso"] = 0.0
    df.loc[df["NumberOfTime30-59DaysPastDueNotWorse"] > 0, "peor_atraso"] = 1.0
    df.loc[df["NumberOfTime60-89DaysPastDueNotWorse"] > 0, "peor_atraso"] = 2.0
    df.loc[df["NumberOfTimes90DaysLate"] > 0, "peor_atraso"] = 3.0

    # CUIDADO: en las filas con centinela las tres columnas son NaN, y como
    # "NaN > 0" es False, esas filas se quedarian en peor_atraso = 0, es decir
    # clasificadas como "sin atrasos". Son justo las de MAYOR riesgo (54.67%),
    # asi que contaminarian el grupo mas seguro. Su historial es DESCONOCIDO,
    # no cero: por eso van a NaN (que en WoE se convierte en su propio bin).
    sin_historial = df[COLS_MORA].isna().all(axis=1)
    df.loc[sin_historial, "peor_atraso"] = np.nan

    # --- 2b. Capacidad de pago ----------------------------------------------
    # Ingreso disponible por persona en el hogar: no es lo mismo ganar 5,000
    # viviendo solo que ganar 5,000 manteniendo a 4 dependientes.
    # El +1 evita dividir entre cero e incluye al titular.
    df["ingreso_por_persona"] = df["MonthlyIncome"] / (
        df["NumberOfDependents"].fillna(0) + 1
    )

    # Deuda mensual absoluta = ratio x ingreso. Solo tiene sentido donde
    # DebtRatio es un ratio de verdad y hay ingreso.
    df["deuda_mensual_abs"] = df["DebtRatio_limpio"] * df["MonthlyIncome"]

    # --- 2c. Estructura del credito -----------------------------------------
    # Lineas NO inmobiliarias = tarjetas y prestamos personales (no garantizados).
    # Es distinto tener 10 lineas siendo 8 hipotecas que teniendo 10 tarjetas.
    df["lineas_no_inmobiliarias"] = (
        df["NumberOfOpenCreditLinesAndLoans"] - df["NumberRealEstateLoansOrLines"]
    ).clip(lower=0)

    # Proporcion del credito que esta respaldado por un inmueble.
    df["prop_inmobiliaria"] = df["NumberRealEstateLoansOrLines"] / df[
        "NumberOfOpenCreditLinesAndLoans"
    ].replace(0, np.nan)

    return df


# =============================================================================
# PASO 3A — VERSION PARA ARBOLES (XGBoost / LightGBM)
# =============================================================================

def preparar_arboles(df):
    """
    Devuelve el dataset listo para XGBoost y LightGBM.

    ¿Por que NO capeamos los outliers aqui?
      Los arboles son INVARIANTES a transformaciones monotonas: un corte en
      "RevUtil > 1.5" separa exactamente las mismas filas tanto si el maximo
      es 2 como si es 50,708. Solo les importa el ORDEN, no la magnitud.
      Capear no aporta nada y si destruye granularidad.

    ¿Por que NO imputamos los NaN?
      XGBoost y LightGBM manejan NaN de forma nativa: en cada division
      aprenden hacia que lado mandar los faltantes. Imputar les quitaria
      informacion que ellos aprovechan mejor por su cuenta.
      (Recuerda: en este dataset el faltante ES predictivo.)
    """
    df = df.copy()

    # Quitamos las columnas intermedias que ya estan representadas por otras,
    # para no meter duplicados exactos.
    a_eliminar = ["DebtRatio"]  # sustituida por DebtRatio_limpio + deuda_bruta_proxy + flag
    df = df.drop(columns=[c for c in a_eliminar if c in df.columns])

    return df

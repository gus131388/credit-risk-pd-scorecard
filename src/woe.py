# -*- coding: utf-8 -*-
"""
woe.py — Weight of Evidence e Information Value
Proyecto: Give Me Some Credit — Probability of Default

Modulo separado de preprocessing.py porque WoE es una tecnica autocontenida:
tiene su propia matematica, sus propios artefactos (las tablas aprendidas) y
solo la usa la rama de regresion logistica.

FLUJO DE USO (el orden importa para no filtrar informacion):

    tablas = aprender_woe(X_train, y_train)   # mira el target: SOLO train
    X_tr   = aplicar_woe(X_train, tablas)     # no mira el target
    X_val  = aplicar_woe(X_val,   tablas)     # <- usa lo aprendido en train
    X_test = aplicar_woe(X_test,  tablas)     # <- idem

Recalcular las tablas sobre validacion o test seria una fuga de datos:
esos conjuntos dejarian de ser "nunca vistos".
"""

import numpy as np
import pandas as pd


# WoE (Weight of Evidence) es la tecnica clasica de los scorecards bancarios.
#
# Idea: en vez de meter el valor crudo a la regresion, partimos la variable en
# rangos (bins) y sustituimos cada rango por un numero que mide cuanto se
# desvia ese rango del riesgo promedio.
#
#   WoE_i = ln( %no-eventos en el bin i  /  %eventos en el bin i )
#
#   WoE positivo -> el bin tiene MENOS default que la media  (mas seguro)
#   WoE negativo -> el bin tiene MAS default que la media    (mas riesgoso)
#   WoE = 0      -> el bin se comporta como la media
#
# Ventajas para regresion logistica:
#   1. Captura relaciones NO lineales: LogReg es un modelo lineal y no puede
#      curvarse solo. Con WoE le damos la forma ya codificada.
#      (Ej: en 90+ dias el salto 0->1 es +29 puntos y el 3->4 solo +9.
#       Una recta no puede representar eso; los bins si.)
#   2. Los outliers dejan de importar: solo cuenta en que bin caes.
#   3. Los NaN son un bin mas — no hay que imputar nada.
#   4. Todas las variables quedan en la misma escala (log-odds), lo que hace
#      los coeficientes comparables entre si.
#
# IV (Information Value) resume el poder predictivo de la variable entera:
#   IV = SUM( (%no-eventos - %eventos) * WoE )
#   < 0.02  inutil | 0.02-0.1 debil | 0.1-0.3 medio
#   0.3-0.5 fuerte | > 0.5 sospechoso (revisar si hay fuga de datos)

# --- Definicion de los bins ---------------------------------------------------
# Variables de CONTEO: cortes manuales. Usar cuantiles falla porque hay
# demasiados valores repetidos (ej. el 94% de las filas tiene 0 en 90+ dias).
BINS_CONTEO = {
    "NumberOfTime30-59DaysPastDueNotWorse": [-0.1, 0, 1, 3, np.inf],
    "NumberOfTime60-89DaysPastDueNotWorse": [-0.1, 0, 1, np.inf],
    "NumberOfTimes90DaysLate":              [-0.1, 0, 1, 3, np.inf],
    "NumberOfDependents":                   [-0.1, 0, 1, 2, np.inf],
    "NumberRealEstateLoansOrLines":         [-0.1, 0, 1, 2, np.inf],
    "NumberOfOpenCreditLinesAndLoans":      [-0.1, 3, 5, 8, 12, np.inf],
    "total_atrasos":                        [-0.1, 0, 1, 3, np.inf],
    "peor_atraso":                          [-0.1, 0, 1, 2, np.inf],
    "lineas_no_inmobiliarias":              [-0.1, 2, 4, 7, 11, np.inf],
}

# Variables CONTINUAS: cortes elegidos a partir de las tablas del EDA,
# donde vimos donde cambia realmente el riesgo.
BINS_CONTINUAS = {
    # 0-0.5 = 2.74% | 0.5-0.9 = 12.20% | 0.9-1.0 = 19.31% | 1.0-1.2 = 38.23%
    "RevolvingUtilizationOfUnsecuredLines": [-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, np.inf],
    # deciles: 11.09% en los mas jovenes -> 2.19% en los mayores
    "age": [-0.1, 30, 40, 45, 50, 55, 60, 65, 72, np.inf],
    # deciles: 8.35% en ingresos bajos -> 4.54% en altos
    "MonthlyIncome": [-0.1, 2000, 3000, 4000, 5400, 7000, 9000, 12000, np.inf],
    # regimen valido: 6.05% -> 9.88% -> 13.12%
    "DebtRatio_limpio": [-0.1, 0.2, 0.4, 0.6, 1.0, np.inf],
    "deuda_bruta_proxy": [-0.1, 500, 1500, 3000, np.inf],
    "ingreso_por_persona": [-0.1, 1000, 2000, 3000, 5000, np.inf],
    "deuda_mensual_abs": [-0.1, 500, 1000, 2000, 4000, np.inf],
    "prop_inmobiliaria": [-0.1, 0.001, 0.2, 0.4, np.inf],
}

BINS_TODOS = {**BINS_CONTEO, **BINS_CONTINUAS}

# Las variables que ya son 0/1 no se binean: se usan tal cual.
# Se detectan por su prefijo en vez de listarlas a mano: si listaramos los
# nombres aqui, anadir un flag nuevo en preprocessing.py y olvidar registrarlo
# lo dejaria fuera del dataset SIN ningun error visible. Con el prefijo,
# cualquier flag nuevo entra solo.
PREFIJO_FLAG = "flag_"


def columnas_flag(df):
    """Devuelve las columnas 0/1 que pasan sin transformar."""
    return [c for c in df.columns if c.startswith(PREFIJO_FLAG)]


def _asignar_bins(serie, cortes):
    """
    Convierte una serie numerica en su bin correspondiente.
    Los NaN se convierten en la categoria 'FALTANTE' — en WoE los faltantes
    son un grupo mas, con su propio nivel de riesgo. No hay que imputarlos.
    """
    binned = pd.cut(serie, bins=cortes)
    # astype(object) permite mezclar intervalos con el texto 'FALTANTE'
    binned = binned.astype(object)
    return binned.where(serie.notna(), "FALTANTE")


def _tabla_woe(bins, y, suavizado=0.5):
    """
    Nucleo del calculo: dada una serie YA bineada y el target, devuelve la
    tabla WoE.

    Existe como funcion aparte porque la usan dos consumidores con propositos
    distintos —aprender_woe() para el pipeline y evaluar_binning() para la
    auditoria— y la formula debe ser literalmente la misma en ambos. Si se
    duplicara, una mejora en una no llegaria a la otra.
    """
    total_eventos = int(y.sum())
    total_no_eventos = int((1 - y).sum())

    tabla = pd.DataFrame({"bin": np.asarray(bins), "y": np.asarray(y)})
    g = tabla.groupby("bin", observed=True)["y"].agg(n="size", eventos="sum")
    g["no_eventos"] = g["n"] - g["eventos"]

    # Proporcion de cada bin dentro de su clase (con suavizado)
    g["dist_eventos"] = (g["eventos"] + suavizado) / total_eventos
    g["dist_no_eventos"] = (g["no_eventos"] + suavizado) / total_no_eventos

    # El nucleo del metodo
    g["woe"] = np.log(g["dist_no_eventos"] / g["dist_eventos"])

    # Contribucion de cada bin al Information Value de la variable
    g["iv_parcial"] = (g["dist_no_eventos"] - g["dist_eventos"]) * g["woe"]

    # Para poder leer la tabla facilmente
    g["default_rate_pct"] = (g["eventos"] / g["n"] * 100).round(2)
    g["pct_poblacion"] = (g["n"] / len(tabla) * 100).round(2)

    return g


def aprender_woe(df_train, y_train, suavizado=0.5, bins=None):
    """
    APRENDE las tablas WoE. Solo se llama con el conjunto de ENTRENAMIENTO,
    porque necesita mirar el target.

    suavizado: se suma a los conteos para evitar log(0) cuando un bin no
    tiene ningun evento o ningun no-evento (correccion de Laplace).

    bins: diccionario {variable: cortes}. Por defecto BINS_TODOS, es decir los
    cortes manuales del proyecto. Se puede pasar otro para AUDITAR una
    propuesta alternativa (ver auditar_binning), pero el pipeline de
    produccion nunca lo hace: su comportamiento por defecto no cambia.

    Devuelve: dict {variable: DataFrame con la tabla WoE}
    """
    bins = BINS_TODOS if bins is None else bins
    tablas = {}

    for col, cortes in bins.items():
        if col not in df_train.columns:
            continue
        tablas[col] = _tabla_woe(_asignar_bins(df_train[col], cortes),
                                 y_train, suavizado)

    return tablas


def calcular_iv(tablas_woe):
    """Resume el Information Value de cada variable, ordenado de mayor a menor."""
    filas = []
    for col, g in tablas_woe.items():
        iv = g["iv_parcial"].sum()
        if iv < 0.02:
            fuerza = "inutil"
        elif iv < 0.10:
            fuerza = "debil"
        elif iv < 0.30:
            fuerza = "medio"
        elif iv < 0.50:
            fuerza = "fuerte"
        else:
            fuerza = "sospechoso (revisar)"
        filas.append({"variable": col, "IV": round(iv, 4), "fuerza": fuerza,
                      "n_bins": len(g)})
    return pd.DataFrame(filas).sort_values("IV", ascending=False).reset_index(drop=True)


def aplicar_woe(df, tablas_woe, bins=None):
    """
    APLICA las tablas ya aprendidas. Esta funcion NO mira el target,
    asi que puede usarse con seguridad en validacion y en test.

    Cada variable original se sustituye por su valor WoE.

    bins: debe ser el MISMO diccionario con el que se llamo a aprender_woe().
    Por defecto BINS_TODOS. Pasar unos cortes distintos de los que produjeron
    las tablas asignaria valores WoE a intervalos que no les corresponden.
    """
    bins = BINS_TODOS if bins is None else bins
    salida = pd.DataFrame(index=df.index)

    for col, tabla in tablas_woe.items():
        if col not in df.columns:
            continue

        bins_col = _asignar_bins(df[col], bins[col])
        mapa = tabla["woe"].to_dict()

        # Un bin que no aparecio en train (raro, pero posible) recibe WoE = 0,
        # que equivale a decir "se comporta como la media".
        salida[f"woe_{col}"] = bins_col.map(mapa).fillna(0.0).astype(float)

    # Los flags 0/1 pasan sin transformar: ya estan en una escala interpretable.
    for col in columnas_flag(df):
        salida[col] = df[col].values

    return salida


# =============================================================================
# AUDITOR DE BINNING  (challenger, NO reemplaza los cortes manuales)
# =============================================================================
# Por que existe
#   Los cortes de BINS_CONTEO / BINS_CONTINUAS se eligieron a mano leyendo las
#   tablas de riesgo del EDA. Funcionan, pero tienen un defecto de gobierno:
#   otro analista con los mismos datos elegiria otros cortes y no hay
#   procedimiento documentado para reconstruirlos (hallazgo H6 de la auditoria).
#
#   Este bloque NO corrige eso sustituyendo el criterio humano por un
#   algoritmo. Hace algo mas util: permite CONTRASTAR el binning manual contra
#   uno derivado por procedimiento, y dejar constancia de la diferencia. Si el
#   automatico no mejora, el manual queda respaldado con evidencia en vez de
#   con autoridad.
#
# Que NO hace
#   No toca BINS_TODOS, no se ejecuta en el pipeline y no cambia ningun
#   artefacto. Es una herramienta de analisis que se invoca a mano.
#
# Criterio de decision (en este orden, NO "maximizar IV")
#   estabilidad > muestra suficiente > separacion de riesgo > interpretabilidad
#   > IV. Un IV mayor logrado fragmentando en 15 bins es peor que un IV algo
#   menor en 5 bins estables.

ETIQUETA_FALTANTE = "FALTANTE"


def _separar_faltante(tabla):
    """
    Divide una tabla WoE en (bins numericos ordenados, fila FALTANTE o None).

    Hace falta porque el indice mezcla Intervalos con la cadena 'FALTANTE', y
    ordenar o derivar monotonia sobre esa mezcla no tiene sentido: el bin de
    faltantes no ocupa ninguna posicion en la recta real.
    """
    es_falt = [i == ETIQUETA_FALTANTE for i in tabla.index]
    falt = tabla[es_falt]
    num = tabla[[not e for e in es_falt]]
    if len(num):
        num = num.iloc[np.argsort([iv.left for iv in num.index])]
    return num, (falt.iloc[0] if len(falt) else None)


def clasificar_monotonicidad(default_rates, tolerancia_pp=0.25):
    """
    Clasifica la forma de la relacion riesgo-bin.

    Devuelve: 'creciente' | 'decreciente' | 'U' | 'U_invertida' | 'zigzag' |
              'aproximadamente_constante'

    tolerancia_pp: variaciones menores que esto (en puntos porcentuales) se
    tratan como ruido y no cuentan como cambio de direccion. Sin ella,
    cualquier oscilacion de 0.03 pp clasificaria como zigzag una variable
    perfectamente monotona en la practica.

    NO se fuerza monotonia en el proyecto: hay relaciones genuinamente en U
    (no tener credito abierto es tan mala senal como tener demasiado), y
    aplanarlas destruiria informacion real.
    """
    r = np.asarray(default_rates, dtype=float)
    if len(r) < 2:
        return "aproximadamente_constante"
    if r.max() - r.min() < tolerancia_pp:
        return "aproximadamente_constante"

    d = np.diff(r)
    signos = np.sign(np.where(np.abs(d) < tolerancia_pp, 0.0, d))
    signos = signos[signos != 0]          # los tramos planos no rompen la forma
    if len(signos) == 0:
        return "aproximadamente_constante"

    cambios = int((signos[1:] != signos[:-1]).sum())
    if cambios == 0:
        return "creciente" if signos[0] > 0 else "decreciente"
    if cambios == 1:
        return "U" if signos[0] < 0 else "U_invertida"
    return "zigzag"


def evaluar_binning(x, y, cortes, *, minimo_pct_bin=0.01, minimo_eventos=30,
                    missing_separado=True, suavizado=0.5):
    """
    Evalua un juego de cortes sobre (x, y) y devuelve la tabla completa.

    Reutiliza _asignar_bins y _tabla_woe, es decir exactamente la misma
    matematica que usa el pipeline: la auditoria no puede medir con una regla
    distinta de la que se aplica en produccion.

    missing_separado=False elimina las filas con NaN en vez de darles bin
    propio. Solo tiene sentido para diagnosticar cuanto aporta el bin FALTANTE;
    el pipeline siempre lo mantiene separado.

    Columnas por bin: limite_inf, limite_sup, n, pct_poblacion, eventos,
    no_eventos, default_rate_pct, dist_eventos (bads), dist_no_eventos (goods),
    woe, iv_parcial, cumple_tamano, cumple_eventos.

    El IV total va en tabla.attrs['iv_total'] (es la suma de iv_parcial).
    """
    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)

    if not missing_separado:
        vivos = x.notna()
        x, y = x[vivos].reset_index(drop=True), y[vivos].reset_index(drop=True)

    bins = _asignar_bins(x, cortes)
    g = _tabla_woe(bins, y, suavizado)

    num, _ = _separar_faltante(g)
    orden = list(num.index)
    if ETIQUETA_FALTANTE in g.index:
        orden.append(ETIQUETA_FALTANTE)
    g = g.loc[orden]

    g.insert(0, "limite_sup", [b.right if b != ETIQUETA_FALTANTE else np.nan
                               for b in g.index])
    g.insert(0, "limite_inf", [b.left if b != ETIQUETA_FALTANTE else np.nan
                               for b in g.index])

    n_total = int(g["n"].sum())
    g["cumple_tamano"] = g["n"] >= minimo_pct_bin * n_total
    g["cumple_eventos"] = g["eventos"] >= minimo_eventos

    g.attrs["iv_total"] = float(g["iv_parcial"].sum())
    g.attrs["n_bins"] = int(len(g))
    return g


# --- Generacion automatica de candidatos: ChiMerge ---------------------------
# Se elige ChiMerge sobre optbinning u otras librerias por tres razones:
#   1. No anade dependencias: son ~80 lineas de numpy.
#   2. Es AUDITABLE: cada fusion se justifica con un estadistico que un
#      supervisor puede recalcular a mano.
#   3. No optimiza el IV. Un algoritmo que maximiza IV tiende a sobreajustar
#      los cortes al train; ChiMerge solo fusiona bins cuyo comportamiento es
#      estadisticamente indistinguible, que es una pregunta distinta.

def _chi2_adyacente(n0_a, n1_a, n0_b, n1_b):
    """Chi-cuadrado de la tabla 2x2 formada por dos bins contiguos."""
    obs = np.array([[n0_a, n1_a], [n0_b, n1_b]], dtype=float)
    total = obs.sum()
    if total == 0:
        return 0.0
    filas = obs.sum(axis=1, keepdims=True)
    cols = obs.sum(axis=0, keepdims=True)
    esp = filas @ cols / total
    # Las celdas con esperado 0 no aportan (convencion estandar de ChiMerge:
    # si una clase no aparece en ninguno de los dos bins, no hay evidencia de
    # que difieran en esa clase).
    mask = esp > 0
    return float((((obs - esp) ** 2)[mask] / esp[mask]).sum())


def _prebins(x, n_prebins=20, special_values=None, max_cardinal_discreto=30):
    """
    Cortes iniciales finos sobre los que ChiMerge ira fusionando.

    Discretas de baja cardinalidad -> un bin por valor observado. Continuas ->
    cuantiles. Se usan cuantiles y no anchos iguales porque con distribuciones
    muy sesgadas (aqui, casi todas) los anchos iguales dejan un bin con el 99%
    de la poblacion y el resto vacios.
    """
    v = pd.Series(x).dropna().to_numpy(dtype=float)
    if len(v) == 0:
        raise ValueError("la variable no tiene valores no nulos")

    unicos = np.unique(v)
    if len(unicos) <= max_cardinal_discreto:
        interiores = list(unicos[:-1])                # un bin por valor
    else:
        qs = np.linspace(0, 1, n_prebins + 1)[1:-1]
        interiores = list(np.unique(np.quantile(v, qs)))

    # Los valores especiales necesitan un corte JUSTO por debajo, para quedar
    # encerrados en un intervalo que solo los contiene a ellos.
    for esp in (special_values or []):
        if esp not in unicos:
            continue
        menores = unicos[unicos < esp]
        izq = float(menores.max()) if len(menores) else -np.inf
        for c in (izq, float(esp)):
            if np.isfinite(c):
                interiores.append(c)

    interiores = sorted(set(float(c) for c in interiores))
    return [-np.inf] + interiores + [np.inf]


def chimerge(x, y, *, min_bins=3, max_bins=8, minimo_pct_bin=0.01,
             minimo_eventos=30, special_values=None, n_prebins=20,
             forzar_monotonia=False):
    """
    Deriva cortes por fusion iterativa de bins contiguos (ChiMerge).

    Procedimiento:
      1. Prebins finos (cuantiles o valores naturales).
      2. FASE A - se fusionan los bins que incumplen tamano minimo o minimo de
         eventos, empezando por el mas pequeno, con el vecino mas parecido.
      3. FASE B - mientras haya mas de max_bins, se fusiona el par contiguo con
         chi-cuadrado mas bajo, es decir el par cuya diferencia de riesgo es
         menos distinguible del azar.
      4. FASE C (opcional) - si se pide monotonia, se fusionan los pares que
         la rompen, empezando por la violacion mas pequena.

    special_values: valores que NUNCA se fusionan con sus vecinos. Pensado para
    el cero de los contadores de mora: 0 atrasos -> ~4% de evento y 1 atraso ->
    ~15%, un salto economicamente decisivo que ningun criterio estadistico
    generico deberia poder borrar.

    forzar_monotonia: False (defecto) | 'increasing' | 'decreasing'.
    El defecto es no forzarla porque varias variables del proyecto tienen forma
    de U legitima. Si se pide y choca con min_bins, la monotonia tiene
    prioridad (suelo absoluto: 2 bins) — ver FASE C.

    Devuelve: lista de cortes utilizable con pd.cut / _asignar_bins.
    """
    if forzar_monotonia not in (False, "increasing", "decreasing"):
        raise ValueError("forzar_monotonia debe ser False, 'increasing' o 'decreasing'")
    if min_bins > max_bins:
        raise ValueError("min_bins no puede superar max_bins")

    xs = pd.Series(x).reset_index(drop=True)
    ys = pd.Series(np.asarray(y)).reset_index(drop=True)
    vivos = xs.notna()
    xs = xs[vivos].to_numpy(dtype=float)
    ys = ys[vivos].to_numpy(dtype=float)

    cortes = _prebins(xs, n_prebins, special_values)
    idx = np.clip(np.searchsorted(cortes, xs, side="left") - 1, 0, len(cortes) - 2)

    # bins[i] = [borde_izq, borde_der, n_no_eventos, n_eventos, bloqueado]
    especiales = set(float(v) for v in (special_values or []))
    bins = []
    for i in range(len(cortes) - 1):
        m = idx == i
        n1 = float(ys[m].sum())
        # Un bin esta BLOQUEADO si contiene un valor especial: no se fusiona
        # nunca, ni siquiera para cumplir el tamano minimo.
        bloq = any(cortes[i] < e <= cortes[i + 1] for e in especiales)
        bins.append([cortes[i], cortes[i + 1], int(m.sum()) - n1, n1, bloq])

    # Los prebins vacios no aportan nada y estorban al calcular chi2.
    bins = [b for b in bins if (b[2] + b[3]) > 0 or b[4]]
    n_total = len(xs)

    def fusionar(i):
        a, b = bins[i], bins[i + 1]
        bins[i:i + 2] = [[a[0], b[1], a[2] + b[2], a[3] + b[3], a[4] or b[4]]]

    def pares_fusionables():
        return [i for i in range(len(bins) - 1)
                if not bins[i][4] and not bins[i + 1][4]]

    def chi_de(i):
        return _chi2_adyacente(bins[i][2], bins[i][3], bins[i + 1][2], bins[i + 1][3])

    def viola(b):
        return (b[2] + b[3]) < minimo_pct_bin * n_total or b[3] < minimo_eventos

    # --- FASE A: tamano minimo -----------------------------------------------
    while len(bins) > min_bins:
        malos = [i for i, b in enumerate(bins) if viola(b) and not b[4]]
        if not malos:
            break
        # el mas pequeno primero: es el que menos informacion pierde al fusionar
        i = min(malos, key=lambda k: bins[k][2] + bins[k][3])
        vecinos = [j for j in (i - 1, i) if j in pares_fusionables()]
        if not vecinos:
            break
        fusionar(min(vecinos, key=chi_de))

    # --- FASE B: max_bins ----------------------------------------------------
    while len(bins) > max_bins:
        cand = pares_fusionables()
        if not cand:
            break
        fusionar(min(cand, key=chi_de))

    # --- FASE C: monotonia opcional -----------------------------------------
    # PRECEDENCIA: si la monotonia choca con min_bins, gana la monotonia, con
    # un suelo absoluto de 2 bins. Motivo: min_bins es una preferencia de
    # granularidad, mientras que forzar_monotonia es una restriccion explicita
    # de FORMA. Respetar min_bins aqui devolveria un resultado no monotono
    # despues de que el usuario pidiera monotonia, que es la peor combinacion:
    # incumple lo pedido y ademas no avisa.
    if forzar_monotonia:
        creciente = forzar_monotonia == "increasing"
        while len(bins) > 2:
            tasas = [b[3] / (b[2] + b[3]) if (b[2] + b[3]) else 0.0 for b in bins]
            d = np.diff(tasas)
            fusionables = pares_fusionables()
            malos = [i for i in range(len(d))
                     if (d[i] < 0 if creciente else d[i] > 0) and i in fusionables]
            if not malos:
                break
            fusionar(min(malos, key=lambda i: abs(d[i])))

    return [b[0] for b in bins] + [bins[-1][1]]


def score_parsimonia(iv, n_bins, lambda_bins=0.01):
    """
    IV penalizado por numero de bins. INDICADOR DESCRIPTIVO, no un criterio.

    Sirve para detectar el caso 5 bins -> IV 0.74 frente a 15 bins -> IV 0.78,
    donde la mejora no compensa la fragmentacion. El valor de lambda_bins es
    convencional: no lo respalda ninguna teoria, solo hace comparables dos
    alternativas del mismo problema.
    """
    return float(iv) - lambda_bins * int(n_bins)


def comparar_binnings(x, y, cortes_manual, cortes_auto, *, minimo_pct_bin=0.01,
                      minimo_eventos=30, umbral_iv_marginal=0.02,
                      lambda_bins=0.01):
    """
    Compara dos juegos de cortes lado a lado.

    NO declara ganador. Devuelve las metricas y una lectura del incremento de
    IV, porque la decision depende de criterios que no caben en un numero
    (interpretabilidad, politica de riesgos, capacidad de explicar el corte a
    un supervisor).

    umbral_iv_marginal: mejora RELATIVA de IV por debajo de la cual se declara
    marginal. 2% por defecto: con ese margen la diferencia suele quedar dentro
    de la variacion entre folds.
    """
    ev = {"manual": evaluar_binning(x, y, cortes_manual,
                                    minimo_pct_bin=minimo_pct_bin,
                                    minimo_eventos=minimo_eventos),
          "automatico": evaluar_binning(x, y, cortes_auto,
                                        minimo_pct_bin=minimo_pct_bin,
                                        minimo_eventos=minimo_eventos)}

    filas = {}
    for nombre, g in ev.items():
        num, _ = _separar_faltante(g)
        filas[nombre] = {
            "n_bins": g.attrs["n_bins"],
            "IV": round(g.attrs["iv_total"], 4),
            "bin_min_pct": f"{g['n'].min() / g['n'].sum() * 100:.2f}%",
            "min_eventos": int(g["eventos"].min()),
            "monotonicidad": clasificar_monotonicidad(num["default_rate_pct"]),
            "bins_bajo_minimo": int((~g["cumple_tamano"]).sum()),
            "score_parsimonia": round(
                score_parsimonia(g.attrs["iv_total"], g.attrs["n_bins"],
                                 lambda_bins), 4),
        }

    tabla = pd.DataFrame(filas)

    iv_m = ev["manual"].attrs["iv_total"]
    iv_a = ev["automatico"].attrs["iv_total"]
    rel = (iv_a - iv_m) / iv_m if iv_m > 0 else np.nan
    tabla.attrs["delta_iv"] = round(iv_a - iv_m, 4)
    tabla.attrs["delta_iv_relativo"] = round(rel, 4)
    tabla.attrs["iv_marginal"] = bool(abs(rel) < umbral_iv_marginal)
    tabla.attrs["tablas"] = ev
    return tabla


def evaluar_estabilidad_binning(X, y, variable, cortes, cv=5, random_state=42,
                                minimo_pct_bin=0.01, minimo_eventos=30):
    """
    Mide si unos cortes producen la MISMA lectura del riesgo en submuestras.

    Un binning puede tener un IV excelente y aun asi ser inservible: si el WoE
    de un bin oscila mucho entre particiones, ese bin describe la muestra, no
    la poblacion. Esta funcion implementa el criterio "estabilidad por encima
    de IV".

    IMPORTANTE: opera SOLO dentro del conjunto que se le pase, que debe ser el
    de entrenamiento. No recibe ni necesita validacion ni test.

    Devuelve: (resumen_por_bin, dict con iv_medio / iv_desv / ivs)
    """
    from sklearn.model_selection import StratifiedKFold

    x = pd.Series(X[variable] if isinstance(X, pd.DataFrame) else X)
    x = x.reset_index(drop=True)
    ys = pd.Series(np.asarray(y)).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    ivs, registros = [], []

    for k, (idx, _) in enumerate(skf.split(x.to_frame(), ys)):
        g = evaluar_binning(x.iloc[idx], ys.iloc[idx], cortes,
                            minimo_pct_bin=minimo_pct_bin,
                            minimo_eventos=minimo_eventos)
        ivs.append(g.attrs["iv_total"])
        for b in g.index:
            registros.append({"fold": k, "bin": str(b),
                              "woe": g.loc[b, "woe"],
                              "default_rate_pct": g.loc[b, "default_rate_pct"],
                              "n": g.loc[b, "n"],
                              "pct": g.loc[b, "n"] / g["n"].sum() * 100,
                              "cumple": bool(g.loc[b, "cumple_tamano"]
                                             and g.loc[b, "cumple_eventos"])})

    det = pd.DataFrame(registros)
    resumen = det.groupby("bin", sort=False).agg(
        woe_medio=("woe", "mean"), woe_desv=("woe", "std"),
        woe_min=("woe", "min"), woe_max=("woe", "max"),
        default_desv=("default_rate_pct", "std"),
        pct_min=("pct", "min"),
        folds_incumplen=("cumple", lambda s: int((~s).sum())),
    ).round(4)
    resumen["rango_woe"] = (resumen["woe_max"] - resumen["woe_min"]).round(4)

    return resumen, {"iv_medio": float(np.mean(ivs)),
                     "iv_desv": float(np.std(ivs, ddof=1)) if len(ivs) > 1 else 0.0,
                     "ivs": [round(v, 4) for v in ivs]}


def benchmark_binning_cv(X, y, bins_a, bins_b, *,
                         nombres=("manual", "automatico"),
                         cv=5, random_state=42, C=1.0):
    """
    Compara dos juegos de cortes entrenando LogReg+WoE en validacion cruzada.

    Es la unica comparacion que mide lo que de verdad importa (el rendimiento
    del modelo) y no una propiedad de la variable aislada.

    ANTI-FUGA: dentro de cada fold las tablas WoE se aprenden SOLO con la parte
    de entrenamiento del fold y se aplican a la de prueba. Aprenderlas fuera
    del bucle contaminaria la estimacion. Todo ocurre dentro del train del
    proyecto: validacion y test no intervienen.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X = X.reset_index(drop=True)
    ys = pd.Series(np.asarray(y)).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    filas = []
    for nombre, bins in zip(nombres, (bins_a, bins_b)):
        aucs = []
        for tr, te in skf.split(X, ys):
            tablas = aprender_woe(X.iloc[tr], ys.iloc[tr], bins=bins)
            Xtr = aplicar_woe(X.iloc[tr], tablas, bins=bins)
            Xte = aplicar_woe(X.iloc[te], tablas, bins=bins)[Xtr.columns]
            m = LogisticRegression(max_iter=1000, C=C).fit(Xtr, ys.iloc[tr])
            aucs.append(roc_auc_score(ys.iloc[te], m.predict_proba(Xte)[:, 1]))

        tablas_full = aprender_woe(X, ys, bins=bins)
        iv_total = sum(t["iv_parcial"].sum() for t in tablas_full.values())
        filas.append({"metodo": nombre,
                      "AUC_CV": round(float(np.mean(aucs)), 5),
                      "SD": round(float(np.std(aucs, ddof=1)), 5),
                      "IV_total": round(float(iv_total), 4),
                      "n_bins_total": int(sum(len(t) for t in tablas_full.values()))})

    return pd.DataFrame(filas)


def auditar_binning(X, y, variable, cortes_manual=None, *, special_values=None,
                    min_bins=3, max_bins=8, minimo_pct_bin=0.01,
                    minimo_eventos=30, forzar_monotonia=False, n_prebins=20,
                    cv=5, random_state=42, estabilidad=True):
    """
    Audita el binning de UNA variable: evalua el manual, deriva una propuesta
    automatica, los compara y mide estabilidad.

    Uso tipico (siempre sobre el TRAIN del proyecto):

        r = woe.auditar_binning(X_tr_c, y_tr, "NumberOfTimes90DaysLate",
                                special_values=[0])
        print(r["comparacion"])
        for a in r["advertencias"]:
            print(a)

    Devuelve un dict con: variable, evaluacion_manual, cortes_auto,
    evaluacion_auto, comparacion, advertencias y (si estabilidad=True)
    estabilidad_manual y estabilidad_auto.
    """
    if cortes_manual is None:
        cortes_manual = BINS_TODOS[variable]

    x = X[variable] if isinstance(X, pd.DataFrame) else pd.Series(X)

    ev_man = evaluar_binning(x, y, cortes_manual, minimo_pct_bin=minimo_pct_bin,
                             minimo_eventos=minimo_eventos)
    cortes_auto = chimerge(x, y, min_bins=min_bins, max_bins=max_bins,
                           minimo_pct_bin=minimo_pct_bin,
                           minimo_eventos=minimo_eventos,
                           special_values=special_values, n_prebins=n_prebins,
                           forzar_monotonia=forzar_monotonia)
    ev_auto = evaluar_binning(x, y, cortes_auto, minimo_pct_bin=minimo_pct_bin,
                              minimo_eventos=minimo_eventos)
    comp = comparar_binnings(x, y, cortes_manual, cortes_auto,
                             minimo_pct_bin=minimo_pct_bin,
                             minimo_eventos=minimo_eventos)

    # --- Advertencias: lo que un numero solo no comunica ---------------------
    adv = []
    for etiqueta, g in (("manual", ev_man), ("automatico", ev_auto)):
        chicos = [str(b) for b in g.index[~g["cumple_tamano"]]]
        pocos = [str(b) for b in g.index[~g["cumple_eventos"]]]
        if chicos:
            adv.append(f"[{etiqueta}] bins por debajo del {minimo_pct_bin:.0%} "
                       f"de poblacion: {chicos}")
        if pocos:
            adv.append(f"[{etiqueta}] bins con menos de {minimo_eventos} "
                       f"eventos: {pocos}")
    if comp.attrs["iv_marginal"]:
        adv.append(f"La diferencia de IV es MARGINAL ({comp.attrs['delta_iv']:+.4f}, "
                   f"{comp.attrs['delta_iv_relativo']:+.1%}): no justifica por si "
                   f"sola cambiar de binning.")
    if (comp.loc["n_bins", "automatico"] > comp.loc["n_bins", "manual"]
            and comp.attrs["delta_iv"] > 0):
        adv.append("El automatico gana IV usando MAS bins. Comparar "
                   "score_parsimonia antes de concluir que es mejor.")
    if special_values:
        adv.append(f"Valores especiales aislados por diseno: {special_values}. "
                   f"Nunca se fusionan, aunque incumplan el tamano minimo.")

    salida = {"variable": variable,
              "evaluacion_manual": ev_man,
              "cortes_auto": cortes_auto,
              "evaluacion_auto": ev_auto,
              "comparacion": comp,
              "advertencias": adv}

    if estabilidad:
        salida["estabilidad_manual"] = evaluar_estabilidad_binning(
            x.to_frame(), y, variable, cortes_manual, cv=cv,
            random_state=random_state, minimo_pct_bin=minimo_pct_bin,
            minimo_eventos=minimo_eventos)
        salida["estabilidad_auto"] = evaluar_estabilidad_binning(
            x.to_frame(), y, variable, cortes_auto, cv=cv,
            random_state=random_state, minimo_pct_bin=minimo_pct_bin,
            minimo_eventos=minimo_eventos)

    return salida

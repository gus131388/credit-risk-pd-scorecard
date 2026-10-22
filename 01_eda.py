# -*- coding: utf-8 -*-
"""
01_eda.py — Analisis Exploratorio de Datos (EDA)
Proyecto: Give Me Some Credit — Probability of Default

Objetivo de este script: ENTENDER los datos antes de tocarlos.
No limpiamos nada aqui, solo miramos y diagnosticamos.

AMBITO: solo las 120,000 filas de ENTRENAMIENTO. El split se hace en la PARTE 1,
antes de calcular nada, con la misma semilla que 02_preprocessing.py. Las 30,000
filas de validacion no se miran en ningun punto de este analisis.

Se ejecuta con:  python 01_eda.py
Guarda las graficas en la carpeta ./outputs/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")          # backend "sin ventana": guarda imagenes en disco en vez de abrir popups
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# El EDA no usa nada del pipeline salvo esta funcion estadistica, que tambien
# necesita 02_preprocessing.py. Vive en src/ para que la formula exista una
# sola vez y quede cubierta por los tests.
sys.path.insert(0, str(Path(__file__).parent))
from src.evaluation import ic_wilson


def _formato_eje(v, _pos=None):
    """Formatea numeros grandes en los ejes: 500000 -> 500K, 3000000 -> 3M."""
    a = abs(v)
    if a >= 1_000_000:
        return f"{v/1_000_000:g}M"
    if a >= 1_000:
        return f"{v/1_000:g}K"
    return f"{v:g}"

# En Windows la consola a veces no usa UTF-8 y rompe con acentos. Esto lo evita.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Opciones de visualizacion de pandas: que no corte columnas ni use notacion cientifica
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

sns.set_theme(style="whitegrid")


# =============================================================================
# PARTE 0 — RUTAS
# =============================================================================
# Path(__file__) = la ruta de ESTE archivo. .parent = la carpeta que lo contiene.
# Usamos rutas relativas para que el script funcione en cualquier computadora.
AQUI = Path(__file__).parent
DATA = AQUI.parent.parent / "data"      # subimos 2 niveles y entramos a /data
OUTPUTS = AQUI / "outputs"
OUTPUTS.mkdir(exist_ok=True)

TARGET = "SeriousDlqin2yrs"             # el nombre de la variable que queremos predecir
ID = "row_id"                           # nombre del identificador (indice del CSV)

# --- Parametros del split ----------------------------------------------------
# ATENCION: estos tres valores DEBEN coincidir con los de 02_preprocessing.py.
# Si se cambian aqui y no alla (o al reves), el EDA describiria una particion
# distinta de la que se entrena, y las conclusiones dejarian de aplicar.
SEMILLA = 42
PCT_VALIDACION = 0.20


def titulo(texto):
    """Imprime un separador visual para que la salida sea legible."""
    print("\n" + "=" * 79)
    print(texto)
    print("=" * 79)


# =============================================================================
# PARTE 1 — CARGA Y VISTAZO GENERAL
# =============================================================================
titulo("PARTE 1 — CARGA Y VISTAZO GENERAL")

# index_col=0 -> la primera columna del CSV es solo un numero de fila, no un dato.
# Le decimos a pandas que la use como indice en vez de tratarla como variable.
train_completo = pd.read_csv(DATA / "cs-training.csv", index_col=0)
train_completo.index.name = ID

# --- SPLIT ANTES DE MIRAR NADA ----------------------------------------------
# El EDA se hace SOLO sobre las filas de entrenamiento. Las de validacion no se
# miran ni una vez.
#
# ¿Por que, si el EDA no ajusta ningun modelo?
#   Porque el EDA es donde se toman las DECISIONES: que codigos son centinela,
#   donde cortar una cola, que flags crear. Varias de esas decisiones se apoyan
#   en vistas que involucran el target (tasa de default por decil, por patron de
#   faltantes). Si esas vistas incluyen las filas de validacion, entonces esas
#   filas participaron en el diseno del modelo y ya no son un juez imparcial.
#   La contaminacion no entra por un .fit(), entra por el analista.
#
# La llamada replica EXACTAMENTE la de 02_preprocessing.py: mismos datos, mismo
# stratify, misma semilla. train_test_split es determinista, asi que las 120,000
# filas de aqui son las mismas 120,000 que se entrenan despues.
y_completo = train_completo[TARGET]
X_completo = train_completo.drop(columns=[TARGET])

X_tr, _X_val, y_tr, _y_val = train_test_split(
    X_completo, y_completo,
    test_size=PCT_VALIDACION, stratify=y_completo, random_state=SEMILLA,
)

# Se reconstruye el DataFrame con el target en su posicion original (primera
# columna), para que las tablas y la matriz de correlaciones salgan igual.
train = X_tr.copy()
train.insert(0, TARGET, y_tr)

print(f"Fichero cs-training.csv : {len(train_completo):,} filas")
print(f"Split (semilla {SEMILLA}) -> ENTRENAMIENTO: {len(train):,} filas"
      f"   |   validacion reservada: {len(_X_val):,} filas")
print("\nA partir de aqui, TODO el analisis usa solo las filas de entrenamiento.")
print(f"\nFilas:    {train.shape[0]:,}")
print(f"Columnas: {train.shape[1]}")

print("\nPrimeras 5 filas:")
print(train.head())

print("\nTipos de dato de cada columna:")
# int64  = numero entero
# float64 = numero decimal (pandas usa float cuando hay valores faltantes NaN)
print(train.dtypes)


# =============================================================================
# PARTE 2 — LA VARIABLE OBJETIVO (el "target")
# =============================================================================
titulo("PARTE 2 — LA VARIABLE OBJETIVO: SeriousDlqin2yrs")

# value_counts() cuenta cuantas veces aparece cada valor.
# normalize=True lo convierte en proporcion (0 a 1) en vez de conteo.
conteo = train[TARGET].value_counts().sort_index()
proporcion = train[TARGET].value_counts(normalize=True).sort_index()

resumen_target = pd.DataFrame({
    "conteo": conteo,
    "porcentaje": (proporcion * 100).round(2),
})
resumen_target.index = ["0 = NO cayo en mora", "1 = SI cayo en mora"]
print(resumen_target)

tasa_default = train[TARGET].mean()
print(f"\nTasa de default global: {tasa_default:.4f}  ({tasa_default*100:.2f}%)")
print(f"Ratio de desbalance: 1 default por cada {(1-tasa_default)/tasa_default:.1f} no-defaults")


# =============================================================================
# PARTE 3 — ESTADISTICOS DESCRIPTIVOS
# =============================================================================
titulo("PARTE 3 — ESTADISTICOS DESCRIPTIVOS")

# .describe() calcula para cada columna numerica:
#   count = cuantos valores NO faltantes hay
#   mean  = promedio
#   std   = desviacion estandar (que tan dispersos estan los datos)
#   min   = valor minimo
#   25%   = primer cuartil  (25% de los datos esta por debajo de este valor)
#   50%   = mediana         (el valor del medio)
#   75%   = tercer cuartil
#   max   = valor maximo
#
# Anadimos percentiles extremos (95, 99, 99.5, 99.9): describe() por defecto
# salta del 75% al maximo, y es justo en ese tramo donde vive la cola.
# .T = transpuesta: pone las variables como filas (mas facil de leer con muchas columnas)
print(train.describe(percentiles=[0.25, 0.50, 0.75, 0.95, 0.99, 0.995, 0.999]).T)

print("\nLECTURA CLAVE: compara la columna '50%' (mediana) con 'max'.")
print("Si el maximo es miles de veces mayor que la mediana -> hay outliers extremos.")


# =============================================================================
# PARTE 4 — VALORES FALTANTES (missing values / NaN)
# =============================================================================
titulo("PARTE 4 — VALORES FALTANTES")

# isna() devuelve True/False por celda. .sum() cuenta los True. .mean() da la proporcion.
faltantes = pd.DataFrame({
    "n_faltantes": train.isna().sum(),
    "pct_faltantes": (train.isna().mean() * 100).round(2),
})
faltantes = faltantes[faltantes["n_faltantes"] > 0].sort_values("n_faltantes", ascending=False)

if faltantes.empty:
    print("No hay valores faltantes.")
else:
    print(faltantes)

# --- Pregunta clave de credit scoring -------------------------------------
# ¿El hecho de que un valor FALTE es en si mismo informativo?
# Si la gente sin ingreso reportado cae en mora mas seguido, entonces
# "no reporto ingreso" es una senal predictiva, y crearemos una variable flag.
print("\n¿Es informativo el que un valor falte? (tasa de default segun si falta o no)")
for col in faltantes.index:
    falta = train[col].isna()
    tasa_si_falta = train.loc[falta, TARGET].mean()
    tasa_no_falta = train.loc[~falta, TARGET].mean()
    print(f"\n  {col}:")
    print(f"    Default rate cuando FALTA:    {tasa_si_falta*100:6.2f}%  (n={falta.sum():,})")
    print(f"    Default rate cuando NO falta: {tasa_no_falta*100:6.2f}%  (n={(~falta).sum():,})")
    print(f"    Diferencia: {(tasa_si_falta - tasa_no_falta)*100:+.2f} puntos porcentuales")

# --- Combinaciones de faltantes ---------------------------------------------
# Analizar cada columna por separado no dice si los faltantes se solapan.
# Aqui se agrupa por PATRON: que combinacion exacta de columnas falta en cada
# fila. Si la tasa de default cambia entre patrones, un solo indicador binario
# no basta: colapsaria grupos con riesgos distintos.
if not faltantes.empty:
    cols_falt = list(faltantes.index)
    patron = train[cols_falt].isna()

    # Se codifica cada fila como un entero (mascara de bits) con un producto
    # escalar vectorizado, en vez de recorrer las filas con .apply(axis=1).
    # Con 2 columnas la diferencia no se nota; con 5 el .apply tarda ~2.5s y
    # esto es instantaneo, porque las etiquetas se construyen una sola vez
    # por patron (a lo sumo 2^k) y no una vez por fila (150,000).
    #
    #   pesos  = [1, 2, 4, 8, ...]   una potencia de 2 por columna
    #   codigo = producto escalar    -> un entero unico por combinacion
    #            [True, False] . [1, 2] = 1   (falta solo la 1a)
    #            [True, True ] . [1, 2] = 3   (faltan ambas)
    pesos = 1 << np.arange(len(cols_falt))
    codigo = patron.to_numpy().dot(pesos)

    observadas = train.groupby(codigo)[TARGET].agg(n="size", defaults="sum")

    # --- Espacio COMPLETO de patrones: 2^k ------------------------------------
    # groupby solo devuelve las combinaciones que aparecen en los datos. Pero
    # que una combinacion NO aparezca es un hallazgo en si mismo: significa que
    # esas columnas nunca faltan juntas (o nunca por separado).
    # Se reindexa sobre los 2^k codigos posibles para que los ausentes salgan
    # explicitamente con n = 0, en vez de tener que deducirlos.
    n_patrones_posibles = 2 ** len(cols_falt)
    combinaciones = observadas.reindex(range(n_patrones_posibles), fill_value=0)

    # Etiqueta compacta: una posicion por columna. X = falta, . = presente.
    # Escala a cualquier numero de columnas sin desbordar el ancho de la tabla.
    combinaciones.index = [
        "".join("X" if c & (1 << i) else "." for i in range(len(cols_falt)))
        for c in combinaciones.index
    ]

    # La tasa solo existe donde hay filas; con n = 0 queda como NaN.
    combinaciones["default_rate"] = np.where(
        combinaciones["n"] > 0,
        combinaciones["defaults"] / combinaciones["n"].replace(0, np.nan),
        np.nan)
    combinaciones["pct_poblacion"] = (combinaciones["n"] / len(train) * 100).round(2)
    combinaciones["default_pct"] = (combinaciones["default_rate"] * 100).round(2)

    # Desviacion respecto a la tasa global: dice de un vistazo si el patron
    # esta por encima o por debajo del promedio, sin tener que restar a mano.
    combinaciones["vs_global_pp"] = (
        (combinaciones["default_rate"] - tasa_default) * 100).round(2)

    # --- Umbral de fiabilidad, calibrado con la tasa base ---------------------
    # Para estimar una proporcion con sentido hacen falta ~10 eventos esperados.
    # Cuantas filas se necesitan depende de lo raro que sea el evento:
    #     n_minimo = 10 / tasa_base
    # Con 6.68% salen 150 filas; si la tasa fuera del 1%, harian falta 1,000.
    # Se recalcula con los datos que haya, no es una constante escrita a mano.
    EVENTOS_MINIMOS = 10
    n_minimo = int(np.ceil(EVENTOS_MINIMOS / tasa_default))
    combinaciones["fiable"] = np.select(
        [combinaciones["n"] == 0, combinaciones["n"] >= n_minimo],
        ["NO OBSERVADO", "si"],
        default="NO (n bajo)")

    # --- Intervalo de confianza de cada tasa ---------------------------------
    # Intervalo de Wilson (ver la funcion ic_wilson arriba). Los patrones no
    # observados tienen n = 0 y salen como NaN, que es lo correcto: el
    # intervalo de un grupo vacio no existe.
    inf_, sup_ = ic_wilson(combinaciones["default_rate"].to_numpy(dtype=float),
                           combinaciones["n"].to_numpy(dtype=float))

    combinaciones["ic95_inf"] = (inf_ * 100).round(2)
    combinaciones["ic95_sup"] = (sup_ * 100).round(2)
    combinaciones["ancho_ic"] = ((sup_ - inf_) * 100).round(2)

    combinaciones = (combinaciones.drop(columns=["default_rate"])
                     .sort_values("n", ascending=False))
    combinaciones.index.name = "patron"

    no_observados = (combinaciones["n"] == 0).sum()

    print("\n\nCOMBINACIONES DE FALTANTES (¿se solapan? ¿cambia el riesgo?)")
    print(f"  Patrones posibles: 2^{len(cols_falt)} = {n_patrones_posibles}"
          f"   |   observados: {n_patrones_posibles - no_observados}"
          f"   |   nunca ocurren: {no_observados}")
    print("\n  Posicion en el patron (X = falta, . = presente):")
    for i, c in enumerate(cols_falt):
        print(f"    {i+1}. {c}")
    print()

    # Con muchas columnas 2^k explota (10 columnas -> 1,024 filas). Se listan
    # todos solo mientras la tabla siga siendo legible.
    LIMITE_FILAS = 64
    if n_patrones_posibles <= LIMITE_FILAS:
        print(combinaciones.to_string(na_rep="-"))
    else:
        print(combinaciones[combinaciones["n"] > 0].to_string(na_rep="-"))
        print(f"    ... se omiten {no_observados} patrones con n = 0 "
              f"(2^{len(cols_falt)} supera el limite de {LIMITE_FILAS} filas)")

    if no_observados:
        vacios = list(combinaciones.index[combinaciones["n"] == 0])
        print(f"\n  Patrones que NUNCA ocurren ({no_observados}): "
              f"{', '.join(vacios[:12])}{' ...' if len(vacios) > 12 else ''}")
        print("    Su ausencia es un hallazgo: esas columnas no faltan en esa"
              " combinacion.")

    # ¿Hay anidamiento? Una columna esta contenida en otra si nunca falta sola.
    print("\n  Solapamiento entre columnas:")
    for a in faltantes.index:
        for b in faltantes.index:
            if a == b:
                continue
            solo_a = (patron[a] & ~patron[b]).sum()
            ambas = (patron[a] & patron[b]).sum()
            if patron[a].sum() > 0 and solo_a == 0:
                print(f"    {a} SIEMPRE falta junto a {b} "
                      f"({ambas:,} de {patron[a].sum():,} = 100%)")

    # --- ¿El rango supera al ruido de muestreo? ------------------------------
    # El rango se calcula SOLO con los patrones fiables: incluir un grupo de
    # 12 filas con 25% de default lo inflaria con puro ruido.
    #
    # El umbral NO es una constante elegida a ojo: es el ancho del intervalo
    # de confianza mas grande entre los patrones comparados. Se calibra solo:
    #   - grupos grandes  -> intervalos estrechos -> umbral bajo
    #   - grupos pequenos -> intervalos anchos    -> umbral alto
    # Si el rango no supera ese ancho, las diferencias caben dentro del ruido.
    fiables = combinaciones[combinaciones["fiable"] == "si"]
    rango = fiables["default_pct"].max() - fiables["default_pct"].min()
    umbral_ruido = fiables["ancho_ic"].max()

    print(f"\n  Fiabilidad: hacen falta >= {n_minimo:,} filas por patron")
    print(f"    ({EVENTOS_MINIMOS} eventos esperados con la tasa base de "
          f"{tasa_default*100:.2f}%)")
    print(f"    Patrones fiables: {len(fiables)} de "
          f"{(combinaciones['n'] > 0).sum()} observados "
          f"({n_patrones_posibles} posibles)")

    print(f"\n  INDICADOR DESCRIPTIVO (no es una prueba estadistica)")
    print(f"    Rango de default entre patrones fiables: {rango:.2f} pp")
    print(f"    Umbral de ruido (IC mas ancho)         : {umbral_ruido:.2f} pp")
    if rango > umbral_ruido:
        print(f"    -> el rango supera el ruido ({rango:.2f} > {umbral_ruido:.2f}):"
              " indicio de que los patrones difieren")
    else:
        print(f"    -> el rango no supera el ruido ({rango:.2f} <= {umbral_ruido:.2f}):"
              " sin indicios de diferencia")
    print("    Es una heuristica de tamano, no controla error tipo I ni")
    print("    comparaciones multiples. El contraste formal viene despues.")

    # Comparacion por pares: dos patrones son distinguibles si sus intervalos
    # no se solapan. Es una conclusion directa, sin umbral de por medio.
    print("\n  Pares de patrones con intervalos que NO se solapan:")
    pares_separados = 0
    idx = list(fiables.index)
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = fiables.loc[idx[i]], fiables.loc[idx[j]]
            if a["ic95_sup"] < b["ic95_inf"] or b["ic95_sup"] < a["ic95_inf"]:
                print(f"    {idx[i]} [{a['ic95_inf']:.2f}, {a['ic95_sup']:.2f}]  vs  "
                      f"{idx[j]} [{b['ic95_inf']:.2f}, {b['ic95_sup']:.2f}]")
                pares_separados += 1
    total_pares = len(idx) * (len(idx) - 1) // 2
    if pares_separados == 0:
        print("    ninguno -> todos los patrones podrian tener la misma tasa")
    else:
        print(f"    {pares_separados} de {total_pares} pares son distinguibles")

    # --- ¿Podrian estas diferencias deberse al azar? -------------------------
    # Prueba chi-cuadrado de independencia sobre la tabla patron x default.
    #   H0: la tasa de default es la misma en todos los patrones.
    #
    # ADVERTENCIA sobre el p-valor con n grande:
    #   con 150,000 filas, casi cualquier diferencia sale "significativa".
    #   La significacion estadistica dice si la diferencia es REAL; no dice si
    #   es GRANDE. Por eso se acompana de la V de Cramer, que mide el tamano
    #   del efecto y no crece con el tamano de la muestra:
    #       V = sqrt(chi2 / n)      (para una tabla k x 2)
    #       < 0.10 efecto pequeno | 0.10-0.30 medio | > 0.30 grande
    # Solo entran los patrones OBSERVADOS: una fila de ceros hace que la
    # frecuencia esperada sea cero y chi2_contingency falla. Un patron que
    # nunca ocurre no aporta informacion sobre la independencia.
    obs = combinaciones[combinaciones["n"] > 0]
    tabla_cont = np.column_stack([
        obs["n"] - obs["defaults"],   # no eventos
        obs["defaults"],              # eventos
    ])

    chi2, p_valor, gl, esperados = stats.chi2_contingency(tabla_cont)
    v_cramer = np.sqrt(chi2 / tabla_cont.sum())

    magnitud = ("pequeño" if v_cramer < 0.10
                else "medio" if v_cramer < 0.30 else "grande")

    print("\n  " + "-" * 70)
    print("  VALIDACION ESTADISTICA DE LOS PATRONES DE FALTANTES")
    print("  " + "-" * 70)

    print(f"\n    [1] Chi-cuadrado de independencia (patron x default)")
    print(f"        H0: la tasa de default es igual en todos los patrones")
    print(f"        chi2 = {chi2:,.2f}   gl = {gl}   p = {p_valor:.3e}")

    print(f"\n    [2] V de Cramer (tamaño del efecto)")
    print(f"        V = {v_cramer:.4f}   ({magnitud})")
    print(f"        escala: <0.10 pequeño | 0.10-0.30 medio | >0.30 grande")

    # El chi-cuadrado exige frecuencias esperadas >= 5 en (casi) todas las celdas.
    # Si no se cumple, el p-valor no es fiable y habria que usar un test exacto.
    bajas = (esperados < 5).sum()
    if bajas:
        print(f"\n        AVISO: {bajas} celdas con frecuencia esperada < 5.")
        print(f"        La aproximacion chi-cuadrado no es fiable: usar test")
        print(f"        exacto de Fisher o agrupar patrones.")

    # El chi-cuadrado es un contraste OMNIBUS: rechazar H0 solo indica que al
    # menos DOS patrones difieren, no que difieran todos. Y un p-valor rechaza
    # la hipotesis nula; no "demuestra" la alternativa.
    print(f"\n    CONCLUSION")
    if p_valor < 0.05:
        print(f"        Existe evidencia estadistica de que la tasa de default")
        print(f"        difiere entre al menos dos patrones de faltantes")
        print(f"        (p = {p_valor:.3e} < 0.05).")
        if v_cramer < 0.10:
            print(f"        Sin embargo, el tamaño del efecto es {magnitud} (V = {v_cramer:.4f}):")
            print(f"        la diferencia es real pero de magnitud modesta.")
        else:
            print(f"        El tañmano del efecto es {magnitud} (V = {v_cramer:.4f}).")
    else:
        print(f"        No hay evidencia suficiente para rechazar que la tasa de")
        print(f"        default sea igual en todos los patrones (p = {p_valor:.3e}).")
    print(f"\n        Nota: con n = {int(tabla_cont.sum()):,} el p-valor tiende a ser")
    print(f"        minusculo aunque la diferencia sea trivial. La significacion")
    print(f"        dice si el efecto es REAL; la V de Cramer, si es GRANDE.")


# =============================================================================
# PARTE 5 — VALORES SOSPECHOSOS Y OUTLIERS
# =============================================================================
titulo("PARTE 5 — VALORES SOSPECHOSOS Y OUTLIERS")

# --- 5a. Deteccion AUTOMATICA de codigos centinela --------------------------
# Un "codigo centinela" es un numero que el sistema de origen usa para decir
# "no se / dato invalido", en vez de dejar la celda vacia.
#
# NO se buscan valores concretos escritos a mano: en este dataset son 96 y 98,
# pero otro sistema usaria 999, -1 o 9999. Se detectan por su FIRMA:
# un centinela esta SEPARADO del resto de la distribucion por un salto
# desproporcionado, porque no es una continuacion de la escala real.
#
#   valores reales   ->  0, 1, 2, ..., 12, 13
#   salto            ->                      |------------- 83 -------------|
#   centinelas       ->                                                   96, 98
#
# El criterio: se ordenan los valores unicos, se miden los saltos entre
# consecutivos, y si el mayor supera FACTOR veces el salto tipico (la mediana),
# todo lo que hay por encima es candidato a codigo.

cols_mora = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

FACTOR_SALTO = 10        # cuantas veces el salto tipico define "desproporcionado"
TOP_VALORES = 10         # cuantos valores altos se muestran por variable


def detectar_codigos_aislados(serie, factor=FACTOR_SALTO):
    """
    Devuelve los valores separados del resto por un salto desproporcionado.

    Funciona con cualquier codigo (96, 98, 999, -1...) porque no busca
    numeros concretos, sino la discontinuidad que los delata.
    Devuelve lista vacia si la distribucion es continua.
    """
    vals = np.sort(serie.dropna().unique())
    if len(vals) < 3:
        return [], None
    saltos = np.diff(vals)
    salto_tipico = np.median(saltos)
    if salto_tipico <= 0:
        return [], None
    i_max = int(np.argmax(saltos))
    if saltos[i_max] < factor * salto_tipico:
        return [], None                       # sin discontinuidad clara
    return list(vals[i_max + 1:]), float(saltos[i_max])


print(f"5a. Distribucion de las variables de morosidad (top {TOP_VALORES} valores):")
for col in cols_mora:
    vc = train[col].value_counts().sort_index(ascending=False).head(TOP_VALORES)
    print(f"\n  {col}:")
    print(vc.to_string())

print("\n  DETECCION AUTOMATICA DE CODIGOS CENTINELA")
print(f"    Criterio 1 (discontinuidad): salto > {FACTOR_SALTO}x el salto tipico")
print(f"    Criterio 2 (dominio)       : un conteo no puede ser negativo")

codigos_detectados = set()
for col in cols_mora:
    codigos, salto = detectar_codigos_aislados(train[col])

    # El criterio del salto NO detecta un centinela pegado a la escala valida:
    # un -1 junto al 0 no deja hueco, luego es invisible para el metodo anterior.
    # Esos se cazan por violacion de dominio, que es conocimiento del negocio.
    negativos = sorted(train.loc[train[col] < 0, col].dropna().unique())

    if codigos or negativos:
        codigos_detectados.update(codigos)
        codigos_detectados.update(negativos)
        print(f"\n    {col}:")
        if codigos:
            vals = np.sort(train[col].dropna().unique())
            ultimo_real = vals[vals < min(codigos)].max()
            print(f"      ultimo valor de la escala real : {ultimo_real:.0f}")
            print(f"      salto detectado                : {salto:.0f}")
            print(f"      codigos por discontinuidad     : "
                  f"{[f'{c:.0f}' for c in codigos]}")
        if negativos:
            print(f"      codigos por dominio (negativos): "
                  f"{[f'{c:.0f}' for c in negativos]}")
    else:
        print(f"\n    {col}: sin discontinuidad ni valores fuera de dominio")

if codigos_detectados:
    codigos_detectados = sorted(codigos_detectados)
    print(f"\n    Codigos detectados en el conjunto: "
          f"{[f'{c:.0f}' for c in codigos_detectados]}")

    es_centinela = train[cols_mora].isin(codigos_detectados).any(axis=1)
    n_cent = int(es_centinela.sum())
    dr_cent = train.loc[es_centinela, TARGET].mean() * 100
    dr_resto = train.loc[~es_centinela, TARGET].mean() * 100

    print(f"\n    Filas afectadas          : {n_cent:,} "
          f"({n_cent/len(train)*100:.2f}% del total)")
    print(f"    Default rate de esas filas: {dr_cent:.2f}%")
    print(f"    Default rate del resto    : {dr_resto:.2f}%")
    print(f"    Ratio                     : {dr_cent/dr_resto:.1f}x")

    # Un centinela suele aparecer en TODAS las columnas de la misma fila:
    # el sistema marca el registro entero como no disponible, no un campo suelto.
    iguales = train.loc[es_centinela, cols_mora].nunique(axis=1) == 1
    print(f"\n    Filas donde las {len(cols_mora)} columnas tienen el MISMO valor: "
          f"{int(iguales.sum()):,} de {n_cent:,} ({iguales.mean()*100:.0f}%)")
    if iguales.mean() > 0.95:
        print("    -> el sistema marca el REGISTRO completo, no campos sueltos")
else:
    es_centinela = pd.Series(False, index=train.index)
    print("\n    No se detectaron codigos aislados.")

# --- 5b. RevolvingUtilization deberia ser un ratio entre 0 y 1 -------------
print("\n5b. RevolvingUtilizationOfUnsecuredLines (deberia estar entre 0 y 1):")
ru = train["RevolvingUtilizationOfUnsecuredLines"]
UMBRALES_RU = [1, 1.5, 2, 5] + list(range(10, 101, 10)) + [1000]
for umbral in UMBRALES_RU:
    m = ru > umbral
    n = int(m.sum())
    dr = f"{train.loc[m, TARGET].mean()*100:6.2f}%" if n else "     -"
    print(f"    Filas con valor > {umbral:>6}: {n:>6,}   default: {dr}")
print(f"    Valor maximo: {ru.max():,.0f}")
print(f"    Tasa global de referencia: {tasa_default*100:.2f}%")

# --- 5c. DebtRatio: ¿se comporta igual con y sin denominador? --------------
# DebtRatio = deuda_mensual / ingreso_mensual. La seccion 5e demuestra que
# cuando falta el ingreso el campo cambia de SIGNIFICADO (pasa de ratio a
# importe). Aqui se comprueba lo que de verdad importa para el modelo: si
# ademas cambia de COMPORTAMIENTO, es decir, si sigue ordenando el riesgo.
#
# Si ambos grupos significaran lo mismo, bastaria una sola columna. Si no,
# mezclarlos obliga al modelo a aprender una relacion que no existe.
print("\n5c. DebtRatio: ¿se comporta igual con y sin denominador?")

debt = train["DebtRatio"]
ingreso = train["MonthlyIncome"]
sin_denom = ingreso.isna() | (ingreso <= 1)
grupos_dn = [("CON ingreso", ~sin_denom), ("SIN ingreso", sin_denom)]

print(f"    Filas: con denominador {int((~sin_denom).sum()):,}   |   "
      f"sin denominador {int(sin_denom.sum()):,}")

# --- (1) ¿Son la misma distribucion? ----------------------------------------
print("\n    (1) DISTRIBUCION DE DebtRatio")
print(f"\n        {'percentil':<12} {'CON ingreso':>14} {'SIN ingreso':>14}")
print("        " + "-" * 42)
print(f"        {'min':<12} {debt[~sin_denom].min():>14,.2f} "
      f"{debt[sin_denom].min():>14,.2f}")
for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
    print(f"        p{int(q*100):<11} {debt[~sin_denom].quantile(q):>14,.2f} "
          f"{debt[sin_denom].quantile(q):>14,.2f}")
print(f"        {'max':<12} {debt[~sin_denom].max():>14,.0f} "
      f"{debt[sin_denom].max():>14,.0f}")

# El minimo importa por si solo: un importe de deuda de 0 es indistinguible de
# un ratio de 0, asi que en ese extremo las dos poblaciones SI coinciden. Es la
# unica zona donde el cambio de unidad no cambia nada.
n_cero = int((debt[sin_denom] == 0).sum())
print(f"\n        Filas SIN ingreso con DebtRatio exactamente 0: {n_cero:,} "
      f"({n_cero/int(sin_denom.sum())*100:.1f}%)")
print("        Un importe de 0 y un ratio de 0 son el mismo numero: en ese")
print("        extremo la ambiguedad de unidad no tiene consecuencias.")

p99_con = debt[~sin_denom].quantile(0.99)
p10_sin = debt[sin_denom].quantile(0.10)
if p10_sin > p99_con:
    print(f"\n        El p10 del grupo SIN ingreso ({p10_sin:,.2f}) ya supera el p99")
    print(f"        del grupo CON ingreso ({p99_con:,.2f}): las distribuciones")
    print("        practicamente no se solapan. No es la misma magnitud.")

# --- (2) Solapamiento por tramos --------------------------------------------
# Se muestran conteo Y porcentaje: el porcentaje permite comparar los dos
# grupos entre si (tienen tamanos muy distintos), y el conteo dice si la cifra
# se apoya en poblacion suficiente. Un 0.01% sobre 94,505 son 10 filas.
print("\n    (2) ¿CUANTAS FILAS DE CADA GRUPO SUPERAN CADA CORTE?")
n_con, n_sin = int((~sin_denom).sum()), int(sin_denom.sum())
print(f"\n        {'corte':<10} {'CON ingreso':>22} {'SIN ingreso':>22}")
print(f"        {'':<10} {f'(de {n_con:,})':>22} {f'(de {n_sin:,})':>22}")
print("        " + "-" * 56)
for lim in [1, 2, 10, 100]:
    m_con = debt[~sin_denom] > lim
    m_sin = debt[sin_denom] > lim
    print(f"        > {lim:<8} {int(m_con.sum()):>13,} ({m_con.mean()*100:>5.2f}%) "
          f"{int(m_sin.sum()):>13,} ({m_sin.mean()*100:>5.2f}%)")

# --- (3) La prueba decisiva: ¿ordena el riesgo dentro de cada grupo? --------
# Que las escalas difieran no bastaria para justificar separarlas: bastaria
# reescalar. Lo que decide es si la variable ORDENA el riesgo. Se mide con la
# correlacion de Spearman entre el indice de decil y la tasa de evento, que
# solo mira el orden y por tanto es inmune al cambio de unidad.
print("\n    (3) GRADIENTE DE RIESGO POR DECIL, DENTRO DE CADA GRUPO")
print("        (si significaran lo mismo, ambos ordenarian el riesgo igual)")

for nombre, m in grupos_dn:
    sub = train.loc[m, ["DebtRatio", TARGET]].copy()
    sub["decil"] = pd.qcut(sub["DebtRatio"], 10, duplicates="drop", labels=False)
    g = sub.groupby("decil", observed=True).agg(
        hasta=("DebtRatio", "max"), n=(TARGET, "size"), tasa=(TARGET, "mean"))
    g["tasa"] *= 100

    rho, p_val = stats.spearmanr(g.index, g["tasa"])
    print(f"\n        --- {nombre} (n={int(m.sum()):,}) ---")
    print(f"        {'decil':>6} {'hasta':>13} {'n':>8} {'default':>9}")
    for i in g.index:
        print(f"        {int(i)+1:>6} {g['hasta'][i]:>13,.2f} {int(g['n'][i]):>8,} "
              f"{g['tasa'][i]:>8.2f}%")
    print(f"        Spearman(decil, tasa) = {rho:+.3f}   (p = {p_val:.4f})")
    print(f"        Rango de tasas: {g['tasa'].min():.2f}% a {g['tasa'].max():.2f}%")

# --- (4) Veredicto ----------------------------------------------------------
print("\n    (4) VEREDICTO")
print("        Con denominador, DebtRatio ordena el riesgo de forma creciente")
print("        en su mitad superior: es apalancamiento y se comporta como tal.")
print("        Sin denominador, el importe de deuda NO ordena nada: la tasa")
print("        rebota sin direccion y el decil mas alto vuelve a la media.")
print("        Son dos variables distintas compartiendo columna. Por eso el")
print("        pipeline las separa en DebtRatio_limpio y deuda_bruta_proxy (ver 5e).")

# --- 5d. Edad ---------------------------------------------------------------
print("\n5d. age:")
edad = train["age"]
print(f"    Min: {edad.min()}   |   Max: {edad.max()}")

# CUIDADO CON LOS NaN EN LAS COMPARACIONES:
#   NaN == 0  -> False        NaN < 18  -> False        NaN > 100 -> False
#   Cualquier comparacion con NaN devuelve False, luego un filtro como
#   (edad < 18) NO cuenta los faltantes: pasarian invisibles.
#   Solo .isna() los detecta. Por eso se listan como categoria aparte.
#
# Las categorias son EXCLUYENTES y suman el total: si no sumaran, habria
# un caso sin clasificar (que es justo como se cuelan los errores).
EDAD_MIN_VALIDA = 18
EDAD_MAX_PLAUSIBLE = 100

cat_nan = edad.isna()
cat_neg = edad < 0                       # imposible por dominio
cat_cero = edad == 0                     # tipico marcador de "no informado"
cat_menor = (edad > 0) & (edad < EDAD_MIN_VALIDA)
cat_valida = (edad >= EDAD_MIN_VALIDA) & (edad <= EDAD_MAX_PLAUSIBLE)
cat_extrema = edad > EDAD_MAX_PLAUSIBLE  # posible pero muy rara

categorias = [
    ("NaN (faltante)",                 cat_nan),
    ("negativa (imposible)",           cat_neg),
    ("= 0 (marcador?)",                cat_cero),
    (f"1 a {EDAD_MIN_VALIDA-1} (menor de edad)", cat_menor),
    (f"{EDAD_MIN_VALIDA} a {EDAD_MAX_PLAUSIBLE} (valida)", cat_valida),
    (f"> {EDAD_MAX_PLAUSIBLE} (extrema)", cat_extrema),
]

print(f"\n    {'categoria':<28} {'n':>8} {'%':>7} {'default':>9}")
print("    " + "-" * 55)
suma = 0
for nombre, mask in categorias:
    n = int(mask.sum())
    suma += n
    dr = f"{train.loc[mask, TARGET].mean()*100:8.2f}%" if n else "       -"
    print(f"    {nombre:<28} {n:>8,} {n/len(train)*100:>6.2f}% {dr}")

print("    " + "-" * 55)
print(f"    {'TOTAL':<28} {suma:>8,}   "
      f"{'OK: suman el total' if suma == len(train) else 'ERROR: no cuadran'}")

sospechosas = int((cat_neg | cat_cero | cat_menor).sum())
print(f"\n    Edades no plausibles (negativa, 0 o menor de edad): {sospechosas:,}")
print(f"    El pipeline las convierte a NaN con el filtro  age < {EDAD_MIN_VALIDA},")
print(f"    que cubre las tres categorias a la vez (incluidas las negativas).")


# --- 5e. MonthlyIncome: ¿el 0 y el 1 son valores o son codigos? -------------
# El pipeline manda MonthlyIncome <= 1 a NaN por considerarlo "no reportado".
# Esa decision NO puede darse por buena sin evidencia: un ingreso bajo y un
# ingreso ausente son cosas distintas, y confundirlos invertiria la senal
# (el modelo colocaria a un "no informado" en el extremo de mayor pobreza,
# donde espera riesgo ALTO, cuando su riesgo real es BAJO).
print("\n5e. MonthlyIncome: ¿el 0 y el 1 son valores o son codigos?")
mi = train["MonthlyIncome"]
dr_col = train["DebtRatio"]

# --- (1) ¿Banda continua o valores discretos? -------------------------------
# Si existieran ingresos de 0.35 o 0.80 estariamos ante una escala continua y
# hablar de "codigo" no tendria sentido. Si solo hay enteros aislados, si.
print("\n    (1) VALORES EXACTOS EN EL EXTREMO INFERIOR (<= 10)")
bajos = mi[mi <= 10].value_counts().sort_index()
for v, n_v in bajos.items():
    print(f"          {v:>6,.1f} -> {n_v:>6,}")
n_frac = int(((mi > 0) & (mi < 1)).sum())
print(f"        valores estrictamente entre 0 y 1: {n_frac}")
print("        -> " + ("no hay banda continua: 0 y 1 son valores DISCRETOS"
                       if n_frac == 0 else
                       "hay valores fraccionarios: la escala es continua"))

# --- (2) Prueba del denominador ---------------------------------------------
# DebtRatio = deuda_mensual / ingreso_mensual.
# Si el sistema de origen no tenia ingreso, no pudo dividir, y lo esperable es
# que guardara el NUMERADOR en bruto. Contraste: comparar el DebtRatio de los
# sospechosos contra la deuda mensual REAL de las filas con ingreso normal,
# reconstruida como DebtRatio x MonthlyIncome.
#   - Si los sospechosos tienen DebtRatio del orden de 0-2, son ratios de verdad
#     y el pipeline los estaria destruyendo sin motivo.
#   - Si lo tienen del orden de cientos o miles de dolares, es un importe.
print("\n    (2) PRUEBA DEL DENOMINADOR")
print("        DebtRatio = deuda / ingreso. Sin ingreso no hay division, luego")
print("        el campo deberia contener el importe de deuda en bruto.")
print("        Se compara contra la deuda real (DebtRatio x MonthlyIncome) de")
print("        las filas con ingreso normal.\n")

normal = train[mi > 100]
deuda_real = normal["DebtRatio"] * normal["MonthlyIncome"]

grupos_dr = [
    ("MonthlyIncome == 0",   dr_col[mi == 0]),
    ("MonthlyIncome == 1",   dr_col[mi == 1]),
    ("MonthlyIncome FALTA",  dr_col[mi.isna()]),
    ("deuda REAL (MI>100)",  deuda_real),
]
print(f"        {'grupo':<22} {'p25':>10} {'p50':>10} {'p75':>10} {'%>2':>7}")
print("        " + "-" * 62)
for nombre, serie in grupos_dr:
    q = serie.quantile([0.25, 0.50, 0.75])
    print(f"        {nombre:<22} {q.iloc[0]:>10,.0f} {q.iloc[1]:>10,.0f} "
          f"{q.iloc[2]:>10,.0f} {(serie > 2).mean()*100:>6.1f}%")
print(f"        {'ratio de verdad':<22} {'~0.1':>10} {'~0.3':>10} {'~0.5':>10} "
      f"{normal['DebtRatio'].gt(2).mean()*100:>6.1f}%")

# --- (3) Riesgo por grupo, con intervalo de confianza -----------------------
# Las tasas puntuales no bastan: 498 filas dan un intervalo ancho. Se necesita
# saber si los grupos son DISTINGUIBLES entre si, no solo si sus medias diferen.
print("\n    (3) TASA DE DEFAULT POR GRUPO (IC 95% de Wilson)")
grupos_riesgo = [
    ("MonthlyIncome == 0",     mi == 0),
    ("MonthlyIncome == 1",     mi == 1),
    ("MonthlyIncome (1,100]",  (mi > 1) & (mi <= 100)),
    ("MonthlyIncome FALTA",    mi.isna()),
    ("MonthlyIncome > 100",    mi > 100),
]
print(f"\n        {'grupo':<24} {'n':>7} {'ev':>6} {'tasa':>8}   {'IC 95%':>16}")
print("        " + "-" * 66)
for nombre, m in grupos_riesgo:
    n_g = int(m.sum())
    if n_g == 0:
        continue
    ev = int(train.loc[m, TARGET].sum())
    lo, hi = ic_wilson(ev / n_g, n_g)
    print(f"        {nombre:<24} {n_g:>7,} {ev:>6,} {ev/n_g*100:>7.2f}%   "
          f"[{lo*100:>5.2f}, {hi*100:>5.2f}]")

# --- (4) ¿Son la misma poblacion que los faltantes? -------------------------
# Importa para decidir si basta UN indicador o hacen falta dos. Si el perfil
# demografico coincide, un solo flag sirve; si difiere, colapsarlos pierde
# informacion.
print("\n    (4) PERFIL: ¿es la misma poblacion que los faltantes?")
print(f"\n        {'grupo':<22} {'edad':>6} {'lineas':>8} {'RevUtil':>9} {'dep.NaN':>9}")
print("        " + "-" * 56)
for nombre, m in [("MonthlyIncome == 0", mi == 0),
                  ("MonthlyIncome == 1", mi == 1),
                  ("MonthlyIncome FALTA", mi.isna()),
                  ("MonthlyIncome > 100", mi > 100)]:
    s = train.loc[m]
    print(f"        {nombre:<22} {s['age'].median():>6.0f} "
          f"{s['NumberOfOpenCreditLinesAndLoans'].median():>8.0f} "
          f"{s['RevolvingUtilizationOfUnsecuredLines'].median():>9.3f} "
          f"{s['NumberOfDependents'].isna().mean()*100:>8.1f}%")

# --- (5) Zona gris ----------------------------------------------------------
# El umbral del pipeline es <= 1. Las filas entre 1 y 100 quedan FUERA y se
# tratan como ingreso legitimo. Conviene dejar constancia de cuantas son y de
# si su comportamiento las delata, aunque su n sea demasiado pequeno para
# justificar un cambio de umbral.
gris = (mi > 1) & (mi <= 100)
if gris.any():
    print(f"\n    (5) ZONA GRIS: {int(gris.sum())} filas con 1 < MonthlyIncome <= 100")
    print(f"        DebtRatio mediano = {dr_col[gris].median():,.1f}  |  "
          f"con DebtRatio > 2: {dr_col[gris].gt(2).mean()*100:.1f}%")
    print("        Se comportan como los codificados, pero son muy pocas para")
    print("        mover el umbral con evidencia. Quedan como ingreso valido.")


# =============================================================================
# PARTE 6 — RELACION DE CADA VARIABLE CON EL TARGET
# =============================================================================
titulo("PARTE 6 — RELACION DE CADA VARIABLE CON EL TARGET")

# Esta es LA vista mas importante del credit scoring.
# Idea: partimos cada variable en 10 grupos de igual tamano (deciles) y
# medimos la tasa de default en cada grupo. Si la tasa sube o baja de forma
# consistente al movernos por los deciles, la variable tiene poder predictivo.
#
# CUANDO LOS DECILES NO SIRVEN:
#   qcut usa los percentiles como cortes. Si una variable tiene demasiados
#   valores repetidos, esos percentiles coinciden y los bins se funden.
#   Ejemplo: NumberOfTimes90DaysLate tiene 94.4% de ceros, luego P10 a P90
#   valen todos 0 y queda UN SOLO bin con las 150,000 filas y la tasa global:
#   una tabla sin ninguna informacion.
#
#   La variable NO es inutil — el salto de 0 a 1 multiplica el riesgo por 7 —
#   lo que falla es el metodo. En esos casos se agrupa por VALOR EXACTO,
#   que es lo que la variable admite.

# Las tres variables de morosidad se analizan JUNTAS y en orden de gravedad
# (30-59 -> 60-89 -> 90+). Verlas consecutivas permite comparar el salto 0->1
# entre severidades, que es lo que revela si aportan informacion distinta.
# features = TODAS las predictoras. Lo usan tambien las graficas de la PARTE 8,
# asi que no debe filtrarse aqui: el subconjunto de 6a va en su propia variable.
features = [c for c in train.columns if c != TARGET]
features_6a = [c for c in features if c not in cols_mora]
MIN_BINS = 3          # por debajo, los deciles no aportan nada


def tabla_relacion(col):
    """Tabla de tasa de default por grupo, eligiendo el metodo adecuado."""
    try:
        deciles = pd.qcut(train[col], q=10, duplicates="drop")
        n_bins = deciles.cat.categories.size
    except ValueError:
        n_bins = 0

    if n_bins >= MIN_BINS:
        agrupador = deciles
        etiqueta = f"{n_bins} deciles"
    else:
        # Fallback: agrupar por valor exacto, capando la cola para no imprimir
        # decenas de filas con una observacion cada una.
        pct_ceros = (train[col] == 0).mean() * 100
        tope = train[col].quantile(0.999)
        agrupador = train[col].clip(upper=tope).rename(None)
        etiqueta = (f"por valor exacto (deciles inservibles: "
                    f"{pct_ceros:.1f}% de ceros -> {n_bins} bin)")

    tabla = train.groupby(agrupador, observed=True).agg(
        n=(TARGET, "size"),
        default_rate=(TARGET, "mean"),
    )
    tabla["default_rate"] = (tabla["default_rate"] * 100).round(2)
    tabla["vs_global_pp"] = (tabla["default_rate"] - tasa_default * 100).round(2)
    return tabla, etiqueta


print("6a. VARIABLES CONTINUAS Y DE ESTRUCTURA")

for col in features_6a:
    tabla, etiqueta = tabla_relacion(col)
    print(f"\n--- {col}  [{etiqueta}] ---")
    print(tabla.to_string())


# --- 6b. Las tres variables de morosidad, juntas y por gravedad -------------
print("\n" + "-" * 79)
print("6b. HISTORIAL DE MOROSIDAD (las tres, en orden de gravedad)")
print("-" * 79)

for col in cols_mora:
    tabla, etiqueta = tabla_relacion(col)
    print(f"\n--- {col}  [{etiqueta}] ---")
    print(tabla.to_string())

# El salto de 0 a 1 es lo que distingue una gravedad de otra: si fuera el
# mismo en las tres, las variables serian redundantes y bastaria con una.
#
# Se dan DOS lecturas, porque responden a preguntas distintas:
#   EXACTAMENTE 1  -> ¿cuanto pesa un unico episodio?
#   AL MENOS 1     -> ¿cuanto pesa tener historial de morosidad?
# La segunda incluye a los reincidentes, luego siempre sale mas alta.
#
# Los codigos centinela se EXCLUYEN de "al menos 1": su historial es
# desconocido, no positivo. Incluirlos mezclaria "tiene atrasos" con
# "no sabemos si los tiene".
_cent = ", ".join(f"{c:.0f}" for c in codigos_detectados) if codigos_detectados else "ninguno"
print("\n  COMPARACION DEL SALTO ENTRE GRAVEDADES")
print(f"    (centinelas excluidos: {_cent} — su historial es desconocido,")
print("     no positivo, luego no cuentan como 'tiene atrasos')")
print(f"\n    {'variable':<38} {'= 0':>8} {'= 1':>8} {'factor':>7} "
      f"{'>= 1':>8} {'factor':>7} {'n(>=1)':>8}")
print("    " + "-" * 92)

for col in cols_mora:
    s = train[col]
    valido = ~s.isin(codigos_detectados) if codigos_detectados else pd.Series(True, index=s.index)

    m0 = valido & (s == 0)
    m1 = valido & (s == 1)
    mge1 = valido & (s >= 1)

    t0 = train.loc[m0, TARGET].mean() * 100
    t1 = train.loc[m1, TARGET].mean() * 100
    tge1 = train.loc[mge1, TARGET].mean() * 100

    print(f"    {col:<38} {t0:>7.2f}% {t1:>7.2f}% {t1/t0:>6.1f}x "
          f"{tge1:>7.2f}% {tge1/t0:>6.1f}x {int(mge1.sum()):>8,}")

print("\n    '= 1' aisla el efecto de UN episodio.")
print("    '>= 1' mide el efecto de tener historial, e incluye reincidentes.")

# =============================================================================
# PARTE 7 — CORRELACIONES
# =============================================================================
titulo("PARTE 7 — CORRELACIONES")

# QUE SE ESTA CALCULANDO EXACTAMENTE:
#   - Coeficiente de PEARSON (el que usa .corr() por defecto).
#   - Sobre los DATOS CRUDOS: los centinelas 96/98 siguen dentro.
#   - Pearson mide asociacion LINEAL: dos variables con relacion fuerte pero
#     curva pueden dar correlacion ~0.
#   - Es extremadamente sensible a valores extremos, porque los momentos de
#     segundo orden crecen con el CUADRADO de la desviacion.
#
#   +1 = cuando una sube, la otra sube siempre
#    0 = no hay relacion lineal
#   -1 = cuando una sube, la otra baja
corr = train.corr(numeric_only=True)

print("Metodo: Pearson sobre datos CRUDOS (centinelas incluidos)\n")

print("Correlacion de cada variable con el TARGET (ordenada por magnitud):")
corr_target = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print(corr_target.to_string())

print("\n  AVISO: estos valores con el target son pequenos y pueden enganar.")
print("  RevolvingUtilization da -0.002 (parece inutil) y sin embargo por")
print("  deciles va de 1.4% a 23.2% de default: es la variable mas predictiva.")
print("  Pearson no ve relaciones no lineales. La vista fiable es la PARTE 6.")

print("\nPares de variables MUY correlacionadas entre si (|corr| > 0.5):")
print("(Importante: variables casi identicas confunden a la regresion logistica)")

# CONTRASTE CON Y SIN CENTINELAS:
#   Un punado de filas con un valor extremo identico en varias columnas puede
#   inflar la correlacion hasta ~1 aunque las variables apenas se relacionen.
#   Recalcular sin esas filas revela si la correlacion es real o un artefacto.
sin_cent = train.loc[~es_centinela] if es_centinela.any() else train
corr_limpia = sin_cent.corr(numeric_only=True)

print(f"\n    {'par de variables':<62} {'crudo':>8} {'sin cent.':>10} {'cambio absoluto':>8}")
print("    " + "-" * 92)
encontrados = False
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        a, b = corr.columns[i], corr.columns[j]
        if TARGET in (a, b):
            continue
        c_crudo = corr.iloc[i, j]
        if abs(c_crudo) <= 0.5:
            continue
        c_limpio = corr_limpia.loc[a, b]
        delta = c_limpio - c_crudo
        par = f"{a[:29]} <-> {b[:29]}"
        marca = "  <- ARTEFACTO" if abs(delta) > 0.3 else ""
        print(f"    {par:<62} {c_crudo:>8.3f} {c_limpio:>10.3f} {delta:>+8.3f}{marca}")
        encontrados = True

if not encontrados:
    print("    Ninguno.")
else:
    print(f"\n    Filas excluidas en 'sin cent.': {int(es_centinela.sum()):,} "
          f"({es_centinela.mean()*100:.2f}% de la muestra)")
    print("    Un cambio grande significa que la correlacion la producian esas")
    print("    pocas filas, no una relacion real entre las variables.")
    print("    Conclusion: NO eliminar variables basandose en la correlacion cruda.")


# =============================================================================
# PARTE 8 — GRAFICAS
# =============================================================================
titulo("PARTE 8 — GRAFICAS (se guardan en ./outputs/)")

# --- Grafica 1: balance de clases -----------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
resumen_target["conteo"].plot(kind="bar", ax=ax, color=["#2a9d8f", "#e76f51"])
ax.set_title("Balance de clases del target", pad=14)
ax.set_ylabel("Numero de personas")
ax.set_xlabel("")
plt.xticks(rotation=0)

# La etiqueta de la barra mas alta se dibuja ENCIMA de ella y choca con el
# titulo si el eje termina justo en el maximo. Se anade un 18% de margen,
# calculado sobre los datos para que siga funcionando si estos cambian.
ax.set_ylim(0, resumen_target["conteo"].max() * 1.18)

for i, v in enumerate(resumen_target["conteo"]):
    ax.text(i, v, f"{v:,}\n({resumen_target['porcentaje'].iloc[i]}%)",
            ha="center", va="bottom")
plt.tight_layout()
plt.savefig(OUTPUTS / "01_balance_target.png", dpi=110)
plt.close()

# --- Grafica 2: histogramas SIN RECORTAR ----------------------------------
# PRINCIPIO: el EDA debe ser lo mas CRUDO posible. Un histograma recortado
# al percentil 99 se ve bonito, pero ESCONDE justo lo que el EDA debe
# encontrar. Si la grafica se ve fea porque un outlier aplasta todo el eje,
# esa fealdad ES el hallazgo, no un defecto de presentacion.
#
# Eje Y en escala log: asi una barra con frecuencia 1 sigue siendo visible.
# Sin esto, las colas desapareceran aunque el eje X las incluya.
fig, axes = plt.subplots(4, 3, figsize=(15, 14))
for ax, col in zip(axes.flatten(), features):
    datos = train[col].dropna()

    # Variables ENTERAS de rango corto: los bins automaticos caen entre
    # numeros y el eje se lee como si fuera continuo ("2.5 dependientes").
    # Con bins centrados en cada entero, una barra = un valor exacto.
    es_entera = np.allclose(datos, datos.round())
    rango = datos.max() - datos.min()

    if es_entera and rango <= 25:
        lo, hi = int(datos.min()), int(datos.max())
        bins = np.arange(lo - 0.5, hi + 1.5, 1)     # cada barra centrada en su entero
        paso = 1 if rango <= 15 else 2              # evita amontonar etiquetas
        ax.set_xticks(np.arange(lo, hi + 1, paso))
    else:
        bins = 60
        # Con rangos de millones matplotlib usa notacion de desplazamiento
        # ("1e6" en una esquina) y quedan solo 6 marcas: imposible situar un
        # valor. Se formatean como 500K / 1.5M y se fuerzan pasos "redondos".
        #
        # steps=[1, 2, 2.5, 5, 10] restringe el paso a esos multiplos de una
        # potencia de 10, es decir a mitades y cuartos: 0.25, 0.5, 1, 2, 2.5,
        # 5, 10, 25, 50... Sin ello, MaxNLocator elige pasos como 400K o 6K,
        # que obligan a hacer cuentas para situar un punto en el eje.
        if datos.max() >= 10_000:
            ax.xaxis.set_major_locator(
                mticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_formato_eje))
            ax.tick_params(axis="x", rotation=45)

    ax.hist(datos, bins=bins, color="#264653", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ratio = datos.max() / max(abs(datos.median()), 1e-9)
    ax.set_title(f"{col}\nmax={datos.max():,.0f}  mediana={datos.median():,.2f}  "
                 f"(max/mediana = {ratio:,.0f}x)", fontsize=8)
    ax.tick_params(labelsize=8)
for ax in axes.flatten()[len(features):]:
    ax.set_visible(False)
fig.suptitle("Distribucion CRUDA, sin recortar — si se ve aplastada, hay outliers",
             fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUTS / "02_histogramas.png", dpi=110)
plt.close()

# --- Grafica 3: tasa de default por decil de cada variable ----------------
# Solo se grafican las variables que ADMITEN deciles. Con muchos valores
# repetidos qcut colapsa a 1 o 2 grupos, y el panel resultante es un punto
# suelto (que no es mas que la tasa global) o una recta entre dos valores:
# ninguno de los dos dice nada y ambos parecen un error del grafico.
# Esas variables se analizan por valor exacto en la PARTE 6b.
graficables, omitidas = [], []
for col in features:
    try:
        n_g = pd.qcut(train[col], q=10, duplicates="drop").cat.categories.size
    except ValueError:
        n_g = 0
    (graficables if n_g >= 3 else omitidas).append((col, n_g))

n_filas = int(np.ceil(len(graficables) / 3))
fig, axes = plt.subplots(n_filas, 3, figsize=(15, 3.5 * n_filas))
for ax, (col, _n) in zip(axes.flatten(), graficables):
    deciles = pd.qcut(train[col], q=10, duplicates="drop", labels=False)
    tasa = train.groupby(deciles, observed=True)[TARGET].mean() * 100
    ax.plot(tasa.index, tasa.values, marker="o", color="#e76f51")
    ax.axhline(tasa_default * 100, color="gray", linestyle="--", linewidth=1,
               label="tasa global")
    ax.set_title(f"{col}  ({len(tasa)} grupos)", fontsize=9)
    ax.set_xlabel("decil (1 = valores mas bajos)", fontsize=7)
    ax.set_ylabel("% default", fontsize=7)

    # El eje x es un indice de decil, no una magnitud continua: se marcan
    # TODOS los grupos existentes. Con 10 caben de sobra, y asi se puede
    # localizar cualquier punto sin interpolar entre marcas automaticas.
    # Se usa tasa.index en vez de range(10) porque en las variables con
    # muchos ceros qcut colapsa y quedan menos grupos.
    #
    # qcut numera desde 0, pero en un informe los deciles se citan del 1 al
    # 10 ("el decil 10 concentra el mayor riesgo"). Se cambian solo las
    # ETIQUETAS: los datos y las posiciones siguen siendo 0..n-1.
    # En las variables con NaN, qcut devuelve un indice de tipo float y las
    # etiquetas saldrian como "1.0, 2.0, 3.0". Se convierten a entero.
    ax.set_xticks(tasa.index)
    ax.set_xticklabels([f"{int(i) + 1}" for i in tasa.index])
    ax.tick_params(labelsize=8)

    # Etiqueta numerica sobre cada punto: en un informe, leer "23.1%" es mas
    # util que estimar la altura contra el eje. offset en puntos (no en
    # unidades de datos) para que la separacion sea la misma en todos los
    # paneles, que tienen escalas de y muy distintas.
    for x, y in zip(tasa.index, tasa.values):
        ax.annotate(f"{y:.1f}", xy=(x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.5, color="#333333")

    # Las etiquetas se dibujan por encima del punto mas alto y se saldrian del
    # marco: se amplia el limite superior un 8% del rango para darles sitio.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.08 * (y1 - y0))

for ax in axes.flatten()[len(graficables):]:
    ax.set_visible(False)

titulo_g3 = ("Tasa de default por decil — ¿la variable separa buenos de malos?"
             f"\nLinea gris discontinua = tasa global de default "
             f"({tasa_default*100:.2f}%): por encima, grupo mas riesgoso que la media")
if omitidas:
    nombres = ", ".join(c for c, _ in omitidas)
    titulo_g3 += (f"\nOmitidas por no admitir deciles (ver PARTE 6b): {nombres}")
fig.suptitle(titulo_g3, fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUTS / "03_default_rate_por_decil.png", dpi=110)
plt.close()

# --- Grafica 4: mapa de calor de correlaciones ----------------------------
fig, ax = plt.subplots(figsize=(11, 9))
# vmin/vmax fijos en el rango teorico de Pearson [-1, 1]. Sin ellos seaborn
# escala a los datos observados y la barra arranca en el minimo real (~-0.2),
# lo que sugiere que -0.2 es "el extremo negativo" cuando en realidad es una
# correlacion debil. Con el rango completo, la palidez de los colores comunica
# por si sola que aqui casi no hay dependencia lineal fuerte.
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1,
            square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.7, "ticks": np.arange(-1, 1.01, 0.25)},
            annot_kws={"size": 7}, ax=ax)
ax.set_title("Matriz de correlaciones")
plt.tight_layout()
plt.savefig(OUTPUTS / "04_correlaciones.png", dpi=110)
plt.close()

# --- Grafica 5: COLAS COMPLETAS en escala logaritmica ---------------------
# POR QUE ESTA GRAFICA:
#   La grafica 2 recorta al percentil 99 para poder ver la forma de la
#   distribucion. Pero eso ESCONDE los outliers, que en un EDA son
#   justamente el hallazgo, no el estorbo.
#   Aqui mostramos el rango COMPLETO en escala logaritmica, superponiendo
#   la tasa de default para ver donde la senal se convierte en ruido.

COLA_LARGA = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
]
# Umbrales que decidimos en la limpieza, para dibujarlos como referencia
UMBRALES = {
    "RevolvingUtilizationOfUnsecuredLines": 100,
    "DebtRatio": 100,
}

fig, axes = plt.subplots(len(COLA_LARGA), 2, figsize=(15, 4.2 * len(COLA_LARGA)))

for fila, col in enumerate(COLA_LARGA):
    datos = train[col].dropna()
    n_ceros = int((datos == 0).sum())
    positivos = datos[datos > 0]

    # --- Panel izquierdo: histograma en escala log ------------------------
    ax = axes[fila, 0]
    # log10 no admite ceros, asi que los reportamos aparte en el titulo
    bins_log = np.logspace(np.log10(positivos.min()), np.log10(positivos.max()), 70)
    ax.hist(positivos, bins=bins_log, color="#264653", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")   # el eje Y tambien: si no, la cola es invisible
    ax.set_title(f"{col}\nrango COMPLETO (log-log)  |  {n_ceros:,} ceros excluidos",
                 fontsize=10)
    ax.set_xlabel("valor (escala log)", fontsize=8)
    ax.set_ylabel("frecuencia (escala log)", fontsize=8)
    if col in UMBRALES:
        ax.axvline(UMBRALES[col], color="#e76f51", linestyle="--", linewidth=2,
                   label=f"umbral elegido = {UMBRALES[col]}")
        ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # --- Panel derecho: tasa de default por orden de magnitud -------------
    # Agrupamos en potencias de 10 para ver donde el riesgo cambia de regimen
    ax = axes[fila, 1]
    cortes = [-0.001, 0]
    e = np.floor(np.log10(positivos.min()))
    while 10 ** e <= positivos.max():
        cortes.append(10 ** e)
        e += 1
    cortes.append(positivos.max() * 1.01)
    cortes = sorted(set(cortes))

    grupos = pd.cut(datos, bins=cortes)
    tabla = train.loc[datos.index].groupby(grupos, observed=True).agg(
        n=(TARGET, "size"), dr=(TARGET, "mean"))
    tabla = tabla[tabla["n"] >= 3]          # ocultamos celdas sin masa suficiente

    x = np.arange(len(tabla))
    ax.bar(x, tabla["dr"] * 100, color="#e76f51", alpha=0.85)
    ax.axhline(tasa_default * 100, color="gray", linestyle="--", linewidth=1.5,
               label=f"media global {tasa_default*100:.2f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in tabla.index], rotation=45,
                       ha="right", fontsize=7)
    ax.set_title(f"{col}\ntasa de default por orden de magnitud", fontsize=10)
    ax.set_ylabel("% default", fontsize=8)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    # Anotamos el n de cada barra: sin eso, una barra alta con n=4 enganna
    for xi, (_, r) in zip(x, tabla.iterrows()):
        ax.text(xi, r["dr"] * 100, f"n={int(r['n']):,}", ha="center",
                va="bottom", fontsize=6, rotation=90)

fig.suptitle("Colas completas: donde la senal se vuelve ruido", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUTS / "05_colas_completas.png", dpi=110)
plt.close()

print("Graficas guardadas:")
for f in sorted(OUTPUTS.glob("*.png")):
    print(f"    {f.name}")

print("\n" + "=" * 79)
print("EDA COMPLETADO")
print("=" * 79)

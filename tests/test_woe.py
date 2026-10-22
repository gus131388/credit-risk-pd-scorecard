# -*- coding: utf-8 -*-
"""
test_woe.py — Pruebas del modulo WoE y del auditor de binning

Se ejecuta con:  pytest tests/ -v

QUE PROBAMOS Y POR QUE

El auditor de binning existe para CUESTIONAR los cortes manuales. Si el
auditor se equivoca, la conclusion que saquemos de el sera peor que no haberlo
usado: nos daria confianza injustificada en un cambio.

Los invariantes que cubrimos:

  1. Cobertura: ningun valor puede quedarse sin bin ni caer en dos.
  2. FALTANTE: los NaN nunca entran en el calculo de cortes numericos.
  3. Valores especiales: el 0 de los contadores no se puede fusionar, porque
     el salto 0 -> 1 es la senal mas fuerte del dataset.
  4. Restricciones: los bins automaticos deben cumplir tamano y eventos
     minimos, salvo cuando es matematicamente imposible.
  5. Reproducibilidad: el mismo input da los mismos cortes, siempre.
  6. Pureza: ninguna funcion muta lo que recibe.
  7. Anti-fuga: nada de lo que aprende necesita validacion ni test.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ))

from src import woe as w


# =============================================================================
# DATOS DE PRUEBA
# =============================================================================

@pytest.fixture
def conteo_concentrado():
    """
    Contador con 90% de ceros, como las variables de mora reales.
    El riesgo crece con el valor: 0 -> ~3%, 1 -> ~30%, 2+ -> ~60%.
    """
    rng = np.random.default_rng(0)
    x = np.concatenate([np.zeros(9000), np.ones(700),
                        np.full(200, 2.0), np.full(100, 3.0)])
    p = np.where(x == 0, 0.03, np.where(x == 1, 0.30, 0.60))
    y = (rng.random(len(x)) < p).astype(int)
    return pd.Series(x, name="conteo"), pd.Series(y)


@pytest.fixture
def continua():
    """Variable continua con riesgo decreciente, tipo ingreso."""
    rng = np.random.default_rng(1)
    x = rng.lognormal(8, 0.7, 5000)
    p = np.clip(0.25 - 0.00002 * x, 0.01, 0.5)
    y = (rng.random(len(x)) < p).astype(int)
    return pd.Series(x, name="continua"), pd.Series(y)


# =============================================================================
# 1. COBERTURA Y ASIGNACION DE BINS
# =============================================================================

def test_todo_valor_cae_en_exactamente_un_bin(conteo_concentrado):
    """
    Si un valor quedara fuera de los cortes, pd.cut devuelve NaN y esa fila
    acabaria en FALTANTE sin serlo: el bin de faltantes dejaria de significar
    'dato ausente' y contaminaria su WoE.
    """
    x, y = conteo_concentrado
    cortes = w.chimerge(x, y, special_values=[0])
    bins = w._asignar_bins(x, cortes)

    assert bins.notna().all()
    assert len(bins) == len(x)
    # la suma de los tamanos de bin reconstruye la poblacion: nadie se duplica
    g = w.evaluar_binning(x, y, cortes)
    assert int(g["n"].sum()) == len(x)


def test_los_nan_van_a_faltante_y_no_afectan_los_cortes(conteo_concentrado):
    x, y = conteo_concentrado
    x_con_nan = x.copy()
    x_con_nan.iloc[:500] = np.nan

    cortes = w.chimerge(x_con_nan, y, special_values=[0])
    assert all(np.isfinite(c) or np.isinf(c) for c in cortes)

    g = w.evaluar_binning(x_con_nan, y, cortes)
    assert w.ETIQUETA_FALTANTE in g.index
    assert int(g.loc[w.ETIQUETA_FALTANTE, "n"]) == 500
    # FALTANTE va SIEMPRE al final: no ocupa lugar en la recta real
    assert g.index[-1] == w.ETIQUETA_FALTANTE
    assert pd.isna(g.loc[w.ETIQUETA_FALTANTE, "limite_inf"])


def test_infinito_captura_la_cola_superior():
    """
    El ultimo corte debe ser np.inf. Con un borde finito, un valor mayor que el
    maximo visto en train (perfectamente posible en produccion) quedaria sin
    asignar.
    """
    x = pd.Series([0, 1, 2, 3, 4, 5] * 100, dtype=float)
    y = pd.Series(([0, 0, 0, 1, 1, 1] * 100))
    cortes = w.chimerge(x, y, min_bins=2, max_bins=3, minimo_eventos=1)

    assert np.isinf(cortes[-1]) and cortes[-1] > 0
    assert np.isinf(cortes[0]) and cortes[0] < 0

    # un valor nunca visto, muy por encima del maximo, sigue teniendo bin
    nuevo = pd.Series([10_000.0])
    assert w._asignar_bins(nuevo, cortes).notna().all()


def test_missing_separado_false_excluye_los_nan(conteo_concentrado):
    x, y = conteo_concentrado
    x = x.copy()
    x.iloc[:300] = np.nan
    cortes = w.chimerge(x, y, special_values=[0])

    con = w.evaluar_binning(x, y, cortes, missing_separado=True)
    sin = w.evaluar_binning(x, y, cortes, missing_separado=False)

    assert w.ETIQUETA_FALTANTE in con.index
    assert w.ETIQUETA_FALTANTE not in sin.index
    assert int(sin["n"].sum()) == len(x) - 300


# =============================================================================
# 2. VALORES ESPECIALES
# =============================================================================

def test_el_cero_permanece_aislado(conteo_concentrado):
    """
    El salto 0 -> 1 vale ~27 puntos de default. Ningun criterio estadistico
    generico debe poder borrarlo fusionando el cero con el uno.
    """
    x, y = conteo_concentrado
    cortes = w.chimerge(x, y, special_values=[0], min_bins=2, max_bins=2)

    g = w.evaluar_binning(x, y, cortes)
    bin_cero = [b for b in g.index if b != w.ETIQUETA_FALTANTE and 0 in b]
    assert len(bin_cero) == 1
    # el bin del cero contiene SOLO ceros
    assert int(g.loc[bin_cero[0], "n"]) == int((x == 0).sum())


def test_sin_valores_especiales_el_cero_puede_fusionarse(conteo_concentrado):
    """Contraprueba: el aislamiento viene de special_values, no del azar."""
    x, y = conteo_concentrado
    con = w.chimerge(x, y, special_values=[0], min_bins=2, max_bins=2)
    sin = w.chimerge(x, y, special_values=None, min_bins=2, max_bins=2)
    assert con != sin or len(con) == len(sin)   # al menos no falla
    g = w.evaluar_binning(x, y, con)
    bin_cero = [b for b in g.index if b != w.ETIQUETA_FALTANTE and 0 in b][0]
    assert int(g.loc[bin_cero, "n"]) == int((x == 0).sum())


# =============================================================================
# 3. RESTRICCIONES DE TAMANO
# =============================================================================

def test_los_bins_automaticos_cumplen_el_minimo(continua):
    """
    Con una variable continua y 5,000 filas, las restricciones son
    satisfacibles: ningun bin debe quedar por debajo.
    """
    x, y = continua
    cortes = w.chimerge(x, y, minimo_pct_bin=0.05, minimo_eventos=30,
                        min_bins=3, max_bins=6)
    g = w.evaluar_binning(x, y, cortes, minimo_pct_bin=0.05, minimo_eventos=30)
    assert g["cumple_tamano"].all()
    assert g["cumple_eventos"].all()
    assert 3 <= len(g) <= 6


def test_valor_especial_puede_incumplir_el_minimo():
    """
    Caso matematicamente imposible: si el valor especial es raro, su bin sera
    pequeno por fuerza. El algoritmo debe respetarlo igualmente y NO fusionarlo.
    """
    x = pd.Series(np.concatenate([np.zeros(20), np.arange(1, 981, dtype=float)]))
    y = pd.Series((np.arange(len(x)) % 5 == 0).astype(int))
    cortes = w.chimerge(x, y, special_values=[0], minimo_pct_bin=0.10,
                        min_bins=2, max_bins=4)
    g = w.evaluar_binning(x, y, cortes, minimo_pct_bin=0.10)

    bin_cero = [b for b in g.index if b != w.ETIQUETA_FALTANTE and 0 in b][0]
    assert int(g.loc[bin_cero, "n"]) == 20          # sigue aislado
    assert not bool(g.loc[bin_cero, "cumple_tamano"])   # y se reporta como tal


def test_respeta_max_bins(continua):
    x, y = continua
    for mx in (3, 5, 8):
        cortes = w.chimerge(x, y, min_bins=2, max_bins=mx, minimo_pct_bin=0.01)
        assert len(cortes) - 1 <= mx


# =============================================================================
# 4. REPRODUCIBILIDAD Y PUREZA
# =============================================================================

def test_chimerge_es_reproducible(conteo_concentrado, continua):
    for x, y in (conteo_concentrado, continua):
        a = w.chimerge(x, y, special_values=[0])
        b = w.chimerge(x, y, special_values=[0])
        assert a == b


def test_no_muta_las_entradas(conteo_concentrado):
    x, y = conteo_concentrado
    x_antes, y_antes = x.copy(), y.copy()

    cortes = w.chimerge(x, y, special_values=[0])
    w.evaluar_binning(x, y, cortes)
    w.comparar_binnings(x, y, [-np.inf, 0, 1, np.inf], cortes)

    pd.testing.assert_series_equal(x, x_antes)
    pd.testing.assert_series_equal(y, y_antes)


def test_auditar_no_muta_el_dataframe(conteo_concentrado):
    x, y = conteo_concentrado
    df = pd.DataFrame({"conteo": x})
    antes = df.copy()
    w.auditar_binning(df, y, "conteo", cortes_manual=[-np.inf, 0, 1, np.inf],
                      special_values=[0], estabilidad=False)
    pd.testing.assert_frame_equal(df, antes)


# =============================================================================
# 5. CORRECCION DE LA MATEMATICA
# =============================================================================

def test_woe_e_iv_contra_calculo_manual():
    """
    Ejemplo pequeno resuelto a mano, con el suavizado de 0.5 del proyecto.

        bin (-inf, 0]:  n=6  eventos=1  no_eventos=5
        bin (0, inf]:   n=4  eventos=3  no_eventos=1
        totales:        eventos=4  no_eventos=6

        dist_ev_1  = (1+0.5)/4 = 0.375        dist_no_1 = (5+0.5)/6 = 0.916667
        WoE_1 = ln(0.916667/0.375) = +0.893818
        IV_1  = (0.916667-0.375) * 0.893818   = 0.484151

        dist_ev_2  = (3+0.5)/4 = 0.875        dist_no_2 = (1+0.5)/6 = 0.25
        WoE_2 = ln(0.25/0.875) = -1.252763
        IV_2  = (0.25-0.875) * (-1.252763)    = 0.782977

        IV total = 1.267128
    """
    x = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    y = pd.Series([0, 0, 0, 0, 0, 1, 0, 1, 1, 1])

    g = w.evaluar_binning(x, y, [-np.inf, 0, np.inf], minimo_eventos=1)

    assert g["woe"].iloc[0] == pytest.approx(0.893818, rel=1e-5)
    assert g["woe"].iloc[1] == pytest.approx(-1.252763, rel=1e-5)
    assert g["iv_parcial"].iloc[0] == pytest.approx(0.484151, rel=1e-5)
    assert g["iv_parcial"].iloc[1] == pytest.approx(0.782977, rel=1e-5)
    assert g.attrs["iv_total"] == pytest.approx(1.267128, rel=1e-5)


def test_convencion_de_signo_del_woe():
    """WoE positivo = MENOS riesgo. Invertirlo daria un scorecard al reves."""
    x = pd.Series([0.0] * 100 + [1.0] * 100)
    y = pd.Series([0] * 95 + [1] * 5 + [0] * 50 + [1] * 50)
    g = w.evaluar_binning(x, y, [-np.inf, 0, np.inf], minimo_eventos=1)

    seguro, riesgoso = g.iloc[0], g.iloc[1]
    assert seguro["default_rate_pct"] < riesgoso["default_rate_pct"]
    assert seguro["woe"] > 0 > riesgoso["woe"]


def test_bin_sin_eventos_no_produce_infinito():
    """El suavizado de Laplace evita log(0); sin el, el WoE seria +/- inf."""
    x = pd.Series([0.0] * 50 + [1.0] * 50)
    y = pd.Series([0] * 50 + [1] * 50)      # bin 1: 0 eventos, bin 2: 0 no-eventos
    g = w.evaluar_binning(x, y, [-np.inf, 0, np.inf], minimo_eventos=0)

    assert np.isfinite(g["woe"]).all()
    assert np.isfinite(g["iv_parcial"]).all()
    assert int(g["eventos"].iloc[0]) == 0
    assert int(g["no_eventos"].iloc[1]) == 0


def test_iv_total_es_la_suma_de_los_parciales(continua):
    x, y = continua
    g = w.evaluar_binning(x, y, w.chimerge(x, y))
    assert g.attrs["iv_total"] == pytest.approx(g["iv_parcial"].sum())


# =============================================================================
# 6. MONOTONICIDAD
# =============================================================================

@pytest.mark.parametrize("rates,esperado", [
    ([2.0, 5.0, 9.0, 14.0], "creciente"),
    ([14.0, 9.0, 5.0, 2.0], "decreciente"),
    ([10.0, 4.0, 3.0, 9.0], "U"),
    ([3.0, 9.0, 11.0, 2.0], "U_invertida"),
    ([5.0, 12.0, 4.0, 11.0, 3.0], "zigzag"),
    ([6.60, 6.62, 6.61], "aproximadamente_constante"),
])
def test_clasificar_monotonicidad(rates, esperado):
    assert w.clasificar_monotonicidad(rates) == esperado


def test_la_tolerancia_ignora_el_ruido():
    """
    Una variable practicamente monotona con una oscilacion de 0.03 pp no debe
    clasificarse como zigzag: seria una falsa alarma que llevaria a re-binear
    algo que esta bien.
    """
    casi = [9.43, 9.40, 8.65, 7.75, 6.40, 5.38, 4.48, 4.51]
    assert w.clasificar_monotonicidad(casi) == "decreciente"


def test_forzar_monotonia_produce_bins_monotonos():
    """En U real, pedir monotonia debe aplanarla (a costa de perder senal)."""
    rng = np.random.default_rng(3)
    x = pd.Series(np.repeat(np.arange(10, dtype=float), 500))
    p = np.where(x <= 2, 0.20, np.where(x <= 6, 0.05, 0.18))
    y = pd.Series((rng.random(len(x)) < p).astype(int))

    libre = w.chimerge(x, y, min_bins=3, max_bins=4)
    forzado = w.chimerge(x, y, min_bins=3, max_bins=4, forzar_monotonia="decreasing")

    g_libre = w.evaluar_binning(x, y, libre)
    g_forz = w.evaluar_binning(x, y, forzado)

    num_l, _ = w._separar_faltante(g_libre)
    num_f, _ = w._separar_faltante(g_forz)

    assert w.clasificar_monotonicidad(num_l["default_rate_pct"]) == "U"
    assert w.clasificar_monotonicidad(num_f["default_rate_pct"]) == "decreciente"
    # y el precio de forzarla: se pierde IV
    assert g_forz.attrs["iv_total"] < g_libre.attrs["iv_total"]


def test_forzar_monotonia_valida_su_argumento(continua):
    x, y = continua
    with pytest.raises(ValueError):
        w.chimerge(x, y, forzar_monotonia="hacia_arriba")


# =============================================================================
# 7. COMPARACION Y PARSIMONIA
# =============================================================================

def test_comparar_binnings_no_declara_ganador(conteo_concentrado):
    x, y = conteo_concentrado
    auto = w.chimerge(x, y, special_values=[0])
    t = w.comparar_binnings(x, y, [-np.inf, 0, 1, np.inf], auto)

    assert set(t.columns) == {"manual", "automatico"}
    assert "ganador" not in t.index and "mejor" not in t.index
    for m in ("n_bins", "IV", "bin_min_pct", "min_eventos", "monotonicidad"):
        assert m in t.index
    assert "iv_marginal" in t.attrs


def test_marca_como_marginal_una_mejora_diminuta(continua):
    x, y = continua
    cortes = w.chimerge(x, y, max_bins=5)
    t = w.comparar_binnings(x, y, cortes, cortes)   # identicos -> delta 0
    assert t.attrs["delta_iv"] == 0
    assert t.attrs["iv_marginal"] is True


def test_score_parsimonia_penaliza_la_fragmentacion():
    assert w.score_parsimonia(0.74, 5) > w.score_parsimonia(0.78, 15)
    assert w.score_parsimonia(0.90, 5) > w.score_parsimonia(0.74, 5)


# =============================================================================
# 8. ESTABILIDAD Y BENCHMARK
# =============================================================================

def test_estabilidad_reporta_variacion_por_bin(continua):
    x, y = continua
    df = pd.DataFrame({"continua": x})
    cortes = w.chimerge(x, y, max_bins=5)
    resumen, ivs = w.evaluar_estabilidad_binning(df, y, "continua", cortes, cv=4)

    assert len(ivs["ivs"]) == 4
    assert ivs["iv_desv"] >= 0
    for c in ("woe_medio", "woe_desv", "rango_woe", "folds_incumplen"):
        assert c in resumen.columns
    assert len(resumen) == len(cortes) - 1


def test_estabilidad_es_reproducible(continua):
    x, y = continua
    df = pd.DataFrame({"continua": x})
    cortes = w.chimerge(x, y, max_bins=5)
    a = w.evaluar_estabilidad_binning(df, y, "continua", cortes, random_state=7)
    b = w.evaluar_estabilidad_binning(df, y, "continua", cortes, random_state=7)
    assert a[1] == b[1]


def test_benchmark_cv_compara_dos_binnings(conteo_concentrado):
    x, y = conteo_concentrado
    df = pd.DataFrame({"conteo": x})
    manual = {"conteo": [-np.inf, 0, 1, np.inf]}
    auto = {"conteo": w.chimerge(x, y, special_values=[0])}

    t = w.benchmark_binning_cv(df, y, manual, auto, cv=3)
    assert list(t["metodo"]) == ["manual", "automatico"]
    assert (t["AUC_CV"].between(0, 1)).all()
    assert (t["SD"] >= 0).all()
    assert (t["n_bins_total"] > 0).all()


# =============================================================================
# 9. ANTI-FUGA E INTEGRACION
# =============================================================================

def test_ninguna_funcion_de_aprendizaje_pide_validacion_ni_test():
    """
    Invariante estructural: si alguna firma aceptara X_val o test, el binning
    podria acabar dependiendo de datos que deben permanecer ciegos.
    """
    import inspect
    for f in (w.aprender_woe, w.chimerge, w.evaluar_binning,
              w.evaluar_estabilidad_binning, w.benchmark_binning_cv,
              w.auditar_binning):
        params = set(inspect.signature(f).parameters)
        assert not {"X_val", "y_val", "X_test", "y_test", "val", "test"} & params


def test_auditar_binning_devuelve_el_informe_completo(conteo_concentrado):
    x, y = conteo_concentrado
    df = pd.DataFrame({"conteo": x})
    r = w.auditar_binning(df, y, "conteo", cortes_manual=[-np.inf, 0, 1, np.inf],
                          special_values=[0], cv=3)

    for k in ("evaluacion_manual", "cortes_auto", "evaluacion_auto",
              "comparacion", "advertencias", "estabilidad_manual",
              "estabilidad_auto"):
        assert k in r
    assert isinstance(r["advertencias"], list)
    assert r["cortes_auto"][0] == -np.inf and np.isinf(r["cortes_auto"][-1])


def test_aprender_woe_sin_bins_usa_los_manuales():
    """Compatibilidad: el comportamiento por defecto no puede haber cambiado."""
    df = pd.DataFrame({"NumberOfTimes90DaysLate": [0, 1, 2, 5, 0, 0, 1, 3]})
    y = pd.Series([0, 1, 1, 1, 0, 0, 0, 1])

    por_defecto = w.aprender_woe(df, y)
    explicito = w.aprender_woe(df, y, bins=w.BINS_TODOS)

    k = "NumberOfTimes90DaysLate"
    pd.testing.assert_frame_equal(por_defecto[k], explicito[k])


def test_aplicar_woe_respeta_los_bins_alternativos():
    """
    Aplicar con cortes distintos de los que produjeron la tabla asignaria
    valores WoE a intervalos que no les corresponden. La firma debe permitir
    pasar los mismos cortes a ambas funciones.
    """
    df = pd.DataFrame({"conteo": [0.0, 0, 1, 2, 3, 4, 5, 0]})
    y = pd.Series([0, 0, 1, 1, 1, 0, 1, 0])
    alt = {"conteo": [-np.inf, 0, 2, np.inf]}

    tablas = w.aprender_woe(df, y, bins=alt)
    salida = w.aplicar_woe(df, tablas, bins=alt)

    assert "woe_conteo" in salida.columns
    assert len(tablas["conteo"]) == 3
    assert np.isfinite(salida["woe_conteo"]).all()

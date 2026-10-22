# -*- coding: utf-8 -*-
"""
test_evaluation.py — Pruebas de las metricas y la regla de negocio

Se ejecuta con:  pytest tests/ -v

QUE PROBAMOS Y POR QUE

Estas funciones deciden si un modelo se despliega y a quien se le presta.
Un fallo aqui no produce un error visible: produce un modelo que parece
correcto y toma decisiones equivocadas.

  - signos_invertidos: si dejara de detectar un signo al reves, el
    scorecard afirmaria "mas atrasos -> menos riesgo" y nadie lo veria.
  - umbral_optimo: si la formula estuviera mal, todo el calculo economico
    del proyecto seria falso.
  - validar_submission: si no detectara un submission con filas de menos,
    la entrega seria invalida.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ))

from src.evaluation import (ks_statistic, gini, tabla_calibracion,
                            error_calibracion, signos_invertidos,
                            signos_incoherentes, direccion_univariada,
                            umbral_optimo, beneficio, beneficio_sin_modelo,
                            validar_submission, ic_wilson, tabla_separacion)


# =============================================================================
# 0. INFERENCIA SOBRE PROPORCIONES
# =============================================================================

def test_wilson_contiene_la_proporcion():
    """El intervalo debe contener siempre el punto estimado."""
    for p, n in [(0.0668, 120_000), (0.5467, 214), (0.0, 1), (1.0, 5)]:
        lo, hi = ic_wilson(p, n)
        assert lo <= p <= hi


def test_wilson_nunca_sale_de_cero_uno():
    """
    La ventaja sobre la aproximacion normal: con eventos raros, p +/- 1.96*EE
    puede devolver limites negativos. Wilson no.
    """
    lo, hi = ic_wilson(0.001, 50)
    assert lo >= 0 and hi <= 1


def test_wilson_se_estrecha_con_mas_datos():
    ancho = [ic_wilson(0.0668, n)[1] - ic_wilson(0.0668, n)[0]
             for n in (100, 1_000, 100_000)]
    assert ancho[0] > ancho[1] > ancho[2]


def test_wilson_sin_datos_es_nan():
    lo, hi = ic_wilson(0.0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_separacion_distingue_senal_de_ruido():
    """
    Dos indicadores con la MISMA direccion pero distinto tamano de muestra:
    solo el que tiene evidencia suficiente debe marcarse como que separa.
    """
    n = 2_000
    X = pd.DataFrame({
        "flag_fuerte": [1] * 200 + [0] * (n - 200),   # 200 filas, 50% de evento
        "flag_ruido":  [0] * (n - 10) + [1] * 10,     # 10 filas, tasa ~ global
    })
    y = pd.Series(0, index=X.index)
    y.iloc[:100] = 1                       # los 200 del flag fuerte -> 50%
    y.iloc[500:590] = 1                    # ~5% de base en el resto

    t = tabla_separacion(X, y, ["flag_fuerte", "flag_ruido"]).set_index("indicador")
    assert t.loc["flag_fuerte", "separa"]
    assert not t.loc["flag_ruido", "separa"]

    # Y los que separan van primero, aunque el ruidoso tenga mas "distancia".
    orden = tabla_separacion(X, y, ["flag_ruido", "flag_fuerte"])["indicador"]
    assert orden.iloc[0] == "flag_fuerte"


def test_separacion_ignora_indicadores_constantes():
    """Un flag que vale siempre 0 (o siempre 1) no admite contraste."""
    X = pd.DataFrame({"flag_vacio": [0] * 100, "flag_lleno": [1] * 100})
    y = pd.Series([0, 1] * 50)
    assert tabla_separacion(X, y, ["flag_vacio", "flag_lleno"]).empty


# =============================================================================
# 1. METRICAS DE DISCRIMINACION
# =============================================================================

def test_ks_de_un_modelo_perfecto_es_uno():
    """Si el modelo separa perfectamente las dos clases, KS = 1."""
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.1] * 50 + [0.9] * 50)
    ks, _ = ks_statistic(y, p)
    assert ks == pytest.approx(1.0)


def test_ks_de_un_modelo_al_azar_es_casi_cero():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 5000)
    p = rng.random(5000)          # sin ninguna relacion con y
    ks, _ = ks_statistic(y, p)
    assert ks < 0.10


def test_gini():
    """Gini = 2*AUC - 1. Azar (0.5) -> 0. Perfecto (1.0) -> 1."""
    assert gini(0.5) == pytest.approx(0.0)
    assert gini(1.0) == pytest.approx(1.0)
    assert gini(0.86272) == pytest.approx(0.72544)


# =============================================================================
# 2. CALIBRACION
# =============================================================================

def test_modelo_perfectamente_calibrado_tiene_error_cero():
    """
    Construimos datos donde la PD predicha coincide EXACTAMENTE con la
    frecuencia real de cada grupo. El error debe ser ~0.
    """
    filas = []
    for pd_verdadera in [0.01, 0.05, 0.10, 0.20, 0.40]:
        n = 2000
        n_malos = int(n * pd_verdadera)
        filas += [(pd_verdadera, 1)] * n_malos + [(pd_verdadera, 0)] * (n - n_malos)
    d = pd.DataFrame(filas, columns=["p", "y"])
    assert error_calibracion(d.y, d.p, n_grupos=5) < 0.5   # en puntos porcentuales


def test_modelo_descalibrado_se_detecta():
    """
    Mismo ordenamiento, pero las probabilidades estan infladas al doble.
    El AUC seria identico; la calibracion debe delatarlo.
    """
    filas = []
    for pd_verdadera in [0.01, 0.05, 0.10, 0.20, 0.40]:
        n = 2000
        n_malos = int(n * pd_verdadera)
        filas += [(pd_verdadera * 2, 1)] * n_malos + [(pd_verdadera * 2, 0)] * (n - n_malos)
    d = pd.DataFrame(filas, columns=["p", "y"])
    assert error_calibracion(d.y, d.p, n_grupos=5) > 5.0


def test_tabla_calibracion_tiene_las_columnas_esperadas():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    t = tabla_calibracion(y, p)
    assert list(t.columns) == ["n", "pd_pred", "pd_real", "error_pp"]
    assert len(t) == 10
    assert t["n"].sum() == 1000


# =============================================================================
# 3. COHERENCIA DEL SCORECARD  (la comprobacion critica)
# =============================================================================

def test_detecta_signo_invertido():
    """
    En variables WoE el coeficiente debe ser NEGATIVO. Uno positivo dice
    "mas atrasos -> menos riesgo": absurdo, e inaceptable ante un regulador.
    """
    cols = ["woe_age", "woe_atrasos", "flag_algo"]
    coefs = [-0.4, +0.3, +0.9]     # el segundo esta mal; el flag es libre
    assert signos_invertidos(coefs, cols) == ["woe_atrasos"]


def test_ningun_signo_invertido_cuando_todos_son_negativos():
    cols = ["woe_age", "woe_atrasos"]
    assert signos_invertidos([-0.4, -0.3], cols) == []


def test_los_flags_no_tienen_signo_obligado():
    """
    Un flag puede ir en cualquier direccion segun lo que codifique
    (ej: 'sin ingreso reportado' resulta MENOS riesgoso). Solo las
    variables woe_* tienen signo esperado.

    Esta exencion es precisamente la que oculto el hallazgo H1; la
    comprobacion completa es signos_incoherentes(), probada mas abajo.
    """
    cols = ["flag_income_faltante", "flag_registro_invalido"]
    assert signos_invertidos([+2.0, -2.0], cols) == []


# =============================================================================
# 3b. LA COMPROBACION COMPLETA  (hallazgo H1)
# =============================================================================

@pytest.fixture
def datos_flags():
    """
    Dos indicadores con direcciones univariadas OPUESTAS:
      flag_peligroso : cuando vale 1, la tasa de evento SUBE  (50% vs 5%)
      flag_protector : cuando vale 1, la tasa de evento BAJA  (0% vs ~20%)
    """
    X = pd.DataFrame({
        "flag_peligroso": [1] * 100 + [0] * 900,
        "flag_protector": [0] * 500 + [1] * 500,
        "woe_algo":       np.linspace(-2, 2, 1000),
    })
    y = pd.Series([1] * 50 + [0] * 50 + [1] * 45 + [0] * 855)
    return X, y


def test_direccion_univariada_detecta_ambos_sentidos(datos_flags):
    X, y = datos_flags
    assert direccion_univariada(X, y, "flag_peligroso") == +1
    assert direccion_univariada(X, y, "flag_protector") == -1


def test_direccion_univariada_devuelve_cero_si_el_grupo_esta_vacio():
    X = pd.DataFrame({"flag_nunca": [0] * 100})
    y = pd.Series([0] * 95 + [1] * 5)
    assert direccion_univariada(X, y, "flag_nunca") == 0


def test_detecta_flag_que_contradice_su_direccion(datos_flags):
    """
    EL CASO DE H1: flag_peligroso marca un grupo con 50% de eventos frente
    al 5% del resto, pero su coeficiente es NEGATIVO — es decir, en el
    scorecard le sumaria puntos a favor. Indefendible.
    """
    X, y = datos_flags
    cols = ["flag_peligroso", "flag_protector", "woe_algo"]
    coefs = [-1.5, -0.8, -0.5]        # el primero contradice los datos
    assert signos_incoherentes(coefs, cols, X, y) == ["flag_peligroso"]


def test_acepta_flags_coherentes_en_ambos_sentidos(datos_flags):
    """Un flag protector CON coeficiente negativo es correcto, no un defecto."""
    X, y = datos_flags
    cols = ["flag_peligroso", "flag_protector", "woe_algo"]
    coefs = [+1.5, -0.8, -0.5]
    assert signos_incoherentes(coefs, cols, X, y) == []


def test_sigue_exigiendo_signo_negativo_a_las_variables_woe(datos_flags):
    X, y = datos_flags
    cols = ["flag_peligroso", "woe_algo"]
    assert signos_incoherentes([+1.5, +0.5], cols, X, y) == ["woe_algo"]


def test_la_comprobacion_completa_es_mas_estricta_que_la_antigua(datos_flags):
    """
    signos_incoherentes() debe detectar todo lo que detecta
    signos_invertidos(), y ademas los flags. Es la garantia de que la
    sustitucion no pierde cobertura.
    """
    X, y = datos_flags
    cols = ["flag_peligroso", "woe_algo"]
    coefs = [-1.5, +0.5]              # ambos mal
    antigua = set(signos_invertidos(coefs, cols))
    completa = set(signos_incoherentes(coefs, cols, X, y))
    assert antigua == {"woe_algo"}
    assert antigua < completa                      # subconjunto estricto
    assert completa == {"woe_algo", "flag_peligroso"}


# =============================================================================
# 4. LA REGLA DE NEGOCIO
# =============================================================================

def test_umbral_optimo_formula():
    """umbral = M / (M + L).  Con M=800 y L=6000 -> 0.1176"""
    assert umbral_optimo(800, 6000) == pytest.approx(800 / 6800)
    assert umbral_optimo(800, 6000) == pytest.approx(0.11765, abs=1e-5)


def test_umbral_sube_cuando_las_perdidas_son_baratas():
    """
    Si perder cuesta poco, puedes permitirte aceptar mas riesgo.
    Si perder cuesta mucho, el umbral se vuelve exigente.
    """
    barato = umbral_optimo(800, 400)
    caro = umbral_optimo(800, 12000)
    assert barato > caro
    assert barato > 0.5      # con L < M conviene prestar casi a cualquiera
    assert caro < 0.1


def test_umbral_sin_perdidas_aprueba_a_todos():
    """Si un default no costara nada, prestarias a todo el mundo."""
    assert umbral_optimo(800, 0) == pytest.approx(1.0)


def test_umbral_rechaza_parametros_absurdos():
    with pytest.raises(ValueError):
        umbral_optimo(0, 6000)
    with pytest.raises(ValueError):
        umbral_optimo(800, -100)


def test_beneficio_cuenta_bien():
    #             PD:  0.05  0.05  0.50  0.50
    p = np.array([0.05, 0.05, 0.50, 0.50])
    y = np.array([0,    1,    0,    1])
    # Umbral 0.10 -> se aprueban las dos primeras: 1 bueno y 1 moroso
    r = beneficio(p, y, 0.10, margen=800, perdida=6000)
    assert r["aprobados"] == 2
    assert r["buenos"] == 1
    assert r["morosos"] == 1
    assert r["beneficio"] == 800 - 6000


def test_beneficio_con_umbral_uno_equivale_a_no_tener_modelo():
    rng = np.random.default_rng(7)
    p = rng.random(500) * 0.9
    y = rng.integers(0, 2, 500)
    assert (beneficio(p, y, 1.0, 800, 6000)["beneficio"]
            == beneficio_sin_modelo(y, 800, 6000))


def test_beneficio_con_umbral_cero_es_no_prestar_a_nadie():
    p = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 1, 0])
    r = beneficio(p, y, 0.0, 800, 6000)
    assert r["aprobados"] == 0 and r["beneficio"] == 0


def test_un_modelo_bueno_supera_a_no_tener_modelo():
    """
    Con un modelo que ordena bien y perdidas caras, filtrar tiene que
    ganarle a aprobar a todo el mundo.
    """
    rng = np.random.default_rng(1)
    n = 4000
    y = (rng.random(n) < 0.07).astype(int)
    # Predicciones correlacionadas con y (modelo decente, no perfecto)
    p = np.clip(y * 0.35 + rng.normal(0.05, 0.06, n), 0.001, 0.999)
    u = umbral_optimo(800, 6000)
    assert (beneficio(p, y, u, 800, 6000)["beneficio"]
            > beneficio_sin_modelo(y, 800, 6000))


# =============================================================================
# 5. EL SUBMISSION
# =============================================================================

def _sub_valida(n=100):
    return pd.DataFrame({"Id": np.arange(1, n + 1),
                         "Probability": np.linspace(0.01, 0.9, n)})


def test_submission_correcta_pasa_todo():
    assert all(ok for _, ok in validar_submission(_sub_valida(), 100))


def test_detecta_filas_de_menos():
    """
    El error mas caro: hay que entregar una prediccion para TODAS las
    personas del test, no solo para las que el pipeline no descarto.
    """
    resultado = dict((d, ok) for d, ok in validar_submission(_sub_valida(99), 100))
    assert not resultado["filas: 99 = 100"]


def test_detecta_probabilidad_fuera_de_rango():
    s = _sub_valida()
    s.loc[0, "Probability"] = 1.5
    fallos = [d for d, ok in validar_submission(s, 100) if not ok]
    assert "probabilidades dentro de [0, 1]" in fallos


def test_detecta_valores_faltantes():
    s = _sub_valida()
    s.loc[0, "Probability"] = np.nan
    fallos = [d for d, ok in validar_submission(s, 100) if not ok]
    assert "sin valores faltantes" in fallos


def test_detecta_ids_duplicados():
    s = _sub_valida()
    s.loc[1, "Id"] = 1
    fallos = [d for d, ok in validar_submission(s, 100) if not ok]
    assert "ids unicos" in fallos


def test_detecta_columnas_mal_nombradas():
    s = _sub_valida().rename(columns={"Probability": "prob"})
    fallos = [d for d, ok in validar_submission(s, 100) if not ok]
    assert "columnas y nombres correctos" in fallos

# -*- coding: utf-8 -*-
"""
test_preprocessing.py — Pruebas del pipeline de preprocesamiento

Se ejecuta con:  pytest tests/ -v      (desde la carpeta del proyecto)

QUE PROBAMOS Y POR QUE

No probamos "que el codigo corra" — eso ya lo sabemos. Probamos los
INVARIANTES que, si se rompen, producen un modelo que parece funcionar
pero esta mal:

  1. Anti-fuga: la transformacion no puede depender del target.
  2. Integridad: no se puede perder ni una fila (en test hay que predecir
     las 101,503).
  3. Codigos centinela: los valores corruptos no pueden colarse como
     numeros reales.
  4. Reproducibilidad: el artefacto guardado tiene que dar el mismo
     resultado que el pipeline en vivo.

Los tests que empiezan con "test_regresion_" existen porque ese bug YA
ocurrio una vez. Estan ahi para que no vuelva.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ))

from src import preprocessing as pp
from src import woe as woe_mod
from src import artifacts


# =============================================================================
# DATOS DE PRUEBA
# =============================================================================

@pytest.fixture
def df_crudo():
    """
    Dataset minimo que contiene DELIBERADAMENTE una fila de cada patologia
    que encontramos en el EDA.
    """
    return pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": [0.5, 1.5, 50708.0, 0.2, 0.9],
        "age":                                  [45,  30,  0,       60,  22],
        "NumberOfTime30-59DaysPastDueNotWorse": [0,   2,   98,      1,   0],
        "DebtRatio":                            [0.4, 1.2, 5000.0,  0.3, 0.8],
        "MonthlyIncome":                        [5000.0, np.nan, np.nan, 0.0, 3000.0],
        "NumberOfOpenCreditLinesAndLoans":      [5,   8,   3,       10,  2],
        "NumberOfTimes90DaysLate":              [0,   1,   98,      0,   0],
        "NumberRealEstateLoansOrLines":         [1,   0,   2,       3,   0],
        "NumberOfTime60-89DaysPastDueNotWorse": [0,   0,   98,      0,   0],
        "NumberOfDependents":                   [2.0, np.nan, np.nan, 1.0, 0.0],
    }, index=pd.Index([1, 2, 3, 4, 5], name="row_id"))


@pytest.fixture
def y_prueba():
    return pd.Series([0, 1, 1, 0, 0], index=[1, 2, 3, 4, 5])


# =============================================================================
# 1. INTEGRIDAD: nunca se pierden filas
# =============================================================================

def test_limpiar_no_pierde_filas(df_crudo):
    """
    En test hay que entregar una prediccion para las 101,503 personas.
    Un pipeline que borre filas se rompe en produccion.
    """
    salida = pp.limpiar(df_crudo)
    assert len(salida) == len(df_crudo)
    assert salida.index.equals(df_crudo.index)


def test_pipeline_completo_no_pierde_filas(df_crudo):
    salida = pp.crear_features(pp.limpiar(df_crudo))
    assert len(salida) == len(df_crudo)


def test_limpiar_no_modifica_el_original(df_crudo):
    """limpiar() debe devolver una copia, no mutar la entrada."""
    antes = df_crudo.copy()
    pp.limpiar(df_crudo)
    pd.testing.assert_frame_equal(df_crudo, antes)


# =============================================================================
# 2. CODIGOS CENTINELA Y VALORES CORRUPTOS
# =============================================================================

def test_centinelas_van_a_nan(df_crudo):
    """Los 96/98 no pueden sobrevivir como numeros: 98 no son 98 atrasos."""
    salida = pp.limpiar(df_crudo)
    for col in pp.COLS_MORA:
        assert not salida[col].isin(pp.SENTINELAS).any()
    # La fila 3 tenia 98 en las tres columnas
    assert salida.loc[3, pp.COLS_MORA].isna().all()


def test_centinelas_dejan_flag(df_crudo):
    """La informacion no se pierde: el flag la conserva (54.67% de default)."""
    salida = pp.limpiar(df_crudo)
    assert salida.loc[3, "flag_registro_invalido"] == 1
    assert salida.loc[1, "flag_registro_invalido"] == 0


def test_revutil_absurdo_va_a_nan(df_crudo):
    """50,708 no es una utilizacion de credito: es ruido de datos."""
    salida = pp.limpiar(df_crudo)
    assert pd.isna(salida.loc[3, "RevolvingUtilizationOfUnsecuredLines"])
    assert salida.loc[3, "flag_revutil_invalido"] == 1


def test_revutil_mayor_que_uno_SE_CONSERVA(df_crudo):
    """
    CRITICO: los valores entre 1 y 100 son sobregiros LEGITIMOS y el grupo
    mas predictivo del dataset (38-48% de default). Capearlos en 1 habria
    invalidado 3,321 filas con 37.25% de default.
    """
    salida = pp.limpiar(df_crudo)
    assert salida.loc[2, "RevolvingUtilizationOfUnsecuredLines"] == 1.5
    assert salida.loc[2, "flag_revutil_invalido"] == 0


def test_ingreso_cero_va_a_nan(df_crudo):
    """Un ingreso de 0 no es pobreza: es 'no reportado' codificado."""
    salida = pp.limpiar(df_crudo)
    assert pd.isna(salida.loc[4, "MonthlyIncome"])
    assert salida.loc[4, "flag_income_cero"] == 1
    # Un ingreso real no se toca
    assert salida.loc[1, "MonthlyIncome"] == 5000.0
    assert salida.loc[1, "flag_income_cero"] == 0


def test_edad_invalida_va_a_nan(df_crudo):
    salida = pp.limpiar(df_crudo)
    assert pd.isna(salida.loc[3, "age"])
    assert salida.loc[3, "flag_edad_invalida"] == 1


def test_flags_son_binarios(df_crudo):
    """Un flag con NaN romperia la regresion logistica."""
    salida = pp.crear_features(pp.limpiar(df_crudo))
    for col in woe_mod.columnas_flag(salida):
        assert salida[col].isin([0, 1]).all(), f"{col} no es binario"


def test_debtratio_se_separa_por_denominador_no_por_magnitud(df_crudo):
    """
    DebtRatio es un ratio SOLO si habia ingreso con que dividir.

    Las dos filas discriminantes de la fixture son las que un umbral de
    magnitud clasificaria al reves:
        fila 2 -> DebtRatio = 1.2 (pequeno) pero SIN ingreso  -> es un importe
        fila 4 -> DebtRatio = 0.3 (pequeno) e ingreso = 0     -> es un importe
    Con la regla antigua (DebtRatio > 2) ambas habrian pasado por ratios validos.
    """
    salida = pp.limpiar(df_crudo)

    # Con denominador -> ratio limpio, sin importe
    for fila in (1, 5):
        assert pd.notna(salida.loc[fila, "DebtRatio_limpio"])
        assert pd.isna(salida.loc[fila, "deuda_bruta_proxy"])

    # Sin denominador -> importe, sin ratio (incluso con DebtRatio pequeno)
    for fila in (2, 3, 4):
        assert pd.isna(salida.loc[fila, "DebtRatio_limpio"])
        assert pd.notna(salida.loc[fila, "deuda_bruta_proxy"])

    # Las dos columnas son complementarias: cada fila cae en una y solo una.
    assert (salida["DebtRatio_limpio"].isna() ^ salida["deuda_bruta_proxy"].isna()).all()


def test_no_existe_flag_debtratio_invalido(df_crudo):
    """
    Se elimino a proposito: bajo la regla del denominador seria identico a
    (flag_income_faltante | flag_income_cero), es decir colinealidad perfecta.
    Es el mecanismo que produjo el hallazgo H1 de la auditoria.
    """
    salida = pp.crear_features(pp.limpiar(df_crudo))
    assert "flag_debtratio_invalido" not in salida.columns

    # Y se comprueba la equivalencia que lo hacia redundante.
    sin_denominador = salida["deuda_bruta_proxy"].notna()
    union_flags = (salida["flag_income_faltante"] | salida["flag_income_cero"]).astype(bool)
    assert sin_denominador.equals(union_flags)


# =============================================================================
# 3. REGRESION: bugs que ya ocurrieron una vez
# =============================================================================

def test_regresion_peor_atraso_no_traga_centinelas(df_crudo):
    """
    BUG REAL (corregido): peor_atraso se calculaba con comparaciones "> 0".
    Como NaN > 0 es False, las filas con centinela caian en peor_atraso = 0,
    o sea "sin atrasos" — justo el grupo MAS seguro (2.74% de default),
    cuando en realidad son el mas peligroso (54.67%).

    Su historial es DESCONOCIDO, no cero.
    """
    salida = pp.crear_features(pp.limpiar(df_crudo))
    assert pd.isna(salida.loc[3, "peor_atraso"]), \
        "las filas con centinela deben tener peor_atraso = NaN, no 0"
    # Y las filas normales si se calculan
    assert salida.loc[1, "peor_atraso"] == 0     # sin atrasos
    assert salida.loc[2, "peor_atraso"] == 3     # llego a 90+ dias


def test_regresion_woe_registra_flags_nuevos(df_crudo, y_prueba):
    """
    BUG POTENCIAL: si las columnas flag se listaran a mano en woe.py,
    anadir un flag en preprocessing.py y olvidar registrarlo lo dejaria
    fuera del dataset SIN error visible. Se detectan por prefijo.
    """
    limpio = pp.crear_features(pp.limpiar(df_crudo))
    limpio["flag_inventado_para_la_prueba"] = 1

    tablas = woe_mod.aprender_woe(limpio, y_prueba)
    salida = woe_mod.aplicar_woe(limpio, tablas)

    assert "flag_inventado_para_la_prueba" in salida.columns


# =============================================================================
# 4. ANTI-FUGA DE DATOS
# =============================================================================

def test_aplicar_woe_no_necesita_el_target(df_crudo, y_prueba):
    """
    Si aplicar_woe() necesitara el target, no podria usarse en test
    (que no lo tiene) y habria fuga en validacion.
    """
    limpio = pp.crear_features(pp.limpiar(df_crudo))
    tablas = woe_mod.aprender_woe(limpio, y_prueba)

    # Sin target en ningun sitio: debe funcionar igual.
    sin_target = limpio.drop(columns=[pp.TARGET], errors="ignore")
    salida = woe_mod.aplicar_woe(sin_target, tablas)
    assert len(salida) == len(df_crudo)


def test_woe_es_determinista(df_crudo, y_prueba):
    """Aplicar dos veces las mismas tablas debe dar exactamente lo mismo."""
    limpio = pp.crear_features(pp.limpiar(df_crudo))
    tablas = woe_mod.aprender_woe(limpio, y_prueba)
    pd.testing.assert_frame_equal(
        woe_mod.aplicar_woe(limpio, tablas),
        woe_mod.aplicar_woe(limpio, tablas),
    )


def test_woe_sin_nan_ni_infinitos(df_crudo, y_prueba):
    """LogReg no acepta NaN ni infinitos. Los NaN deben ser un bin, no un hueco."""
    limpio = pp.crear_features(pp.limpiar(df_crudo))
    tablas = woe_mod.aprender_woe(limpio, y_prueba)
    salida = woe_mod.aplicar_woe(limpio, tablas)

    assert salida.isna().sum().sum() == 0
    assert np.isfinite(salida.to_numpy()).all()


def test_bin_faltante_existe_cuando_hay_nan(df_crudo, y_prueba):
    """Los NaN deben tener su propio bin con su propio nivel de riesgo."""
    limpio = pp.crear_features(pp.limpiar(df_crudo))
    tablas = woe_mod.aprender_woe(limpio, y_prueba)
    assert "FALTANTE" in tablas["MonthlyIncome"].index


# =============================================================================
# 5. SIGNO DEL WoE
# =============================================================================

def test_signo_del_woe():
    """
    Convencion: WoE = ln(%no-eventos / %eventos)
      positivo -> el bin tiene MENOS default que la media (mas seguro)
      negativo -> el bin tiene MAS default que la media  (mas riesgoso)
    Si el signo se invirtiera, los coeficientes del scorecard tendrian
    el sentido contrario y nadie lo notaria de inmediato.
    """
    n = 1000
    df = pd.DataFrame({"age": [25] * (n // 2) + [70] * (n // 2)})
    # Los jovenes fallan mucho, los mayores casi nada
    y = pd.Series([1] * 400 + [0] * 100 + [1] * 10 + [0] * 490)

    tablas = woe_mod.aprender_woe(df, y)
    t = tablas["age"]

    woe_jovenes = t.loc[[i for i in t.index if i != "FALTANTE" and i.right <= 30][0], "woe"]
    woe_mayores = t.loc[[i for i in t.index if i != "FALTANTE" and i.left >= 65][0], "woe"]

    assert woe_jovenes < 0, "un grupo de alto riesgo debe tener WoE negativo"
    assert woe_mayores > 0, "un grupo de bajo riesgo debe tener WoE positivo"


# =============================================================================
# 6. EL ARTEFACTO GUARDADO
# =============================================================================

def test_artefacto_reproduce_la_transformacion(df_crudo, y_prueba, tmp_path):
    """
    La prueba de fuego: guardar el transformador, recargarlo y aplicarlo a
    datos crudos debe dar EXACTAMENTE lo mismo que el pipeline en vivo.
    Sin esto no se puede puntuar a un cliente nuevo de forma auditable.
    """
    limpio = pp.crear_features(pp.limpiar(df_crudo))
    tablas = woe_mod.aprender_woe(limpio, y_prueba)
    esperado = woe_mod.aplicar_woe(limpio, tablas)

    art = artifacts.construir_artefacto(
        tablas_woe=tablas,
        columnas_trees=pp.preparar_arboles(limpio).columns,
        columnas_woe=esperado.columns,
        n_filas_train=len(df_crudo),
        tasa_default=y_prueba.mean(),
        semilla=42,
    )
    ruta = artifacts.guardar(art, tmp_path / "test.joblib")
    recargado = artifacts.cargar(ruta)

    obtenido = artifacts.transformar_datos_nuevos(df_crudo, recargado, rama="woe")
    pd.testing.assert_frame_equal(obtenido, esperado)


def test_artefacto_detecta_pipeline_desincronizado(df_crudo, y_prueba, tmp_path):
    """
    Si el codigo cambia despues de entrenar, el artefacto debe FALLAR
    ruidosamente en vez de devolver un dataset silenciosamente distinto.
    """
    limpio = pp.crear_features(pp.limpiar(df_crudo))
    tablas = woe_mod.aprender_woe(limpio, y_prueba)

    art = artifacts.construir_artefacto(
        tablas_woe=tablas,
        columnas_trees=pp.preparar_arboles(limpio).columns,
        columnas_woe=list(woe_mod.aplicar_woe(limpio, tablas).columns) + ["columna_fantasma"],
        n_filas_train=len(df_crudo),
        tasa_default=y_prueba.mean(),
        semilla=42,
    )
    ruta = artifacts.guardar(art, tmp_path / "roto.joblib")

    with pytest.raises(ValueError, match="pipeline cambio"):
        artifacts.transformar_datos_nuevos(df_crudo, artifacts.cargar(ruta), rama="woe")

# 09 — Guía del usuario y del desarrollador

[← Comparación metodológica](08_comparacion_metodologica.md) · [Índice](README.md) · [Siguiente: Referencias →](10_referencias.md)

---

## ⛔ 9.0 Antes de usar este sistema: dos brechas abiertas

> **Este proyecto NO es desplegable en un entorno bancario en producción.**
> Todo lo que sigue explica cómo ejecutarlo, extenderlo y mantenerlo — pero
> **no** cómo ponerlo en producción, porque no está listo para eso.

Dos hallazgos críticos permanecen **abiertos** y ninguno es subsanable con el
conjunto de datos actual:

### H2 — Sin validación *out-of-time*

**Qué falta.** Toda la validación es sobre particiones **aleatorias** de la misma
población. No existe evaluación sobre un periodo temporal posterior.

**Por qué bloquea.** BCBS (2005) y la práctica supervisora exigen demostrar que
el modelo mantiene su poder discriminante **en el tiempo**. Una partición
aleatoria mide estabilidad *muestral*; una partición temporal mide estabilidad
*temporal*. **Son propiedades distintas**: un modelo puede tener AUC 0.87 en
validación aleatoria y 0.72 doce meses después, por cambios en el ciclo
económico, en la política de originación o en la composición de la demanda.

**Por qué no se puede arreglar aquí.** `cs-training.csv` **no contiene ninguna
marca temporal** —ni fecha de originación, ni de solicitud, ni cohorte—. Es
imposible construir una partición temporal.

**Qué haría falta.**
1. Una variable de fecha en el conjunto de originación.
2. Partición por cohortes: entrenar con originaciones hasta $T$, validar con
   $T+1, T+2, \dots$
3. Curvas de AUC y KS por cohorte para detectar degradación.

### H3 — Sin marco de monitoreo ni backtesting

**Qué falta.** No existe procedimiento definido para detectar degradación tras el
despliegue.

**Por qué bloquea.** Un modelo de crédito en producción requiere vigilancia
continua. Sin ella, la degradación se detecta **cuando ya ha generado pérdidas**.

**Nota importante.** El PSI se evaluó y se descartó con justificación correcta:
con particiones aleatorias de la misma población, el PSI máximo entre las 15
variables es **0.00073** frente a un umbral de alerta de 0.10 — no hay deriva que
medir. El descarte fue correcto; lo que falta es el **marco**, que solo puede
construirse cuando existan datos temporales.

**Qué haría falta.**

| Indicador | Frecuencia | Umbral de acción |
|---|---|---|
| PSI del score | Mensual | > 0.10 vigilar, > 0.25 recalibrar |
| CSI por variable | Mensual | > 0.10 investigar |
| Calibración por cohorte | Trimestral | Desviación > 2 pp |
| AUC móvil 12 meses | Trimestral | Caída > 0.03 |
| Tasa de aprobación | Semanal | Desviación > 5 pp del plan |

### Qué SÍ está resuelto

| Hallazgo | Estado |
|---|---|
| H1 — Scorecard con atribución contraria al riesgo | ✅ corregido y verificado |
| H4 — Punto ciego en la comprobación de signos | ✅ corregido, 6 pruebas nuevas |
| H11 — Código muerto | ✅ corregido |

Detalle completo en el [capítulo 07](07_auditoria.md).

---

## 9.1 Instalación

### 9.1.1 Requisitos

- Python **3.11 o superior** (desarrollado y verificado con 3.13.9)
- ~500 MB de espacio (datos originales + artefactos derivados)
- Sin GPU: todo corre en CPU en 113 segundos

### 9.1.2 Entorno

```bash
cd "Market Risk/Give Me Some Credit (PD)/programs/Probability_of_Default"

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 9.1.3 Verificación de la instalación

```bash
pytest tests/ -q
# Esperado: 48 passed
```

Las pruebas usan datos sintéticos, luego funcionan **antes** de generar nada.

### 9.1.4 Ubicación de los datos

Los CSV originales deben estar en `../../data/` respecto al proyecto:

```
Give Me Some Credit (PD)/
├── data/
│   ├── cs-training.csv
│   ├── cs-test.csv
│   ├── sampleEntry.csv
│   └── Data Dictionary.xls
└── programs/Probability_of_Default/    ← ejecutar desde aquí
```

Si se mueven, hay que actualizar la constante `DATA` en `01`, `02` y `06`.

---

## 9.2 Ejecución

### 9.2.1 Cadena completa

```bash
python 01_eda.py                  #  15 s → outputs/*.png
python 02_preprocessing.py        #   6 s → datasets/ + preprocessor_v1.joblib
python 03_train_logistic.py       #  14 s → logistic_v1.joblib
python 04_train_xgboost.py        #  39 s → xgboost_v1.joblib
python 05_train_lightgbm.py       #  21 s → lightgbm_v1.joblib + comparación CV
python 06_evaluation.py           #  10 s → submission + preprocessor_final
python 07_business_evaluation.py  #   4 s → análisis económico
python 08_scorecard.py            #   4 s → scorecard.csv
```

**Códigos de salida:** `02`, `06` y `08` devuelven `1` si alguna verificación
interna falla. Aptos para uso en CI.

### 9.2.2 Inferencia

```bash
# Demostración con tres perfiles inventados
python predict.py --demo

# Sobre un CSV real
python predict.py --entrada solicitantes.csv --salida decisiones.csv

# Cambiando el modelo
python predict.py --entrada solicitantes.csv --modelo logistic_v1

# Cambiando la política de crédito (sin tocar el modelo)
python predict.py --entrada solicitantes.csv --margen 800 --perdida 20000
```

**Formato de entrada:** las 10 columnas originales más una columna de índice.
`SeriousDlqin2yrs` se ignora si está presente.

**Formato de salida:**

```
row_id, PD, PD_pct, score, decision
1,      0.009875, 0.99, 620, APROBAR
2,      0.119429, 11.94, 545, RECHAZAR
```

---

## 9.3 Errores comunes

| Síntoma | Causa | Solución |
|---|---|---|
| `FileNotFoundError: cs-training.csv` | Ejecutado desde otro directorio | `cd` a la carpeta del proyecto |
| `No hay modelos en artifacts/models/` | `07` antes que `03`/`04`/`05` | Ejecutar en orden |
| `No existe preprocessor_v1.joblib` | `predict.py` antes que `02` | Ejecutar `02` primero |
| `ValueError: El artefacto espera columnas que el codigo actual no genera` | `src/` cambió tras entrenar | Re-ejecutar desde `02` |
| `Al CSV le faltan columnas obligatorias` | Formato de entrada incorrecto | Comparar con `COLUMNAS_REQUERIDAS` en `predict.py` |
| `05` falla con `KeyError: 'logistic_v1'` | `03` no ejecutado | `05` depende de `03` y `04` |
| `shap no esta instalado` | Dependencia opcional ausente | `pip install shap` (o ignorar: el script continúa) |
| Warning de convergencia en `LogisticRegression` | `max_iter` insuficiente | No debería ocurrir (uso real: 27 de 2,000) |
| `bagging_fraction` sin efecto en LightGBM | Falta `bagging_freq` | Ya fijado a 1 en el código |
| Resultados distintos entre ejecuciones | Semilla modificada | `SEMILLA = 42` en `02`–`05` |

---

## 9.4 Depuración

### 9.4.1 Verificar la integridad del pipeline

```bash
python 02_preprocessing.py 2>&1 | grep -E "\[OK\]|\[ERROR\]"
# Esperado: 14 líneas [OK], 0 [ERROR]
```

### 9.4.2 Inspeccionar un artefacto

```python
import sys; sys.path.insert(0, ".")
from src import artifacts

art = artifacts.cargar("artifacts/preprocessors/preprocessor_final.joblib")
print(artifacts.resumen(art))
print(art["tablas_woe"]["age"])          # tabla WoE de una variable
print(art["config_limpieza"])            # umbrales congelados
```

### 9.4.3 Comprobar que artefacto y código siguen sincronizados

```python
from src import preprocessing as pp
for k, v in art["config_limpieza"].items():
    actual = getattr(pp, k)
    print(f"{k}: artefacto={v}  código={actual}  {'OK' if v == actual else 'DESINCRONIZADO'}")
```

### 9.4.4 Reproducir una decisión concreta

```python
import joblib, pandas as pd
from src import artifacts

nuevo = pd.DataFrame([{...}], index=pd.Index([1], name="row_id"))
art = artifacts.cargar("artifacts/preprocessors/preprocessor_final.joblib")
mod = joblib.load("artifacts/models/xgboost_v1.joblib")

X = artifacts.transformar_datos_nuevos(nuevo, art, rama="trees")
print(mod["modelo"].predict_proba(X[mod["columnas"]])[0, 1])
```

### 9.4.5 Reconstrucción limpia

```bash
rm -rf datasets artifacts outputs src/__pycache__ .pytest_cache
for s in 01 02 03 04 05 06 07 08; do python ${s}_*.py || break; done
```

---

## 9.5 Extensión

### 9.5.1 Añadir una variable derivada

1. Implementarla en `crear_features()` (`src/preprocessing.py`).
2. Si debe entrar en la rama WoE, añadir sus cortes a `BINS_CONTINUAS` o
   `BINS_CONTEO` en `src/woe.py`.
3. Si es un indicador binario, nombrarlo con prefijo `flag_` — se detecta solo.
4. Re-ejecutar `02` en adelante.
5. **Comprobar que no rompe los signos**: `03` fallará el criterio de selección
   si introduce colinealidad.

**Advertencia [RECOM]:** el proyecto demostró dos veces que añadir variables
resta. Las 7 derivadas no aportaron nada y rompieron la interpretabilidad
([E5](05_experimentos.md#e5--selección-de-variables-para-la-regresión-logística)),
y los 7 indicadores de calidad introducían colinealidad perfecta con los bins
`FALTANTE` ([E16](05_experimentos.md#e16--corrección-de-h1-coherencia-completa-de-signos)).
El scorecard final usa **solo las 10 variables originales**.

Toda variable nueva debe justificar su existencia con una mejora **medida** en
validación, y `03_train_logistic.py` la rechazará automáticamente si introduce
un coeficiente que contradiga su dirección univariada.

### 9.5.2 Cambiar los umbrales de limpieza

Editar la constante en `src/preprocessing.py`, **actualizando el comentario que
la justifica**. Re-ejecutar desde `02`.

Los cambios quedan registrados automáticamente en el artefacto
(`config_limpieza`), lo que permite detectar posteriormente con qué reglas se
construyó un modelo dado.

### 9.5.3 Sustituir o añadir un modelo

Crear `0X_train_<modelo>.py` siguiendo el patrón de `04`:

```python
joblib.dump({
    "modelo": modelo,              # debe exponer predict_proba
    "columnas": COLS,              # contrato de entrada
    "hiperparametros": MEJORES,
    "n_arboles": n,                # opcional
    "semilla": SEMILLA,
}, MODELOS / "<nombre>_v1.joblib")
```

Requisitos para integrarse automáticamente:

| Consumidor | Requisito |
|---|---|
| `07_business_evaluation.py` | Claves `modelo` y `columnas`. La rama se infiere: `"logistic" in stem` → `woe`, si no → `trees` |
| `06_evaluation.py` | Registrar el modelo en los diccionarios `RAMA` y `NOMBRE` |
| `predict.py` | Nada; se selecciona con `--modelo <stem>` |

### 9.5.4 Modificar los supuestos económicos

Editar `MARGEN`, `LGD`, `PRESTAMO` en `07_business_evaluation.py` (y
`MARGEN`/`PERDIDA` en `06_evaluation.py`, que los usa para `UMBRAL_NEGOCIO`).

Alternativa sin tocar código: `predict.py --margen X --perdida Y`.

**El umbral se recalcula solo** a partir de $M/(M+L)$. No hay ningún valor de
umbral fijado a mano en el código.

### 9.5.5 Reentrenar con datos nuevos

1. Sustituir `cs-training.csv` conservando el esquema de columnas.
2. Re-ejecutar la cadena completa.
3. **Revisar el EDA**: los umbrales de limpieza se derivaron de estos datos
   concretos y pueden no aplicar.
4. **Comparar el nuevo AUC contra el actual** usando la misma validación cruzada.
5. Versionar los artefactos como `_v2` en lugar de sobrescribir `_v1`.

**[RECOM]** No sobrescribir nunca un artefacto de un modelo que haya tomado
decisiones reales. La trazabilidad de "qué modelo decidió el 14 de marzo" exige
conservarlo.

---

## 9.6 Contribución

### 9.6.1 Convenciones

| Aspecto | Convención |
|---|---|
| Idioma | Español en código, comentarios y documentación |
| Constantes | `MAYUSCULAS`, con comentario que justifique el valor |
| Indicadores binarios | Prefijo `flag_` (obligatorio: se detectan por prefijo) |
| Funciones que ven el objetivo | Prefijo `aprender_` |
| Funciones que no lo ven | Prefijo `aplicar_` |
| Rutas | `pathlib.Path`, siempre relativas a `Path(__file__).parent` |
| Semilla | `SEMILLA = 42` en todo el proyecto |

### 9.6.2 Antes de un commit

```bash
pytest tests/ -q                       # 42 pruebas
python 02_preprocessing.py             # 14 verificaciones internas
python -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('*.py')+glob.glob('src/*.py')]"
```

### 9.6.3 Qué NO se versiona

Definido en `.gitignore`: `datasets/`, `artifacts/{models,preprocessors,submissions}`,
`*.png`, `__pycache__/`, `.pytest_cache/`.

**Sí se versiona** `artifacts/metrics/` — son ficheros de texto pequeños que
documentan los resultados publicados y permiten contrastar el README sin
re-ejecutar nada.

---

## 9.7 Consideraciones para producción

**[RECOM]** El proyecto **no está listo para producción bancaria** por las dos
brechas descritas en [§9.0](#-90-antes-de-usar-este-sistema-dos-brechas-abiertas)
(H2 y H3). Además de cerrarlas, un despliegue real requeriría:

| Componente | Estado | Nota |
|---|---|---|
| Serialización portable (ONNX/PMML) | ❌ | `joblib` acopla a la versión de Python y scikit-learn |
| API de inferencia | ❌ | `predict.py` es CLI por lotes |
| Registro de decisiones | ❌ | Necesario para auditoría y disputas |
| Monitoreo (PSI/CSI/calibración) | ❌ | **Bloqueante** — ver [§9.0](#-90-antes-de-usar-este-sistema-dos-brechas-abiertas) |
| Validación *out-of-time* | ❌ | **Bloqueante** — ver [§9.0](#-90-antes-de-usar-este-sistema-dos-brechas-abiertas) |
| Pruebas A/B o *champion–challenger* | ❌ | |
| Plan de reversión | ❌ | |
| Validación independiente (3.ª línea) | ❌ | |
| Registro de incidencias del modelo | ❌ | Ver observación final de [§7.5](07_auditoria.md#75-opinión-de-validación) |

**Advertencia sobre `joblib` [RECOM]:** un artefacto serializado con
`scikit-learn 1.7.2` puede no cargar con una versión distinta. Para producción,
exportar a ONNX o PMML, que son formatos independientes de la implementación.

---

[← Comparación metodológica](08_comparacion_metodologica.md) · [Índice](README.md) · [Siguiente: Referencias →](10_referencias.md)

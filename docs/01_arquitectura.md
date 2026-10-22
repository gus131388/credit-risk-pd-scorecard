# 01 — Arquitectura del sistema

[← Índice](README.md) · [Siguiente: Fundamentos matemáticos →](02_fundamentos_matematicos.md)

---

## 1.1 Principio rector

El repositorio sigue una separación estricta entre **orquestadores** y
**biblioteca**:

```
scripts numerados (01–08, predict)   →  coordinan, imprimen, guardan
src/*.py                             →  contienen la lógica, no se ejecutan solos
```

**Por qué [RECOM]:** un script procedural no puede importarse sin ejecutarse
entero, luego no puede probarse. Toda lógica que decida algo relevante —cómo se
limpia un dato, si un coeficiente es aceptable, dónde va el umbral— vive en
`src/` y está cubierta por pruebas. Los scripts solo encadenan llamadas.

Esta separación se introdujo tarde en el desarrollo (ver
[capítulo 05](05_experimentos.md)) precisamente porque su ausencia impedía
escribir pruebas para los scripts de modelado.

---

## 1.2 Árbol del repositorio

```
Probability_of_Default/
│
├── 01_eda.py                    Análisis exploratorio (solo lectura)
├── 02_preprocessing.py          Split único → dos ramas → artefactos
├── 03_train_logistic.py         Regresión logística sobre WoE
├── 04_train_xgboost.py          XGBoost sobre valores crudos
├── 05_train_lightgbm.py         LightGBM + comparación CV de los tres
├── 06_evaluation.py             Elección final + SHAP + submission
├── 07_business_evaluation.py    Traducción a dólares y sensibilidad
├── 08_scorecard.py              Tabla de puntos
├── predict.py                   Inferencia sobre solicitantes nuevos
│
├── src/
│   ├── __init__.py              (vacío — marca el paquete)
│   ├── preprocessing.py         Limpieza por reglas + ingeniería de variables
│   ├── woe.py                   Weight of Evidence e Information Value
│   ├── evaluation.py            Métricas, calibración, regla de negocio
│   └── artifacts.py             Persistencia y aplicación de transformadores
│
├── tests/
│   ├── test_preprocessing.py    19 pruebas de invariantes del pipeline
│   └── test_evaluation.py       29 pruebas de métricas y regla de negocio
│
├── docs/                        Este manual
│
├── datasets/processed/          [derivado, ignorado por git]
│   ├── trees/{train,val,test}.parquet
│   └── woe/{train,val,test}.parquet
│
├── artifacts/
│   ├── preprocessors/           [derivado] preprocessor_v1, preprocessor_final
│   ├── models/                  [derivado] logistic_v1, xgboost_v1, lightgbm_v1
│   ├── metrics/                 [VERSIONADO] resultados publicados
│   ├── plots/                   [derivado] gráficas de evaluación
│   └── submissions/             [derivado] predicciones sobre cs-test.csv
│
├── outputs/                     [derivado] gráficas del EDA
├── requirements.txt
├── .gitignore
└── README.md
```

**Datos fuente:** viven fuera del proyecto, en `../../data/`
(`cs-training.csv`, `cs-test.csv`, `sampleEntry.csv`, `Data Dictionary.xls`).
No se duplican dentro del proyecto para evitar ambigüedad sobre cuál es la
fuente canónica.

### Por qué `artifacts/metrics/` sí se versiona

Es texto plano (JSON y CSV, ~40 KB) y documenta los resultados publicados. Si el
README afirma un AUC de 0.86628, ese número debe poder contrastarse contra un
fichero del repositorio sin re-ejecutar nada. El resto de `artifacts/` son
binarios regenerables y quedan fuera.

---

## 1.3 Grafo de dependencias y orden de ejecución

```
                        ../../data/*.csv
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
        ▼                     ▼                      ▼
   01_eda.py           02_preprocessing.py      (06 lee cs-test
   (independiente)             │                 directamente para
        │                      │                 el reentrenamiento)
        ▼                      ▼
    outputs/*.png    ┌─── datasets/processed/ ───┐
                     │                            │
                     ▼                            ▼
              woe/{train,val,test}         trees/{train,val,test}
                     │                            │
                     ▼                     ┌──────┴──────┐
            03_train_logistic.py            ▼             ▼
                     │              04_train_xgboost  05_train_lightgbm
                     │                      │             │
                     └──────────┬───────────┴─────────────┘
                                ▼
                        artifacts/models/
                                │
                ┌───────────────┼────────────────┐
                ▼               ▼                ▼
        06_evaluation.py  07_business_...  08_scorecard.py
                │                                │
                ▼                                ▼
    submissions/ + preprocessor_final    scorecard.csv + bandas
                │
                ▼
           predict.py  (inferencia; también usa preprocessors/)
```

**Regla operativa [IMPL]:** modificar un nodo obliga a re-ejecutar todo lo que
esté aguas abajo.

| Si se modifica… | Re-ejecutar |
|---|---|
| `src/preprocessing.py` o `src/woe.py` | 02 → 03 → 04 → 05 → 06 → 07 → 08 |
| `src/evaluation.py` | 03, 06, 07, 08 (y `pytest`) |
| `03_train_logistic.py` | 03 → 05 (usa el modelo para la CV) → 06 → 07 → 08 |
| `04` o `05` | ese script → 06 → 07 |
| Supuestos económicos (`MARGEN`, `LGD`) | 06 (usa `UMBRAL_NEGOCIO`), 07, 08 |
| `predict.py` | nada (es hoja del grafo) |

**Nota de acoplamiento [IMPL]:** `05_train_lightgbm.py` carga
`artifacts/models/logistic_v1.joblib` y `xgboost_v1.joblib` para construir la
comparación por validación cruzada de los tres modelos. Es un acoplamiento
deliberado —permite usar particiones idénticas para los tres— pero implica que
`05` no puede ejecutarse antes que `03` y `04`.

---

## 1.4 Módulos de la biblioteca (`src/`)

### 1.4.1 `src/preprocessing.py` — 245 líneas

Limpieza determinista por reglas e ingeniería de variables. **No aprende nada de
los datos**, luego puede aplicarse a cualquier conjunto sin riesgo de fuga.

#### Constantes

| Constante | Valor | Significado | Origen |
|---|---|---|---|
| `TARGET` | `"SeriousDlqin2yrs"` | Nombre de la variable objetivo | Dataset |
| `SENTINELAS` | `[96, 98]` | Códigos de "dato no disponible" en las variables de morosidad | EDA |
| `COLS_MORA` | 3 nombres | Las tres variables de conteo de atrasos | Dataset |
| `REVUTIL_UMBRAL_INVALIDO` | `100.0` | Por encima, `RevolvingUtilization` deja de comportarse como deuda | EDA — ver §1.4.1.1 |
| `EDAD_MINIMA` | `18` | Edad mínima plausible para un solicitante | Dominio |
| `INGRESO_MINIMO_PLAUSIBLE` | `1.0` | Por debajo, el ingreso es un código, no un valor | EDA |

##### 1.4.1.1 Justificación de `REVUTIL_UMBRAL_INVALIDO = 100.0` [MEDIDO]

| Banda | n | Tasa de default | Lectura |
|---|---|---|---|
| 1.0 – 1.2 | 1,797 | 38.23% | señal real (sobregiro) |
| 1.2 – 1.5 | 363 | 46.83% | señal real |
| 1.5 – 2.0 | 188 | 45.21% | señal real |
| 2 – 100 | 120 | 30.83% | señal real, muestra pequeña |
| 100 – 1,000 | 33 | **3.03%** | ruido |
| > 1,000 | 139 | **7.19%** | ruido |

Tasa global de referencia: 6.68%. Por encima de 100 el riesgo deja de
distinguirse del azar: se hunde a 3.03% en una banda y vuelve a 6.68% en la
otra. Ninguna de las dos es compatible con "deuda extrema", que debería dar
tasas del orden del 40% como las bandas inferiores. Indica corrupción de dato.

**Contrafáctico crítico [MEDIDO]:** fijar el umbral en 1.0 —la elección
intuitiva, ya que es un ratio— habría invalidado 2,640 filas con **37.50% de
default**, el grupo más predictivo del conjunto. La intuición matemática ("un
ratio no puede superar 1") habría destruido la señal más fuerte disponible.

#### Funciones

##### `limpiar(df) -> DataFrame`

Aplica las reglas de calidad de dato y crea los indicadores. **Nunca elimina
filas.**

| Paso | Acción | Justificación |
|---|---|---|
| 1a | `flag_income_faltante`, `flag_dependents_faltante` | El faltante es predictivo: 5.57% vs 6.96% |
| 1a' | `flag_income_cero`; `MonthlyIncome ≤ 1 → NaN` | Ingreso 0/1 es código, no valor (3.68% vs 6.73%) |
| 1b | `flag_registro_invalido`; `96/98 → NaN` en `COLS_MORA` | 214 filas con 54.67% de default |
| 1c | `flag_revutil_invalido`; `> 100 → NaN` | Ver §1.4.1.1 |
| 1d | Separa `deuda_bruta_proxy` y `DebtRatio_limpio` según exista denominador | `DebtRatio` es un ratio solo si había ingreso (§3.3.2) |
| 1e | `flag_edad_invalida`; `age < 18 → NaN` | 1 fila |

**Decisión de diseño clave [IMPL]:** los valores corruptos se convierten en
`NaN` y se marcan con un indicador; **no se eliminan filas ni se capean**.

- *No eliminar*: en `cs-test.csv` hay que entregar una predicción para las
  101,503 personas. Un pipeline que dependa de descartar filas se rompe en
  inferencia.
- *No capear*: capear coloca al grupo corrupto en el extremo de riesgo máximo.
  Como se demostró, esos grupos tienen riesgo **por debajo** de la media, luego
  capear invierte la señal.

##### `crear_features(df) -> DataFrame`

Construye 7 variables derivadas con lógica de negocio:

| Variable | Fórmula | Pregunta que responde |
|---|---|---|
| `total_atrasos` | $\sum$ de las 3 variables de mora | ¿Cuántos episodios acumula? |
| `flag_algun_atraso` | $\mathbb{1}\{\text{total} > 0\}$ | ¿Tiene alguno? |
| `peor_atraso` | 0/1/2/3 según gravedad máxima | ¿Cuál fue el peor? |
| `ingreso_por_persona` | $\dfrac{\text{MonthlyIncome}}{\text{Dependents} + 1}$ | Capacidad real por miembro del hogar |
| `deuda_mensual_abs` | $\text{DebtRatio\_limpio} \times \text{MonthlyIncome}$ | Deuda en unidades monetarias |
| `lineas_no_inmobiliarias` | $\max(\text{OpenLines} - \text{RealEstate},\, 0)$ | Crédito no garantizado |
| `prop_inmobiliaria` | $\dfrac{\text{RealEstate}}{\text{OpenLines}}$ | Fracción garantizada |

**Corrección documentada [IMPL]** — `peor_atraso` contiene una salvaguarda
explícita:

```python
sin_historial = df[COLS_MORA].isna().all(axis=1)
df.loc[sin_historial, "peor_atraso"] = np.nan
```

Sin ella, las 214 filas con centinela caían en `peor_atraso = 0` ("sin
atrasos") porque `NaN > 0` evalúa a `False`. Esas filas tienen 54.67% de default
y contaminaban el grupo más seguro (2.74%). El bug existió y fue corregido; la
prueba `test_regresion_peor_atraso_no_traga_centinelas` impide su reaparición.

##### `preparar_arboles(df) -> DataFrame`

Elimina `DebtRatio` (sustituida por `DebtRatio_limpio` + `deuda_bruta_proxy` +
indicador) y devuelve el conjunto para modelos de árbol. **No capea ni imputa.**

**Fundamento [TEORÍA]:** los árboles de decisión son invariantes a
transformaciones monótonas de las variables. Si $g$ es estrictamente creciente,
un corte en $x > c$ separa exactamente las mismas observaciones que $g(x) >
g(c)$. Capear introduce empates por encima del tope, lo que **reduce** el
espacio de cortes posibles sin aportar nada.

---

### 1.4.2 `src/woe.py` — 199 líneas

Codificación Weight of Evidence e Information Value. Solo la usa la rama de
regresión logística.

#### Estructuras de configuración

| Objeto | Contenido | Nota |
|---|---|---|
| `BINS_CONTEO` | 9 variables con cortes manuales | Los cuantiles fallan: >94% de ceros en algunas |
| `BINS_CONTINUAS` | 8 variables con cortes derivados del EDA | Los cortes se eligieron donde cambia el riesgo, no por percentiles |
| `BINS_TODOS` | Unión de ambos | |
| `PREFIJO_FLAG` | `"flag_"` | |

**Decisión de diseño [IMPL]:** las columnas indicador se detectan por prefijo,
no por lista explícita:

```python
def columnas_flag(df):
    return [c for c in df.columns if c.startswith(PREFIJO_FLAG)]
```

Motivo: una lista escrita a mano en `woe.py` mientras los indicadores se crean
en `preprocessing.py` es un acoplamiento frágil. Añadir un indicador nuevo y
olvidar registrarlo lo dejaría fuera del conjunto **sin ningún error visible**.
La prueba `test_regresion_woe_registra_flags_nuevos` verifica el mecanismo.

#### Funciones

| Función | Mira el objetivo | Uso |
|---|---|---|
| `_asignar_bins(serie, cortes)` | No | Interna; `NaN → "FALTANTE"` |
| `aprender_woe(df_train, y_train, suavizado=0.5)` | **Sí** | **Solo con entrenamiento** |
| `calcular_iv(tablas_woe)` | No (usa lo aprendido) | Diagnóstico |
| `aplicar_woe(df, tablas_woe)` | No | Seguro en validación y test |
| `columnas_flag(df)` | No | Utilidad |

El patrón `aprender_*` / `aplicar_*` es la salvaguarda estructural contra fuga
de datos: la única función que ve `y` se llama exclusivamente con el conjunto de
entrenamiento. La matemática está en el
[capítulo 02](02_fundamentos_matematicos.md#22-weight-of-evidence).

---

### 1.4.3 `src/evaluation.py` — 165 líneas

Métricas y regla de negocio. Extraído de los scripts para hacerlo comprobable.

| Función | Devuelve | Probada por |
|---|---|---|
| `ks_statistic(y, p)` | `(ks, umbral)` | 2 pruebas |
| `gini(auc)` | $2\,\text{AUC} - 1$ | 1 prueba |
| `tabla_calibracion(y, p, n_grupos=10)` | DataFrame por decil | 1 prueba |
| `error_calibracion(y, p, n_grupos=10)` | Error medio en puntos porcentuales | 2 pruebas |
| `signos_invertidos(coefs, cols, prefijo_woe="woe_")` | Variables WoE con signo incorrecto (versión débil, conservada por compatibilidad) | 3 pruebas |
| `direccion_univariada(X, y, columna)` | +1 / −1 / 0 según cómo mueve el riesgo la variable, medido sin modelo | 2 pruebas |
| `signos_incoherentes(coefs, cols, X_train, y_train)` | **Comprobación completa**: variables WoE por convención de signo e indicadores contra su dirección univariada | 4 pruebas |
| `umbral_optimo(margen, perdida)` | $M/(M+L)$ | 4 pruebas |
| `beneficio(p, y, umbral, margen, perdida)` | dict con conteos y beneficio | 4 pruebas |
| `beneficio_sin_modelo(y, margen, perdida)` | Referencia sin modelo | 1 prueba |
| `validar_submission(sub, n_esperado, columnas)` | Lista `(descripción, ok)` | 6 pruebas |

**Historia de `signos_invertidos()` [IMPL]:** esta función solo evalúa variables
con prefijo `woe_` y exime a los indicadores por considerarlos de dirección
libre. Esa exención **ocultó un defecto real**
([H1](07_auditoria.md#h1)): tres indicadores llegaron al modelo con el signo
contrario a su comportamiento univariado.

Se sustituyó por `signos_incoherentes()`, que es **estrictamente más fuerte** —
una prueba lo verifica explícitamente. La función original se conserva porque
sus pruebas documentan la convención de signo del WoE.

---

### 1.4.4 `src/artifacts.py` — 136 líneas

Persistencia del transformador entrenado.

| Función | Propósito |
|---|---|
| `construir_artefacto(...)` | Empaqueta tablas WoE, cortes, umbrales, contrato de columnas y metadatos |
| `guardar(artefacto, ruta)` | Serializa con `joblib` |
| `cargar(ruta)` | Deserializa |
| `transformar_datos_nuevos(df_crudo, artefacto, rama)` | **La función de producción**: aplica lo congelado |
| `resumen(artefacto)` | Descripción legible para inspección |

#### Contenido del artefacto

```python
{
  "version": "1.0",
  "creado": "2026-08-01T01:59:23",
  "tablas_woe": {17 DataFrames},          # lo aprendido de los datos
  "config_limpieza": {5 umbrales},        # las reglas vigentes al entrenar
  "bins": {17 listas de cortes},
  "columnas_trees": [25 nombres],         # contrato de salida
  "columnas_woe": [25 nombres],
  "metadatos": {n_filas_train, tasa_default_train, semilla},
}
```

**Por qué guardar el transformador y no solo los datos [RECOM]:** un
`.parquet` es el *resultado* de una transformación; el artefacto es la
transformación. Para puntuar a un solicitante mañana hay que aplicarle los
mismos cortes y los mismos valores WoE del día del entrenamiento. Sin el
artefacto habría que re-ejecutar todo el pipeline, y bastaría que alguien
hubiera modificado una constante para obtener una PD distinta sin advertirlo.

`transformar_datos_nuevos()` falla ruidosamente si el código actual no genera
las columnas que el artefacto espera:

```python
if faltantes:
    raise ValueError(
        f"El artefacto espera columnas que el codigo actual no genera: "
        f"{faltantes}. El pipeline cambio desde que se entreno el modelo.")
```

Esto convierte una desincronización silenciosa en un error explícito, y está
cubierto por `test_artefacto_detecta_pipeline_desincronizado`.

---

## 1.5 Scripts orquestadores

### `01_eda.py` — 421 líneas

Ocho secciones: carga, objetivo, descriptivos, faltantes, valores sospechosos,
relación variable–objetivo por deciles, correlaciones, gráficas.

**Principio de diseño [RECOM]:** las gráficas usan datos **crudos y sin
recortar**, con eje $y$ logarítmico. Un histograma recortado al percentil 99 se
ve mejor pero oculta exactamente lo que el EDA debe encontrar. Las bimodalidades
de `RevolvingUtilization` y `DebtRatio` —dos poblaciones separadas por un valle
vacío— solo son visibles con el rango completo.

Salidas: `outputs/01_balance_target.png` … `05_colas_completas.png`.

### `02_preprocessing.py` — 272 líneas

Orquestador del preprocesamiento. Constantes: `SEMILLA = 42`,
`PCT_VALIDACION = 0.20`, `ID = "row_id"`.

Siete pasos: carga → split estratificado único → limpieza + variables → rama
árboles → rama WoE → persistencia del artefacto → **14 verificaciones**.

Las verificaciones incluyen una prueba de ida y vuelta: recargar el artefacto
desde disco y aplicarlo a datos crudos debe reproducir exactamente el
`.parquet` guardado.

### `03_train_logistic.py` — 528 líneas

Nueve pasos: carga → VIF → selección de subconjunto → modelo final →
evaluación → coeficientes → **diagnósticos** → gráficas → guardado.

Sub-partición interna: 96,000 ajuste / 24,000 selección. La validación no
participa en ninguna decisión.

### `04_train_xgboost.py` — 442 líneas

Diez pasos, incluidos *early stopping* con conjunto propio y búsqueda sobre
11 configuraciones.

### `05_train_lightgbm.py` — 421 líneas

Ocho pasos. Además de entrenar, ejecuta la **comparación por validación cruzada
de los tres modelos** con particiones idénticas y contraste emparejado.

### `06_evaluation.py` — 494 líneas

Comparación multi-criterio → SHAP → elección → reentrenamiento con 150,000
filas → submission → 8 verificaciones.

### `07_business_evaluation.py` — 249 líneas

Traducción a unidades monetarias y análisis de sensibilidad al ratio
pérdida/ganancia. Detecta automáticamente todos los modelos en
`artifacts/models/`.

### `08_scorecard.py` — 335 líneas

Construcción de la tabla de puntos, verificación de reconciliación exacta,
explicación individual y bandas de riesgo.

### `predict.py` — 189 líneas

Interfaz de línea de comandos para inferencia. Único script con `argparse`.

```bash
python predict.py --demo
python predict.py --entrada nuevos.csv --modelo logistic_v1 --salida out.csv
python predict.py --entrada nuevos.csv --margen 800 --perdida 20000
```

Prefiere `preprocessor_final.joblib` (emparejado con el modelo reentrenado) y
recurre a `preprocessor_v1.joblib` si aquel no existe.

---

## 1.6 Dependencias

```
pandas 2.3.3, numpy 2.3.5, scipy 1.16.3, pyarrow 21.0.0
scikit-learn 1.7.2, xgboost 3.3.0, lightgbm 4.7.0
matplotlib 3.10.6, seaborn 0.13.2
joblib 1.5.2, shap 0.52.0, pytest 8.4.2
```

`shap` se importa dentro de un `try/except` en `06_evaluation.py`: si no está
instalado, el script omite ese paso y continúa. Es la única dependencia opcional.

---

## 1.7 Decisiones arquitectónicas y sus alternativas

| Decisión | Alternativa habitual | Por qué la elegida |
|---|---|---|
| Scripts numerados | Notebooks Jupyter | Reproducibles, versionables, sin estado oculto de ejecución |
| Constantes en el módulo | `config.yaml` | La constante conserva su justificación al lado; el YAML la separa de su razón |
| `joblib` | `pickle`, ONNX, PMML | `joblib` es el estándar de scikit-learn; ONNX/PMML serían preferibles para despliegue multiplataforma [RECOM] |
| Parquet | CSV | Preserva tipos y `NaN`; ~10× menor |
| Funciones con dict de parámetros | Clases `fit`/`transform` de sklearn | Hace explícito y visible *qué* se aprendió; sklearn sería preferible para integrarse en un `Pipeline` [RECOM] |
| Dos ramas de datos | Un único conjunto | Cada familia de modelos exige un tratamiento distinto e incompatible |

---

[← Índice](README.md) · [Siguiente: Fundamentos matemáticos →](02_fundamentos_matematicos.md)

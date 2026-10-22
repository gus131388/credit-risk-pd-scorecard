# 07 — Auditoría técnica independiente

[← Negocio y regulación](06_negocio_y_regulacion.md) · [Índice](README.md) · [Siguiente: Comparación metodológica →](08_comparacion_metodologica.md)

---

## 7.0 Alcance y método

Esta revisión **no asume que la implementación sea correcta**. Cada afirmación
del proyecto se ha contrastado contra el código y contra ejecuciones
independientes.

**Método aplicado:**

1. Reconstrucción desde cero tras eliminar todo artefacto derivado.
2. Contraste byte a byte contra los resultados publicados.
3. Validación cruzada anidada para detectar sesgo por selección.
4. Análisis de coherencia entre coeficientes y comportamiento univariado.
5. Detección estática de código muerto.
6. Verificación de que los números del README existen en los artefactos.

**Convención de gravedad:**

| Nivel | Significado |
|---|---|
| 🔴 **Crítico** | Impide el despliegue o invalida una conclusión publicada |
| 🟠 **Mayor** | Compromete interpretabilidad, gobierno o mantenibilidad |
| 🟡 **Menor** | Deuda técnica sin impacto en los resultados |
| ⚪ **Observación** | Sin defecto; se documenta por transparencia |

---

## 7.1 Resultados que SÍ se sostienen

Antes de los hallazgos, lo verificado como correcto.

### V1 — Reproducibilidad exacta ✅

Eliminados `datasets/`, `artifacts/`, `outputs/` y cachés, y reconstruido:

```
Reconstrucción       : 113 segundos, 8/8 scripts sin error
Métricas             : 12/12 ficheros idénticos byte a byte
Submissions          : 101,503/101,503 predicciones idénticas
Artefactos esperados : 17/17 generados
```

### V2 — Ausencia de fuga de datos en los datos ✅

```
Correlación máxima con el objetivo : 0.3916 (total_atrasos)
Objetivo entre las características : NO
Valores 96/98 residuales           : 0
```

Umbral de alarma convencional: correlación > 0.9. No se alcanza.

### V3 — Ausencia de fuga en el proceso ✅

Auditados los criterios de selección de los tres scripts de entrenamiento:

| Script | Selecciona por | Validación participa |
|---|---|---|
| `03` | `AUC_seleccion` (24,000 filas del train) | No |
| `04` | `auc_es` (24,000 filas del train) | No |
| `05` | `auc_es` (24,000 filas del train) | No |

**Nota histórica:** `03` **sí** usaba `AUC_val` para seleccionar en una versión
anterior. Fue corregido. La versión actual no lo hace.

**Límite de esta verificación:** V2 y V3 examinan los datos y los criterios de
selección automáticos. No examinan el **ámbito del análisis exploratorio**, que
es un canal distinto —las decisiones del analista— y por el que sí había
contaminación. Ver [H13](#h13).

### V4 — El AUC reportado no está inflado ✅

Validación cruzada anidada (rejilla completa dentro de cada fold):

| | Media | Desv. |
|---|---|---|
| CV reportada | 0.86628 | 0.00306 |
| Nested CV | **0.86661** | 0.00337 |

Diferencia −0.00028: por debajo del ruido. **Éste era el único experimento capaz
de invalidar la conclusión central.**

### V5 — Determinismo ✅

Entrenamientos repetidos con la misma semilla producen coeficientes y
predicciones idénticos (`np.array_equal` → `True`).

### V6 — Corrección matemática del scorecard ✅

$$\max_i \left| \text{Score}_i^{\text{suma de puntos}} - \text{Score}_i^{\text{transformación de PD}} \right| = 0.0000000000$$

### V7 — Aditividad de SHAP ✅

$$\phi_0 + \sum_i \phi_i = -2.6612 + 4.7953 = +2.1341 \;\Rightarrow\; \sigma(2.1341) = 89.4\% = \hat p_{\text{modelo}}$$

### V8 — Integridad de artefactos ✅

Los 5 umbrales congelados coinciden con el código actual; el artefacto recargado
reproduce la transformación; `preprocessor_final.joblib` declara correctamente
150,000 filas.

---

## 7.2 Hallazgos

### <a name="h1"></a>H1 ✅ — El scorecard atribuía puntos en dirección contraria al riesgo (CORREGIDO)

> **Estado: cerrado.** Lo que sigue documenta el defecto tal como se detectó, su
> causa, la corrección aplicada y su coste medido. Se conserva completo porque el
> historial de un defecto corregido es parte del expediente del modelo.

#### 1. Qué estaba mal

Tres variables indicador tenían coeficientes cuyo signo **contradecía su
comportamiento univariado**:

| Variable | Evento si = 1 | si = 0 | Dirección real | $\beta$ | Coherencia |
|---|---|---|---|---|---|
| `flag_registro_invalido` | **54.67%** | 6.60% | **más riesgo** | **−0.5852** | ❌ |
| `flag_revutil_invalido` | 6.40% | 6.68% | menos riesgo | +0.1042 | ❌ |
| `flag_debtratio_invalido` | 5.55% | 6.98% | menos riesgo | +0.5563 | ❌ |

El caso más grave era el primero: el grupo con **8.3 veces** el riesgo de la
población recibía un coeficiente que decía "menos riesgo", y por tanto **puntos
positivos** en el scorecard.

#### 2. Por qué estaba mal

El scorecard existe para explicar decisiones. Si un supervisor pregunta por qué
se rechazó a un cliente y la tabla indica que "registro inválido" le **sumó**
puntos, la explicación es indefendible — pese a que el modelo, en conjunto,
predice correctamente.

**Distinción crítica:** la **predicción agregada es correcta**; lo incorrecto es
la **atribución**. Y la atribución es la única razón de ser de un scorecard.

#### 3. Por qué ocurrió

**Colinealidad perfecta con los bins `FALTANTE` del WoE.** Verificado:

| Indicador | Variable WoE asociada | Filas | Valor WoE único | Coincidencia |
|---|---|---|---|---|
| `flag_registro_invalido` | `woe_NumberOfTimes90DaysLate` | 214 | −2.8228 | **100%** |
| `flag_debtratio_invalido` | `woe_DebtRatio_limpio` | 24,718 | +0.1976 | **100%** |
| `flag_revutil_invalido` | `woe_RevolvingUtilizationOfUnsecuredLines` | 172 | +0.0059 | **100%** |

Cuando un valor se convierte en `NaN`, ocurren **dos cosas a la vez**: WoE lo
coloca en el bin `FALTANTE` **y** se enciende el indicador. Son exactamente las
mismas filas contadas dos veces. El estimador reparte el efecto entre ambas
variables de la forma que minimiza la pérdida, sin restricción de signo, y el
indicador acaba actuando como **corrección** del bin, no como señal propia.

**Por qué la comprobación existente no lo detectó:** `signos_invertidos()`
excluye deliberadamente las columnas indicador:

```python
return [x for x in columnas if x.startswith(prefijo_woe) and c[x] > 0]
```

La exención se justificó porque un indicador *puede* legítimamente ir en
cualquier dirección (ej. "sin ingreso reportado" resulta **menos** riesgoso).
Pero al eximirlos por completo, la comprobación quedó ciega a esta clase de
defecto.

#### 4. Corrección aplicada

**Corrección de fondo:** en un scorecard WoE, **los indicadores de calidad de
dato son redundantes por construcción**. El bin `FALTANTE` ya codifica esa
información con su propio nivel de riesgo medido. Se eliminaron de la rama WoE:
el modelo final usa **solo las 10 variables originales**.

**Corrección de la comprobación:** se añadieron a `src/evaluation.py` las
funciones `direccion_univariada()` y `signos_incoherentes()`. La segunda verifica
variables `woe_` contra la convención de signo **e indicadores contra su
dirección univariada medida en los datos**:

```python
def signos_incoherentes(coeficientes, columnas, X_train, y_train,
                        prefijo_woe="woe_"):
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
```

`03_train_logistic.py` usa ahora esta función como criterio de selección.

#### 5. Impacto medido de la corrección [MEDIDO]

Se evaluaron ocho subconjuntos con el criterio completo:

| Subconjunto | n | AUC selección | VIF máx | Incoherentes |
|---|---|---|---|---|
| A: todas | 25 | 0.86135 | 80.5 | 8 |
| E: originales + indicadores (anterior) | 18 | 0.86059 | 8.8 | 3 |
| F: sin indicadores de calidad | 18 | 0.86077 | 58.7 | 5 |
| G: originales + `flag_algun_atraso` | 11 | 0.86006 | 3.6 | 1 |
| **H: solo las 10 originales** | **10** | 0.85749 | **1.3** | **0** ✅ |

**Hallazgo decisivo:** retirar solo los tres indicadores colineales **no basta**.
El problema se **desplaza** a otros (`flag_dependents_faltante`, y después
`woe_NumberOfOpenCreditLinesAndLoans`). Únicamente el subconjunto de las diez
variables originales alcanza coherencia total.

Esto confirma el diagnóstico de fondo: **en un scorecard WoE, todo indicador de
calidad de dato es redundante por construcción**, porque el bin `FALTANTE` ya
codifica esa información con su propio nivel de riesgo medido.

#### 6. Coste real de la corrección [MEDIDO]

| Métrica | Antes (18 vars) | Después (10 vars) | Δ |
|---|---|---|---|
| AUC validación | 0.86272 | 0.86059 | −0.00257 |
| AUC (CV) | 0.85948 | 0.85794 | −0.00187 |
| KS | 0.5760 | 0.5696 | −0.0064 |
| Average Precision | 0.3938 | 0.3869 | −0.0069 |
| Calibración | 0.53 pp | **0.75 pp** | **peor** |
| VIF máximo | 8.8 | **1.3** | mejor |
| Valor de negocio | $175.33 | $171.95 | −$3.38 |
| **Coeficientes incoherentes** | **3** | **0** | ✅ |

**Lectura honesta:** la corrección **empeora las cinco métricas de rendimiento**,
incluida la calibración. No es un caso de "corrección gratis". Se acepta porque
un scorecard cuya tabla de puntos contradice la evidencia no sirve para su único
propósito, por muy alto que sea su AUC.

#### 7. Verificación del cierre

```
Subconjunto elegido : H: solo las 10 originales
Variables           : 10
Incoherencias       : 0
VIF máximo          : 1.3
Coeficientes        : los 10 negativos (convención WoE)
Reconciliación del scorecard : diferencia máxima 0.0000000000
```

**Prevención:** nueva función `signos_incoherentes()` en `src/evaluation.py`,
cubierta por **6 pruebas**, entre ellas
`test_la_comprobacion_completa_es_mas_estricta_que_la_antigua`, que verifica
formalmente que detecta un superconjunto estricto de lo que detectaba la
comprobación anterior.

**Estado:** ✅ **CORREGIDO Y VERIFICADO.**

---

### <a name="h2"></a>H2 🔴 — Ausencia de validación *out-of-time*

#### 1. Qué está mal

Toda la validación es sobre particiones **aleatorias** de la misma población. No
existe evaluación sobre un periodo temporal posterior.

#### 2. Por qué está mal

**[TEORÍA]** BCBS (2005) y la práctica supervisora estándar exigen demostrar que
el modelo mantiene su poder discriminante **en el tiempo**, no solo en una
muestra retenida contemporánea. Los modelos de crédito se degradan por cambios en
el ciclo económico, en la política de originación y en la composición de la
demanda.

Una partición aleatoria mide **estabilidad muestral**, no **estabilidad
temporal**. Son propiedades distintas: un modelo puede tener AUC 0.87 en
validación aleatoria y 0.72 doce meses después.

#### 3. Por qué ocurrió

**Limitación estructural del conjunto de datos.** `cs-training.csv` no contiene
ninguna marca temporal: ni fecha de originación, ni de solicitud, ni cohorte. Es
imposible construir una partición temporal.

#### 4. Cómo se corrige

No es corregible con estos datos. Requiere:

- Una variable de fecha en el conjunto de originación.
- Partición por cohortes: entrenar con originaciones hasta $T$, validar con
  $T+1, T+2, \dots$
- Curvas de AUC/KS por cohorte para detectar degradación.

#### 5. Impacto esperado

**[RECOM]** Es la brecha más grave desde la óptica supervisora. Su ausencia
**bloquea el despliegue** en un entorno regulado, independientemente de lo bueno
que sea el AUC in-sample.

**Estado:** no corregible en el alcance actual. Documentado como limitación.

---

### <a name="h3"></a>H3 🔴 — Sin marco de monitoreo ni backtesting

#### 1. Qué está mal

No existe procedimiento definido para detectar degradación tras el despliegue.

#### 2. Por qué está mal

Un modelo de crédito en producción requiere vigilancia continua de: estabilidad
de la población (PSI), estabilidad de las características (CSI), calibración
(PD predicha vs observada por cohorte) y discriminación (AUC móvil).

#### 3. Por qué ocurrió

El PSI se evaluó y se descartó **con justificación correcta** (E14): con
particiones aleatorias de la misma población, el PSI máximo es 0.00073 frente a
un umbral de alerta de 0.10. No hay deriva que medir.

**El descarte fue correcto para este conjunto**; la ausencia de un *marco* de
monitoreo es lo que constituye la brecha.

#### 4. Cómo se corrige

Definir e implementar:

| Indicador | Frecuencia | Umbral de acción |
|---|---|---|
| PSI del score | Mensual | > 0.10 vigilar, > 0.25 recalibrar |
| CSI por variable | Mensual | > 0.10 investigar |
| Calibración por cohorte | Trimestral | Desviación > 2 pp |
| AUC móvil 12 meses | Trimestral | Caída > 0.03 |
| Tasa de aprobación | Semanal | Desviación > 5 pp del plan |

#### 5. Impacto esperado

Sin monitoreo, la degradación se detecta cuando ya ha generado pérdidas.

**Estado:** no implementado. Requiere datos con dimensión temporal.

---

### <a name="h4"></a>H4 ✅ — La comprobación de signos tenía un punto ciego (CORREGIDO)

`signos_invertidos()` eximía a los indicadores, lo que ocultó los tres defectos
de H1.

**Corrección aplicada:** nuevas funciones `direccion_univariada()` y
`signos_incoherentes()` en `src/evaluation.py`. La segunda verifica variables
`woe_` contra la convención de signo **e indicadores contra su dirección
univariada medida en los datos**.

`signos_invertidos()` se conserva —sus pruebas documentan la convención WoE— pero
`03_train_logistic.py` ya no la usa como criterio.

**Estado:** ✅ **CORREGIDO.** 6 pruebas nuevas; total 48.

---

### <a name="h5"></a>H5 🟡 — Suavizado WoE no canónico

#### Qué está mal

```python
g["dist_eventos"]    = (g["eventos"]    + suavizado) / total_eventos
g["dist_no_eventos"] = (g["no_eventos"] + suavizado) / total_no_eventos
```

Se suma $\alpha$ al numerador pero **no se corrige el denominador**. La forma
canónica sería $(n_{1j}+\alpha)/(N_1 + J\alpha)$.

#### Por qué importa poco

$$\sum_j \hat P_j = 1 + \frac{J\alpha}{N} \approx 1 + \frac{10 \times 0.5}{10^5} = 1 + 5\times10^{-5}$$

Sesgo del orden de $10^{-5}$ en las distribuciones y de $\sim10^{-5}$ en el WoE.

#### Corrección

```python
g["dist_eventos"] = (g["eventos"] + suavizado) / (total_eventos + len(g) * suavizado)
```

**Impacto esperado:** cambios en el quinto decimal del WoE. No altera ninguna
conclusión.

**Estado:** documentado, no corregido (el cambio invalidaría la reproducibilidad
byte a byte sin beneficio material).

---

### <a name="h6"></a>H6 🟠 — Binning manual, no reproducible por un tercero

#### Qué está mal

Los 17 conjuntos de cortes en `BINS_CONTEO` y `BINS_CONTINUAS` fueron elegidos
por inspección visual del EDA.

#### Por qué está mal

Otro analista partiendo de los mismos datos obtendría cortes distintos. En un
entorno regulado, un componente central del modelo debe ser derivable mediante un
procedimiento documentado, no por criterio individual.

#### Corrección

Contrastar contra un binning monótono automático (`optbinning`, árbol de
decisión con restricción de monotonía, o ChiMerge) y documentar la diferencia.

**Impacto esperado [RECOM]:** probablemente pequeño en AUC —los cortes manuales
siguen la evidencia empírica— pero convierte una decisión subjetiva en un
procedimiento auditable.

**Estado:** no corregido. Ver [capítulo 08](08_comparacion_metodologica.md) para
la comparación de métodos.

---

### <a name="h7"></a>H7 🟡 — Los scripts de modelado carecen de pruebas unitarias

Las 42 pruebas cubren `src/`. Los scripts `01`–`08` y `predict.py` no tienen
pruebas propias.

**Mitigación existente:** cada script ejecuta verificaciones internas (14 en el
`02`, 8 en el `06`, 1 exacta en el `08`) y falla con código de salida distinto de
cero.

**Corrección propuesta:** pruebas de humo que ejecuten cada script sobre una
muestra reducida y verifiquen los artefactos de salida.

**Estado:** parcialmente mitigado.

---

### <a name="h8"></a>H8 🟡 — Siete registros con edad > 100 años

`EDAD_MINIMA = 18` filtra por abajo; no hay filtro superior. Quedan 7 registros
con edad entre 101 y 109, con tasa de evento del 14.29% (1 de 7 — ruido).

**Valoración:** una edad de 103 años no es imposible. Su comportamiento es
coherente con el resto (no presenta la inversión característica de los códigos de
sistema).

**Estado:** documentado, decisión de no actuar justificada.

---

### <a name="h9"></a>H9 🟠 — LGD y EAD tratados como constantes

El cálculo económico asume $\text{LGD} = 0.60$ y $\text{EAD} = \$10{,}000$
uniformes.

**Por qué es una simplificación relevante:** LGD y EAD son variables aleatorias
que se modelan por separado en un sistema IRB completo. LGD depende de garantías,
antigüedad y agresividad de recobro; EAD del uso de líneas comprometidas.

**Consecuencia:** el valor unitario de $180.37 es indicativo, no una estimación
de ECL auditada.

**Estado:** documentado. Fuera del alcance del proyecto.

---

### <a name="h10"></a>H10 🟠 — Reject inference no abordado

Ver [§6.7.3](06_negocio_y_regulacion.md#673-reject-inference).

**Estado:** inobservable con este conjunto. Documentado.

---

### <a name="h11"></a>H11 🟡 — Código muerto (corregido durante esta auditoría)

Detección estática reveló en `06_evaluation.py`:

```python
art_e = modelos[ELEGIDO] if 'ELEGIDO' in dir() else None   # nunca usado
```

**Corregido durante esta revisión**: sustituido por un comentario explicativo.

Permanecen dos asignaciones `fig = plt.subplots(...)` cuyo objeto no se usa
directamente (se opera sobre `axes`). Es idiomático en matplotlib y no constituye
defecto.

---

### <a name="h12"></a>H12 ⚪ — El objetivo es un proxy del default regulatorio

`SeriousDlqin2yrs` implementa el criterio de 90+ días de Basilea, pero **no** el
de *unlikeliness to pay*. Además su horizonte (24 meses) difiere del horizonte de
12 meses de la PD de capital.

**No es un defecto del proyecto** —es la definición del conjunto de datos— pero
debe declararse en cualquier uso regulatorio.

---

### <a name="h13"></a>H13 ✅ — El EDA se ejecutaba sobre el fichero completo (CORREGIDO)

**Detectado por el autor del proyecto, no por esta auditoría.** El punto V2/V3
verificaba la ausencia de fuga en los datos y en los criterios de selección, pero
no examinó el ámbito del análisis exploratorio.

#### 1. Qué estaba mal

`01_eda.py` leía las 150,000 filas de `cs-training.csv` y calculaba sobre ellas
todos los diagnósticos, incluidos los que involucran el objetivo: tasa de evento
por decil, por patrón de faltantes, y por banda de las variables recortadas.

Las decisiones de diseño derivadas de ese análisis —`SENTINELAS`,
`REVUTIL_UMBRAL_INVALIDO`, `DEBTRATIO_UMBRAL_INVALIDO`
(hoy sustituido, ver [H14](#h14)), `INGRESO_MINIMO_PLAUSIBLE`, y la creación de los indicadores de faltante— se
tomaron, por tanto, viendo también las 30,000 filas reservadas para validación.

#### 2. Por qué estaba mal

No es fuga por ajuste: el EDA no estima ningún parámetro que entre en el modelo.
Es **fuga del analista**: las filas de validación participaron en el diseño y
después se usaron para juzgarlo. Ninguna estructura de carpetas ni ningún
`train_test_split` detecta esta forma de contaminación, porque no pasa por
ninguna llamada a `.fit()`.

#### 3. Corrección aplicada

`01_eda.py` ejecuta ahora el split en la PARTE 1, antes de calcular nada, con la
misma semilla (42), el mismo `test_size` (0.20) y el mismo `stratify` que
`02_preprocessing.py`. Se verificó que ambas particiones son idénticas fila a
fila. Todo el análisis opera sobre las 120,000 de entrenamiento.

#### 4. Impacto medido [MEDIDO]

**Ningún umbral cambió.** Los cuatro se recalcularon sobre la partición de
entrenamiento y las rupturas de régimen caen en el mismo sitio:

| Umbral | Evidencia sobre 150,000 | Evidencia sobre 120,000 | ¿Cambia? |
|---|---|---|---|
| `SENTINELAS = [96, 98]` | 269 filas, 54.65% vs 6.60% | 214 filas, 54.67% vs 6.60% | No |
| `REVUTIL = 100` | señal hasta 100, ruido encima | señal 39.67% hasta 100; 3.03% y 7.19% encima | No |
| `DEBTRATIO = 2` | 6.03 → 9.83 → 13.17%, luego rompe | 6.05 → 9.88 → 13.12%, luego rompe | No |
| `INGRESO_MÍNIMO = 1` | ceros a 4.04% vs decil pobre 9.67% | ceros a 4.16% vs decil pobre 8.35% | No |

Los modelos entrenados son **bit a bit idénticos**: `01` no produce artefactos de
datos, solo gráficas y diagnóstico. El conocimiento viaja al pipeline a través de
constantes escritas a mano, y ninguna de ellas cambió.

#### 5. Valor de la corrección

Nulo en métricas, relevante en método. Lo que se gana es que la afirmación "las
30,000 filas de validación nunca intervinieron en el diseño" pasa de ser
aproximadamente cierta a ser **verificable**, y que el orden de ejecución del
proyecto refleja la disciplina que documenta.

**Estado:** corregido. Cifras de los capítulos 02, 03, 05 y del README
actualizadas a la partición de entrenamiento.

---

### <a name="h14"></a>H14 ✅ — `DebtRatio` se separaba por magnitud, no por causa (CORREGIDO)

**Detectado por el autor del proyecto** al exigir evidencia propia para el
tratamiento de `MonthlyIncome ≤ 1`.

#### 1. Qué estaba mal

La separación de `DebtRatio` en ratio e importe usaba el umbral
`DEBTRATIO_UMBRAL_INVALIDO = 2.0`. Era un **proxy**: se apoyaba en que los
importes son numéricamente grandes, no en la razón por la que existen.

#### 2. Por qué estaba mal

El proxy confunde dos poblaciones. De las filas con `DebtRatio > 2`, **955
tienen ingreso válido**: son ratios auténticos de personas con ingreso bajo, y
predicen **por encima** de la media (7.24%–9.43% frente al 6.68% global). El
umbral las descartaba como dato corrupto y volcaba su valor en `deuda_bruta_proxy`,
donde el modelo no podía leerlas como apalancamiento.

#### 3. Corrección aplicada

El criterio pasa a ser la existencia de denominador, que es la causa real
(§[3.3.2](03_pipeline_datos.md#332-de-un-umbral-de-magnitud-al-criterio-del-denominador-medido)):

```python
sin_denominador = (df["flag_income_faltante"] | df["flag_income_cero"]).astype(bool)
```

`flag_debtratio_invalido` **se elimina**: bajo la regla nueva sería idéntico a
la unión de los dos indicadores de ingreso — colinealidad perfecta, el mismo
mecanismo de [H1](#h1). Las características bajan de 25 a 24.

#### 4. Impacto medido [MEDIDO]

| Modelo | AUC CV antes | AUC CV después | KS | Calibración |
|---|---|---|---|---|
| XGBoost | 0.86633 | **0.86628** | 0.5865 → 0.5832 | 0.34 → 0.35 pp |
| LightGBM | 0.86588 | **0.86619** | 0.5832 → 0.5885 | 0.39 → 0.36 pp |
| LogReg (WoE) | 0.85761 | **0.85794** | 0.5696 → 0.5720 | 0.75 → 0.72 pp |

Todos los movimientos quedan dentro del ruido de muestreo ($\pm 0.003$). El
campeón sigue siendo XGBoost. Dos pruebas nuevas cubren la regla
(`test_debtratio_se_separa_por_denominador_no_por_magnitud`,
`test_no_existe_flag_debtratio_invalido`); la suite pasa de 48 a 50.

#### 5. Valor de la corrección

Neutro en métricas, positivo en método: la regla deja de ser una correlación
observada y pasa a expresar el mecanismo que genera el dato, y desaparece una
redundancia estructural de la familia que ya causó H1.

**Estado:** corregido.

---

## 7.3 Cuadro resumen

| # | Hallazgo | Gravedad | Corregible aquí | Estado |
|---|---|---|---|---|
| H1 | Scorecard con atribución contraria al riesgo | 🔴 | Sí | ✅ **CORREGIDO** |
| H2 | Sin validación out-of-time | 🔴 | **No** (faltan datos) | ⛔ Abierto |
| H3 | Sin marco de monitoreo | 🔴 | **No** (faltan datos) | ⛔ Abierto |
| H4 | Punto ciego en la comprobación de signos | 🟠 | Sí | ✅ **CORREGIDO** |
| H5 | Suavizado WoE no canónico | 🟡 | Sí | Documentado |
| H6 | Binning manual no reproducible | 🟠 | Sí | Documentado |
| H7 | Scripts sin pruebas unitarias | 🟡 | Sí | Mitigado |
| H8 | 7 registros con edad > 100 | 🟡 | Sí | Sin acción (justificado) |
| H9 | LGD/EAD constantes | 🟠 | Fuera de alcance | Documentado |
| H10 | Reject inference | 🟠 | **No** | Documentado |
| H11 | Código muerto | 🟡 | Sí | **Corregido** |
| H12 | Objetivo como proxy | ⚪ | N/A | Declarado |
| H13 | EDA sobre el fichero completo | 🟠 | Sí | ✅ **CORREGIDO** |
| H14 | `DebtRatio` separado por magnitud, no por causa | 🟠 | Sí | ✅ **CORREGIDO** |

---

## 7.4 Ruta de remediación priorizada [RECOM]

| Prioridad | Acción | Esfuerzo | Bloquea despliegue |
|---|---|---|---|
| ~~1~~ | ~~Retirar indicadores de la rama WoE + `signos_incoherentes()` (H1, H4)~~ | ✅ hecho | — |
| **1** | **Obtener datos con dimensión temporal y validar OOT (H2)** | Cambio de alcance | **Sí** |
| **2** | **Definir e implementar marco de monitoreo (H3)** | 1 semana (tras H2) | **Sí** |
| 4 | Binning monótono automático como contraste (H6) | 2–3 días | No |
| 5 | Análisis de impacto dispar (§6.7) | 3–5 días | Depende de jurisdicción |
| 6 | Pruebas de humo para los scripts (H7) | 1 día | No |
| 7 | Suavizado canónico (H5) | 1 hora | No |

---

## 7.5 Opinión de validación

**[RECOM]**

El proyecto es **metodológicamente sólido y técnicamente reproducible**. Las
salvaguardas contra fuga de datos son explícitas y verificadas; las conclusiones
estadísticas resisten validación cruzada anidada; el pipeline se reconstruye byte
a byte.

Sus defectos son de **dos naturalezas distintas**:

- **Corregibles** (H1, H4, H5, H6, H7): errores de implementación o decisiones
  perfectibles. Ninguno invalida los resultados publicados; H1 sí invalida el uso
  del scorecard como herramienta de explicación.
- **Estructurales** (H2, H3, H10): consecuencia directa de las propiedades del
  conjunto de datos. No son subsanables sin cambiar el alcance.

**Conclusión:** apto como investigación aplicada y como base de desarrollo. **No
apto para despliegue en producción bancaria** hasta cerrar **H2 y H3**, ambos
imposibles con este conjunto de datos por carecer de dimensión temporal.

H1 y H4, los únicos defectos críticos subsanables dentro del alcance actual, han
sido **corregidos y verificados**.

**Observación sobre el proceso [RECOM]:** dos de los hallazgos de una auditoría
anterior (fuga de selección en `03`, óptimo empírico en `07`) fueron
identificados y **no corregidos hasta que se solicitó una auditoría explícita**.
Que un defecto conocido persista es más grave que no haberlo detectado: sugiere
que el proyecto carecía de un mecanismo de seguimiento de hallazgos. En un
entorno con gobierno de modelos formal, esto se gestiona con un registro de
incidencias con responsable y fecha de cierre.

---

[← Negocio y regulación](06_negocio_y_regulacion.md) · [Índice](README.md) · [Siguiente: Comparación metodológica →](08_comparacion_metodologica.md)

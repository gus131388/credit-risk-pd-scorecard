# 05 — Registro de experimentos

[← Modelado](04_modelado.md) · [Índice](README.md) · [Siguiente: Negocio y regulación →](06_negocio_y_regulacion.md)

---

Este capítulo documenta **todo experimento ejecutado durante el desarrollo**,
incluidos los que fracasaron y los que corrigieron decisiones previas. Se
incluyen deliberadamente los errores cometidos: en validación de modelos, el
historial de decisiones descartadas es tan informativo como el resultado final.

---

## E1 — Umbral de `RevolvingUtilizationOfUnsecuredLines`

**Objetivo.** Determinar a partir de qué valor la variable deja de representar
deuda real.

**Hipótesis inicial (errónea).** Al ser un ratio, cualquier valor $>1$ es
inválido; capear en 1.

**Método.** Tasa de evento por banda, con la tasa global (6.68%) como referencia.

**Resultado [MEDIDO].**

| Banda | n | Tasa de evento | Lectura |
|---|---|---|---|
| 1.0 – 1.2 | 1,797 | 38.23% | señal |
| 1.2 – 1.5 | 363 | 46.83% | señal |
| 1.5 – 2.0 | 188 | 45.21% | señal |
| 2 – 100 | 148 | 14–57% | señal (muestra pequeña) |
| 100 – 1,000 | 33 | **3.03%** | ruido |
| > 1,000 | 139 | **7.19%** | ruido |

**Sensibilidad del umbral [MEDIDO].**

| Umbral | Filas descartadas | Tasa de evento de las descartadas | Veredicto |
|---|---|---|---|
| **1** | 2,640 | **37.50%** | ❌ destruye la señal más fuerte |
| 2 | 371 | 14.56% | ❌ aún descarta señal |
| 5 | 254 | 7.87% | aceptable |
| 20 | 234 | 5.56% | aceptable |
| **100** | 223 | **4.93%** | ✅ solo ruido |

**Interpretación.** La banda 1.0–2.0 es la **más predictiva del conjunto**
(hasta 46.83%, siete veces la tasa global). Son sobregiros y líneas excedidas
reales. La hipótesis inicial habría invalidado 2,640 registros con 37.50% de
evento.

**Confirmación independiente.** El histograma log-log muestra bimodalidad con un
valle vacío entre $10^1$ y $10^2$; el umbral de 100 cae exactamente ahí.

**Lección.** Una intuición matemáticamente razonable ("un ratio no supera 1")
puede ser catastrófica si contradice la evidencia empírica.

---

## E2 — Códigos centinela 96/98

**Objetivo.** Determinar si los valores 96 y 98 son datos o códigos.

**Método.** Distribución por valor exacto; tasa de evento; coincidencia entre
columnas; efecto sobre las correlaciones.

**Resultado [MEDIDO].** 214 filas con 54.67% de evento (vs 6.60% del resto).
**Las tres columnas de morosidad tienen el mismo valor simultáneamente en las
269**. Su eliminación reduce las correlaciones entre las tres variables de
~0.99 a ~0.25.

**Interpretación.** Código de sistema, no dato. El grupo más predictivo del
conjunto.

**Tratamiento.** `96/98 → NaN` + `flag_registro_invalido`. Adicionalmente, los
valores legítimos se agrupan a partir de 4 porque la relación satura.

---

## E3 — Ingresos codificados como cero

**Objetivo.** Verificar si `MonthlyIncome = 0` es un ingreso real.

**Motivación.** Detectado al sustituir el histograma recortado al percentil 99
por uno de rango completo, que reveló un pico anómalo.

**Resultado [MEDIDO].** Ver [§3.2.3](03_pipeline_datos.md#323-monthlyincome--1).

**Interpretación.** Mismo patrón que E1 y E2: código disfrazado de número, con
riesgo **inverso** al que el valor sugiere.

**Impacto medido tras el tratamiento.**

| Métrica | Antes | Después |
|---|---|---|
| IV de `MonthlyIncome` | 0.0703 | **0.0784** |
| IV de `ingreso_por_persona` | 0.0903 (débil) | **0.1049 (medio)** |

**Lección metodológica.** El recorte al percentil 99 en el EDA ocultaba tres
patologías. **Un histograma que se ve aplastado por un outlier está haciendo su
trabajo: esa fealdad es el hallazgo.**

---

## E4 — Bug en `peor_atraso`

**Objetivo.** (No planificado — detectado al revisar las tablas WoE.)

**Síntoma [MEDIDO].**

```
total_atrasos   bin (-0.1, 0.0]  →  n=95,692  evento 2.74%  + bin FALTANTE n=214 (54.67%)
peor_atraso     bin (-0.1, 0.0]  →  n=95,906  evento 2.86%  ← sin bin FALTANTE
```

Diferencia exacta: **214 filas**, las de los centinelas.

**Causa.** `peor_atraso` se calculaba con comparaciones `> 0`. Como `NaN > 0`
evalúa a `False`, las filas con centinela quedaban en `peor_atraso = 0` ("sin
atrasos") — el grupo más seguro — teniendo 54.67% de evento.

**Corrección.**

```python
sin_historial = df[COLS_MORA].isna().all(axis=1)
df.loc[sin_historial, "peor_atraso"] = np.nan
```

**Impacto [MEDIDO].** IV de `peor_atraso`: 1.3326 → **1.4052**. El bin limpio
bajó de 2.86% a 2.74%.

**Prevención.** Prueba de regresión
`test_regresion_peor_atraso_no_traga_centinelas`.

---

## E5 — Selección de variables para la regresión logística

**Objetivo.** Elegir el subconjunto que evite multicolinealidad.

**Resultado [MEDIDO].** Ver [§4.2.2](04_modelado.md#422-selección-de-variables).

**Hallazgo central.** El subconjunto con **mejor AUC** (B, 0.86300) tenía **4
coeficientes con signo invertido**, incluido uno que afirmaba "más episodios de
mora leve → menos riesgo".

**Causa raíz.** Todas las variables derivadas —no solo las de morosidad—
competían con sus variables padre:

| Derivada | Compite con |
|---|---|
| `ingreso_por_persona` | `MonthlyIncome`, `NumberOfDependents` |
| `prop_inmobiliaria` | `NumberRealEstateLoansOrLines`, `NumberOfOpenCreditLinesAndLoans` |
| `deuda_bruta_proxy` | `DebtRatio_limpio` |
| `total_atrasos`, `peor_atraso`, `flag_algun_atraso` | Las tres originales de mora |

**Decisión.** Cambiar el criterio de selección: **exigir signos coherentes
primero**, maximizar AUC después.

**Coste [MEDIDO].** +0.00076 de AUC (0.088%).

> **Nota posterior.** Este experimento solo revisaba las variables `woe_`. La
> auditoría descubrió después que la comprobación era incompleta y que tres
> indicadores llegaban al modelo con el signo contrario a su comportamiento
> univariado. Ver **E16**.

**Lección.** La ingeniería de variables puede restar. Siete variables con lógica
de negocio sólida no aportaron poder predictivo y rompieron la interpretabilidad.

---

## E6 — Sensibilidad al parámetro `C`

**Objetivo.** Determinar si la regularización importa.

**Resultado [MEDIDO].** Ver [§2.4.4](02_fundamentos_matematicos.md#244-regularización).

**Interpretación.** Con ~6,666 observaciones por parámetro, los datos dominan por
completo. El AUC de entrenamiento (0.85913) es **menor** que el de validación
(0.86272), lo que descarta sobreajuste y por tanto la necesidad de regularizar.

**Lección.** Saber qué **no** ajustar ahorra tanto tiempo como saber qué ajustar.
Los parámetros que movieron la aguja en este proyecto fueron los de limpieza
(umbrales, centinelas), no los del algoritmo.

---

## E7 — Sobreajuste de XGBoost con valores por defecto

**Objetivo.** Establecer la línea base antes de ajustar.

**Resultado [MEDIDO].** `AUC ajuste 0.92919` / `AUC validación 0.85592` — brecha
de 7.3 puntos, y **peor que la regresión logística**.

Tras el ajuste: `0.87651 / 0.86854`, brecha de 0.8 puntos.

**Interpretación.** Los valores por defecto de XGBoost son agresivos. Es el error
más común al usar la biblioteca.

---

## E8 — ¿Aportan las derivadas a los árboles?

**Objetivo.** Verificar si la conclusión de E5 se extiende a modelos no lineales.

**Resultado [MEDIDO]** (sobre el conjunto de *early stopping*):

| Subconjunto | n | AUC ajuste | AUC early-stop |
|---|---|---|---|
| Todas | 25 | 0.87853 | 0.86350 |
| **Originales + indicadores** | 18 | 0.87619 | **0.86405** |
| Solo originales | 10 | 0.87616 | 0.86342 |

**Interpretación.** Mismo veredicto que E5, por razón distinta: en la rama lineal
las derivadas **rompían** el modelo (signos invertidos); en la rama de árboles
solo **redistribuyen el crédito** en la tabla de importancia sin aportar.

**Decisión.** Se conservan las 25 en la rama de árboles: la diferencia está
dentro del ruido y no hay coeficientes que puedan invertirse.

---

## E9 — Importancia de variables: inestabilidad entre algoritmos

**Objetivo.** Verificar si `feature_importances_` es interpretable.

**Resultado [MEDIDO].**

| Variable | XGBoost | LightGBM | SHAP (XGBoost) |
|---|---|---|---|
| `peor_atraso` | 34.2% | 31.9% | 31.94% |
| `total_atrasos` | 9.9% | **26.7%** | 14.69% |
| `flag_algun_atraso` | **27.8%** | fuera del top 12 | 4.48% |
| `RevolvingUtilization` | **4.7%** | **24.1%** | **30.91%** |

**Interpretación.** `RevolvingUtilization` recibe 4.7% en un algoritmo y 24.1%
en otro **con datos idénticos**. Con variables redundantes, cada algoritmo
reparte el crédito a su manera.

**Verificación complementaria [MEDIDO].** Las tres variables que acaparan el
71.9% de la importancia en XGBoost pueden eliminarse sin que el AUC empeore
(E8). **Importancia alta ≠ imprescindible.**

**Lección.** La importancia nativa es una propiedad **del modelo entrenado**, no
de los datos. Presentarla como "esto es lo que predice la mora" es presentar un
artefacto.

---

## E10 — Comparación de los tres modelos

**Objetivo.** Determinar si las diferencias son reales o ruido.

**Método.** Validación cruzada de 5 particiones **idénticas** para los tres
modelos (misma semilla, mismas filas por fold), y contraste emparejado.

**Resultado [MEDIDO].**

| Modelo | Media | Desv. | Mín | Máx |
|---|---|---|---|---|
| LogReg (WoE) | 0.85948 | 0.00335 | 0.85395 | 0.86370 |
| **XGBoost** | **0.86628** | 0.00306 | 0.86145 | 0.87091 |
| LightGBM | 0.86619 | 0.00367 | 0.86059 | 0.87139 |

| Comparación | $\bar d$ | Veredicto |
|---|---|---|
| LogReg vs XGBoost | −0.00685 | **real** |
| LogReg vs LightGBM | −0.00640 | **real** |
| XGBoost vs LightGBM | +0.00045 | **dentro del ruido** |

**Interpretación.** Con un solo split se habría concluido "XGBoost 0.86854 vs
LightGBM 0.86905, gana XGBoost". La validación cruzada muestra que esa diferencia
no es distinguible del azar.

---

## E11 — Validación cruzada anidada

**Objetivo.** Verificar si el AUC reportado está inflado por selección.

**Sospecha.** Los hiperparámetros se eligieron con un subconjunto del
entrenamiento que después aparece en los folds de test.

**Resultado [MEDIDO].**

| | Media | Desv. |
|---|---|---|
| CV reportada | 0.86628 | 0.00306 |
| **Nested CV** | **0.86661** | 0.00337 |

Diferencia: **−0.00028** (el anidado sale ligeramente *superior*).

**Interpretación.** El sesgo no es detectable. La rejilla es pequeña (11
configuraciones) frente al tamaño de la muestra.

**Importancia.** Éste era el único experimento capaz de invalidar la conclusión
central del proyecto.

---

## E12 — ¿Debe buscarse el umbral en los datos?

**Objetivo.** Contrastar el umbral teórico $M/(M+L)$ contra el hallado
empíricamente.

**Método.** Partir validación en dos mitades; hallar el óptimo en A; aplicarlo
en B.

**Resultado [MEDIDO].**

| Umbral | Origen | Beneficio en la mitad B |
|---|---|---|
| 0.1184 | empírico (mitad A) | $7,993,200 |
| **0.1176** | **fórmula $M/(M+L)$** | **$8,006,000** |

**Interpretación.** La fórmula supera al umbral ajustado. **El umbral se deriva
de la economía, no se aprende de los datos.**

---

## E13 — Ensemble

**Objetivo.** Verificar si promediar mejora.

**Resultado [MEDIDO].** AUC 0.86839, por debajo de XGBoost solo (0.86854), y
peor calibración (0.453 pp vs 0.338 pp).

**Interpretación.** Los tres modelos se equivocan en las mismas personas
(correlación 0.9928 entre XGBoost y LightGBM). Sin diversidad de errores, no hay
ganancia por promediado.

---

## E14 — ¿Es aplicable el PSI?

**Objetivo.** Evaluar si procede implementar monitoreo de estabilidad
poblacional.

**Resultado [MEDIDO].**

```
PSI máximo de las 15 variables : 0.00073
PSI medio                       : 0.00018
Umbral de alerta convencional   : 0.10        ← 137× superior
```

**Interpretación.** El PSI detecta deriva **temporal**. Aquí train y test son
particiones aleatorias de la misma población de 2011: estructuralmente no hay
deriva que medir.

**Decisión.** No implementarlo. Se documenta por qué no aplica, lo cual es más
informativo que publicar un 0.0007 sin significado.

**Lección.** Aplicar una técnica donde no hay nada que medir es relleno, no
rigor.

---

## E15 — Reproducibilidad desde cero

**Objetivo.** Verificar que el proyecto se reconstruye sin deriva.

**Método.** Eliminar `datasets/`, `artifacts/`, `outputs/` y todo caché; ejecutar
los 8 scripts en orden; comparar contra los resultados publicados.

**Resultado [MEDIDO].**

```
Reconstrucción completa       : 113 segundos
Ficheros de métricas idénticos: 12/12 byte a byte
Submissions idénticas         : 101,503/101,503 predicciones
Artefactos generados          : 17/17
Pruebas                       : 42/42
```

| Métrica | Publicado | Reconstruido |
|---|---|---|
| LogReg AUC (CV) | 0.85948 | 0.85948 ✅ |
| XGBoost AUC (CV) | 0.86628 | 0.86628 ✅ |
| LightGBM AUC (CV) | 0.86619 | 0.86619 ✅ |

**Interpretación.** Semillas correctamente fijadas, sin dependencia del orden de
ejecución ni de estado previo.

---

## E16 — Corrección de H1: coherencia completa de signos

**Objetivo.** Eliminar los coeficientes que contradicen la dirección univariada
de su variable.

**Motivación.** La auditoría (capítulo [07](07_auditoria.md#h1)) detectó que
`signos_invertidos()` eximía a los indicadores, y que tres de ellos tenían el
signo al revés. El peor, `flag_registro_invalido`, identifica al grupo con
**54.67%** de eventos frente al 6.60% de la población, y su coeficiente negativo
le asignaba **puntos positivos** en el scorecard.

**Causa raíz [MEDIDO].** Colinealidad perfecta con los bins `FALTANTE`:

| Indicador | Variable WoE | Filas | Coincidencia |
|---|---|---|---|
| `flag_registro_invalido` | `woe_NumberOfTimes90DaysLate` | 214 | **100%** |
| `flag_debtratio_invalido` | `woe_DebtRatio_limpio` | 24,718 | **100%** |
| `flag_revutil_invalido` | `woe_RevolvingUtilizationOfUnsecuredLines` | 172 | **100%** |

Cuando un valor se convierte en `NaN` ocurren dos cosas a la vez: WoE lo coloca
en `FALTANTE` **y** se enciende el indicador. Las mismas filas contadas dos
veces.

**Método.** Nueva función `signos_incoherentes()` que verifica variables `woe_`
contra la convención de signo **e indicadores contra su dirección univariada
medida en los datos**. Ocho subconjuntos candidatos evaluados con el criterio
completo.

**Resultado [MEDIDO].**

| Subconjunto | n | AUC selección | VIF máx | Incoherentes |
|---|---|---|---|---|
| A: todas | 25 | 0.86135 | 80.5 | 8 |
| E: originales + indicadores (anterior) | 18 | 0.86059 | 8.8 | 3 |
| G: originales + `flag_algun_atraso` | 11 | 0.86006 | 3.6 | 1 |
| **H: solo las 10 originales** | **10** | 0.85749 | **1.3** | **0** |

**Hallazgo central.** Solo el subconjunto de las 10 variables originales alcanza
coherencia total. Retirar los tres indicadores colineales **desplaza** el problema
a otros (`flag_dependents_faltante`, luego
`woe_NumberOfOpenCreditLinesAndLoans`) en lugar de resolverlo.

**Coste [MEDIDO].**

| Métrica | Antes | Después | Δ |
|---|---|---|---|
| AUC (CV) | 0.85948 | 0.85794 | −0.00187 |
| KS | 0.5760 | 0.5696 | −0.0064 |
| Calibración | 0.53 pp | 0.75 pp | peor |
| VIF máximo | 8.8 | **1.3** | mejor |
| Valor de negocio | $175.33 | $171.95 | −$3.38 |

**Interpretación.** La corrección **empeora todas las métricas de rendimiento**.
Se acepta porque un scorecard cuya tabla de puntos contradice la evidencia no
sirve para su único propósito. A cambio, los predictores quedan prácticamente
ortogonales (VIF 1.3) y los coeficientes, estables.

**Lección.** En un scorecard WoE, la codificación de las variables originales ya
contiene los faltantes (bin `FALTANTE`), los extremos (bins de los bordes) y la
no-linealidad (el binning). **Todo indicador de calidad de dato es redundante por
construcción.**

**Prevención.** Seis pruebas nuevas, entre ellas
`test_la_comprobacion_completa_es_mas_estricta_que_la_antigua`, que verifica que
`signos_incoherentes()` detecta un superconjunto estricto de lo que detectaba la
comprobación anterior.

---

## Síntesis de lecciones

| # | Lección |
|---|---|
| 1 | Los códigos de sistema invierten la señal. Tres variables tenían valores que parecían extremos y cuyo riesgo real era el **opuesto**. |
| 2 | La ingeniería de variables puede restar. Siete derivadas con lógica sólida no aportaron nada y rompieron la interpretabilidad. |
| 3 | La importancia de variables es del modelo, no de los datos. |
| 4 | Un solo split no permite comparar modelos: ±0.003 de ruido. |
| 5 | El umbral de decisión se deriva de la economía, no se aprende. |
| 6 | Los hiperparámetros del algoritmo importaron mucho menos que los umbrales de limpieza. |
| 7 | Las gráficas del EDA deben usar datos crudos; recortar oculta el hallazgo. |
| 8 | Aplicar una técnica donde no hay nada que medir es relleno. |
| 9 | Una comprobación con exenciones puede ocultar defectos: eximir a los indicadores del control de signos permitió que tres llegaran al modelo con la dirección invertida. |
| 10 | En un scorecard WoE los indicadores de faltante son redundantes por construcción — el bin `FALTANTE` ya los codifica. |

---

[← Modelado](04_modelado.md) · [Índice](README.md) · [Siguiente: Negocio y regulación →](06_negocio_y_regulacion.md)

# 04 — Modelado: los tres estimadores y sus hiperparámetros

[← Pipeline de datos](03_pipeline_datos.md) · [Índice](README.md) · [Siguiente: Experimentos →](05_experimentos.md)

---

## 4.1 Por qué dos preparaciones de datos incompatibles

**[RECOM]** El error habitual es aplicar un único preprocesamiento a todos los
modelos. Las familias tienen requisitos opuestos:

| Aspecto | Árboles (XGBoost, LightGBM) | Regresión logística |
|---|---|---|
| Outliers | Sin capear | Binning (los absorbe) |
| Faltantes | `NaN` nativo | Bin propio (`FALTANTE`) |
| Escalado | Innecesario | Innecesario (WoE ya normaliza) |
| Codificación | Valores crudos | Weight of Evidence |
| No-linealidad | La aprende sola | Hay que codificarla |
| Multicolinealidad | Indiferente | **Crítica** |
| Variables derivadas | Tolerables | Peligrosas |

### 4.1.1 Fundamento de "no capear para árboles"

**[TEORÍA]** Sea $g$ una transformación estrictamente creciente. Un árbol evalúa
divisiones de la forma $\mathbb{1}\{x \leq c\}$. Como

$$x \leq c \iff g(x) \leq g(c)$$

el conjunto de particiones alcanzables es idéntico. Capear no es una
transformación estrictamente creciente: introduce empates por encima del tope, y
**elimina** la posibilidad de cortar ahí. Es una pérdida estricta de capacidad.

### 4.1.2 Fundamento de "no imputar para árboles"

Ver [§2.6.4](02_fundamentos_matematicos.md#264-manejo-nativo-de-faltantes). El
algoritmo aprende la dirección por defecto que maximiza la ganancia. Con
faltantes informativos (5.57% vs 6.96%), imputar destruye señal.

---

## 4.2 Modelo 1 — Regresión logística sobre WoE

**Script:** `03_train_logistic.py` (~540 líneas, 9 pasos)

### 4.2.1 Rol en el proyecto

Es el **baseline**: el número de referencia contra el que se compara todo lo
demás. Sin baseline, afirmar que un modelo "es bueno" carece de contenido.

Además es el único que produce un **scorecard** (capítulo
[02 §2.11](02_fundamentos_matematicos.md#211-construcción-del-scorecard)).

### 4.2.2 Selección de variables

Se comparan **ocho** subconjuntos, entrenando en 96,000 filas y evaluando en las
24,000 de selección. **La validación no participa.**

**[MEDIDO]**

Con la comprobación **completa** de coherencia —variables WoE contra la
convención de signo **e indicadores contra su dirección univariada medida en los
datos** (ver [H1](07_auditoria.md#h1)):

| Subconjunto | n vars | AUC ajuste | AUC selección | VIF máx | Incoherentes |
|---|---|---|---|---|---|
| A: todas | 25 | 0.86007 | 0.86135 | 80.5 | 8 |
| B: sin `flag_algun`, con `total_atrasos` | 23 | 0.86011 | 0.86108 | 18.1 | 8 |
| C: sin `flag_algun`, con `peor_atraso` | 23 | 0.85955 | 0.86102 | 10.8 | 6 |
| D: sin derivadas de morosidad | 22 | 0.85830 | 0.85889 | 10.8 | 4 |
| E: originales + todos los indicadores | 18 | 0.85885 | 0.86059 | 8.8 | 3 |
| F: sin indicadores de calidad | 18 | 0.85955 | 0.86077 | 58.7 | 5 |
| G: originales + `flag_algun_atraso` | 11 | 0.85859 | 0.86006 | 3.6 | 1 |
| **H: solo las 10 originales** | **10** | 0.85718 | **0.85749** | **1.3** | **0** |

**Resultado central:** solo el subconjunto de las 10 variables originales alcanza
coherencia total. Añadir cualquier indicador o variable derivada **desplaza** la
colinealidad en lugar de eliminarla.

### 4.2.3 Criterio de selección: signos primero, AUC después

```python
inc = signos_incoherentes(m.coef_[0], cols, X_fit, y_fit)
...
validos = comparacion[comparacion["incoherentes"] == 0]
mejor = validos.sort_values(["AUC_seleccion", "VIF_max"],
                            ascending=[False, True]).iloc[0]
```

**Justificación [RECOM]:** un coeficiente que contradice la evidencia univariada
no es un detalle estético. Afirma literalmente "más episodios de mora → menos
riesgo", o le da puntos a favor al grupo con 54.67% de eventos. Ante un
supervisor —o ante el cliente rechazado— es indefendible.

**Coste de la restricción [MEDIDO]:** +0.00386 de AUC de selección (0.45%). El
mejor AUC absoluto lo daba un subconjunto con 8 coeficientes incoherentes.

**Principio subyacente [RECOM]:** en un scorecard WoE la codificación de las
variables originales **ya lo contiene todo** — los faltantes (bin `FALTANTE`),
los extremos (bins de los bordes) y la no-linealidad (el propio binning). Añadir
indicadores de calidad o variables derivadas solo puede introducir redundancia.

### 4.2.4 Hiperparámetros

| Parámetro | Valor | Justificación |
|---|---|---|
| `penalty` | `l2` (defecto) | Convexa y diferenciable; con $n \gg p$ apenas actúa |
| `C` | `1.0` (defecto) | **Verificado irrelevante**: barrido de 0.001 a 100 mueve el AUC en $6\times10^{-5}$ |
| `solver` | `lbfgs` (defecto) | Cuasi-Newton; converge holgadamente |
| `max_iter` | `2000` | Holgura defensiva; el uso real es de decenas de iteraciones |
| `class_weight` | `None` | `'balanced'` distorsiona la PD media de 6.7% a 33% sin ganar AUC |
| `random_state` | `42` | Reproducibilidad |

### 4.2.5 Coeficientes del modelo final [MEDIDO]

Intercepto: $\beta_0 = -2.9639$

| Variable | $\beta$ | Signo |
|---|---|---|
| `woe_DebtRatio_limpio` | −0.8402 | ✅ |
| `woe_RevolvingUtilizationOfUnsecuredLines` | −0.6123 | ✅ |
| `woe_NumberRealEstateLoansOrLines` | −0.5780 | ✅ |
| `woe_NumberOfTimes90DaysLate` | −0.5248 | ✅ |
| `woe_NumberOfTime30-59DaysPastDueNotWorse` | −0.4917 | ✅ |
| `woe_age` | −0.3873 | ✅ |
| `woe_NumberOfTime60-89DaysPastDueNotWorse` | −0.3714 | ✅ |
| `woe_NumberOfDependents` | −0.2679 | ✅ |
| `woe_MonthlyIncome` | −0.0747 | ✅ |
| `woe_NumberOfOpenCreditLinesAndLoans` | −0.0457 | ✅ |

**Los diez coeficientes son negativos**, como exige la convención WoE, y ninguno
contradice su dirección univariada.

### 4.2.6 Diagnósticos [MEDIDO]

```
[1] CONVERGENCIA     converge holgadamente dentro de max_iter=2000
[2] CALIBRACIÓN      error absoluto medio 0.75 pp
[3] INCERTIDUMBRE    CV 5 folds: 0.85794 ± 0.00364
[4] PARÁMETRO C      de 0.001 a 100 → el AUC no se mueve
```

### 4.2.7 Coste medido de la corrección H1

| Métrica | Antes (18 vars, 3 incoherencias) | Después (10 vars, 0) | Δ |
|---|---|---|---|
| AUC validación | 0.86272 | 0.86059 | −0.00257 |
| AUC (CV) | 0.85948 | 0.85794 | −0.00187 |
| KS | 0.5760 | 0.5696 | −0.0064 |
| Calibración | 0.53 pp | **0.75 pp** | peor |
| VIF máximo | 8.8 | **1.3** | mejor |
| Valor de negocio | $175.33 | $171.95 | −$3.38 |

**Lectura honesta [RECOM]:** la corrección **empeora** las cuatro métricas de
rendimiento. Se acepta porque un scorecard cuya tabla de puntos contradice la
evidencia no sirve para su único propósito —explicar decisiones— por muy alto que
sea su AUC. La contrapartida es un VIF de 1.3: los predictores son
prácticamente ortogonales y los coeficientes, estables.

---

## 4.3 Modelo 2 — XGBoost

**Script:** `04_train_xgboost.py` (442 líneas, 10 pasos)

### 4.3.1 Verificación del formato de entrada [MEDIDO]

```
Dimensiones          : 120,000 × 25
Todas numéricas      : True
Valores infinitos    : 0
Columnas con NaN     : 14 de 25 (203,312 valores)
Nombres problemáticos: ninguno
```

Los `NaN` se conservan **a propósito**: son información.

### 4.3.2 El conjunto de *early stopping*: por qué no se usa validación

```python
X_fit, X_es, y_fit, y_es = train_test_split(
    X_tr_full, y_tr_full, test_size=0.2, stratify=y_tr_full, random_state=SEMILLA)
```

**[RECOM]** Si se usara validación para decidir cuándo dejar de añadir árboles,
se estaría *ajustando* a ese conjunto, y el AUC reportado sobre él sería
optimista. Apartar un trozo del propio entrenamiento mantiene validación
intacta.

### 4.3.3 El sobreajuste por defecto [MEDIDO]

```
Con parámetros por defecto:
  AUC ajuste     : 0.92919   ← memorizando
  AUC validación : 0.85592   ← peor que la regresión logística
```

Brecha de **7.3 puntos**. Es el error más común al usar XGBoost: sin control de
complejidad sigue construyendo árboles hasta memorizar el ruido.

Tras el ajuste: `0.87651 / 0.86854` — brecha de 0.8 puntos y +0.013 de AUC.

### 4.3.4 Rejilla de hiperparámetros [MEDIDO]

| `max_depth` | `min_child_weight` | `subsample` | `colsample` | `λ` | Árboles | AUC (early-stop) |
|---|---|---|---|---|---|---|
| 3 | 1 | 1.0 | 1.0 | 1 | 164 | 0.86489 |
| 4 | 1 | 1.0 | 1.0 | 1 | 113 | 0.86464 |
| 5 | 5 | 0.8 | 0.8 | 1 | 102 | 0.86555 |
| **5** | **20** | **0.8** | **0.6** | **1** | **105** | **0.86598** ✅ |
| 7 | 20 | 0.8 | 0.6 | 1 | 44 | 0.86591 |
| 6 | 10 | 0.8 | 0.8 | 5 | 91 | 0.86572 |

**La selección se hace por `auc_es`, no por `auc_val`** — elegir por validación
la convertiría en parte de la decisión.

### 4.3.5 Significado de cada hiperparámetro

| Parámetro | Valor | Significado matemático | Efecto |
|---|---|---|---|
| `max_depth` | 5 | Profundidad máxima; $\leq 2^5 = 32$ hojas | Orden máximo de interacción capturable |
| `min_child_weight` | 20 | $\sum_{k\in I_j} h_k \geq 20$ con $h_k = p_k(1-p_k)$ | Con $p\approx0.07$, ~300 observaciones por hoja |
| `subsample` | 0.8 | Fracción de filas por árbol | Reduce varianza (*bagging* dentro del *boosting*) |
| `colsample_bytree` | 0.6 | Fracción de columnas por árbol | Descorrelaciona árboles |
| `learning_rate` | 0.05 | $\nu$ en $F_M = \sum \nu f_m$ | Bajo = más lento, mejor generalización |
| `n_estimators` | 105 (por *early stopping*) | $M$ | Complejidad total |
| `reg_lambda` | 1 (defecto) | $\lambda$ en $\Omega(f)$ | Encoge los pesos de las hojas |

**Sobre `subsample` y `colsample` [TEORÍA]:** se le oculta información a cada
árbol de forma deliberada. La lógica es la de un comité: 105 evaluadores que
vieron datos parciales distintos producen un consenso más robusto que 105 que
vieron todo y comparten el mismo sesgo.

### 4.3.6 `scale_pos_weight` [MEDIDO]

| Configuración | AUC | PD media predicha |
|---|---|---|
| Sin ajustar | **0.86854** | **6.68%** (real: 6.68%) |
| `scale_pos_weight = 14.0` | 0.86793 | 33.48% |

Descartado: no mejora la discriminación y destruye la calibración.

---

## 4.4 Modelo 3 — LightGBM

**Script:** `05_train_lightgbm.py` (421 líneas, 8 pasos)

### 4.4.1 Diferencia estructural con XGBoost

Ver [§2.6.3](02_fundamentos_matematicos.md#263-lightgbm-crecimiento-por-hojas).
Crecimiento por hojas frente a por niveles; gobierna `num_leaves`, no
`max_depth`.

**Consecuencia práctica [RECOM]:** no se puede copiar la configuración de
XGBoost. `max_depth=5` en XGBoost permite hasta 32 hojas equilibradas; en
LightGBM hay que fijar `num_leaves` explícitamente.

### 4.4.2 Velocidad y calidad de los valores por defecto [MEDIDO]

| | Tiempo | AUC ajuste | AUC validación |
|---|---|---|---|
| LightGBM | **0.4 s** | 0.90697 | **0.86644** |
| XGBoost | 1.7 s | 0.92919 | 0.85592 |

LightGBM es ~4× más rápido y sus valores por defecto están notablemente mejor
calibrados: con `n_estimators=100` sin ajustar ya supera a la regresión
logística, mientras XGBoost por defecto queda por debajo.

### 4.4.3 Rejilla [MEDIDO]

| `num_leaves` | `min_child_samples` | `feature_fraction` | `bagging_fraction` | Árboles | AUC (early-stop) |
|---|---|---|---|---|---|
| **15** | **50** | **0.8** | **0.8** | **90** | **0.86569** ✅ |
| 31 | 20 | 1.0 | 1.0 | 105 | 0.86526 |
| 31 | 100 | 0.6 | 0.8 | 70 | 0.86545 |
| 63 | 20 | 1.0 | 1.0 | 85 | 0.86294 |
| 127 | 20 | 1.0 | 1.0 | 24 | 0.86237 |

**Ganó la configuración más simple probada.** Con crecimiento por hojas la
complejidad se dispara rápido: `num_leaves=127` degrada el AUC en 0.003 y el
*early stopping* corta a 24 árboles porque empieza a sobreajustar de inmediato.

### 4.4.4 Correspondencia de hiperparámetros entre bibliotecas

| LightGBM | XGBoost aproximado | Nota |
|---|---|---|
| `num_leaves` | (sin equivalente directo) | El más importante en LightGBM |
| `min_child_samples` | `min_child_weight` | LightGBM cuenta observaciones; XGBoost suma hessianas |
| `feature_fraction` | `colsample_bytree` | Idéntico |
| `bagging_fraction` + `bagging_freq` | `subsample` | LightGBM requiere `bagging_freq ≥ 1` para activarlo |
| `reg_lambda` | `reg_lambda` | Idéntico |

**[IMPL]** Se fija `bagging_freq=1`; sin él, `bagging_fraction` se ignora
silenciosamente — un error frecuente al migrar configuraciones.

---

## 4.5 El ensemble evaluado y descartado

**[MEDIDO]** Se probó el promedio simple de las tres probabilidades:

| Modelo | AUC val. | KS | Calibración |
|---|---|---|---|
| XGBoost | **0.86854** | **0.58323** | **0.337 pp** |
| LightGBM | 0.86905 | 0.58848 | 0.386 pp |
| Ensemble | 0.86803 | 0.58118 | 0.572 pp |
| LogReg | 0.86059 | 0.57197 | 0.750 pp |

**No ganó.** El promedio ayuda cuando los modelos se equivocan en observaciones
distintas; aquí los tres fallan en las mismas personas (correlación de 0.9928
entre las predicciones de XGBoost y LightGBM). Además empeoró notablemente la
calibración (0.572 pp frente a 0.337 pp del mejor individual), porque promediar
probabilidades de modelos con calibraciones distintas degrada la resultante.

**[RECOM]** Se conserva el código del ensemble porque el intento documentado
vale tanto como el resultado: demuestra que la opción se evaluó, no que se
ignoró.

---

## 4.6 Reentrenamiento final

**[IMPL]** Una vez tomadas **todas** las decisiones (variables, hiperparámetros,
número de árboles), el modelo final se reentrena con las **150,000** filas
etiquetadas.

**Justificación [TEORÍA]:** apartar validación tenía un propósito —permitir
medir sin engañarse— que ya se cumplió. Esos 30,000 registros etiquetados son
información utilizable. Lo que **no** puede hacerse es volver a mirarlos para
decidir algo.

**Detalle técnico [IMPL]:** al reentrenar no hay conjunto de vigilancia, luego
`n_estimators` se fija en el valor que eligió el *early stopping* (105) y se
desactiva `early_stopping_rounds`.

**Comprobación de cordura [MEDIDO]:** el AUC in-sample sobre las filas que antes
eran validación sube a 0.87860. **No es una estimación de rendimiento** —esas
filas ahora forman parte del entrenamiento— sino una verificación de que el
reajuste funcionó.

---

## 4.7 Resumen comparativo

| Criterio | LogReg (WoE) | XGBoost | LightGBM |
|---|---|---|---|
| AUC (CV, 5 folds) | 0.85794 ± 0.00364 | **0.86628 ± 0.00306** | 0.86619 ± 0.00367 |
| Gini | 0.7203 | **0.7374** | 0.7370 |
| KS | 0.5696 | **0.5865** | 0.5832 |
| Average Precision | 0.3882 | **0.4057** | 0.4034 |
| Error de calibración | 0.72 pp | **0.35 pp** | 0.36 pp |
| Interpretabilidad | **Total (scorecard)** | Baja (requiere SHAP) | Baja (requiere SHAP) |
| Tiempo de entrenamiento | Instantáneo | 1.7 s | **0.4 s** |
| Complejidad de despliegue | **Mínima** | Media | Media |
| Riesgo de sobreajuste | **Muy bajo** | Medio | Medio-alto |
| Valor de negocio | $172.59/sol. | $180.37/sol. | **$182.21/sol.** |

**Nota sobre la última fila [MEDIDO]:** LightGBM **gana en dinero pese a perder
en AUC**. No es contradictorio: el AUC promedia el rendimiento sobre todos los
umbrales, mientras que el beneficio depende solo de cómo ordena alrededor del
11.76%. LightGBM acierta mejor justo en esa zona.

**[RECOM]** Es otra razón para no elegir modelo únicamente por AUC.

---

[← Pipeline de datos](03_pipeline_datos.md) · [Índice](README.md) · [Siguiente: Experimentos →](05_experimentos.md)

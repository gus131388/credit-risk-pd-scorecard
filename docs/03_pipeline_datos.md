# 03 — Pipeline de datos: limpieza, ingeniería y codificación

[← Fundamentos](02_fundamentos_matematicos.md) · [Índice](README.md) · [Siguiente: Modelado →](04_modelado.md)

---

## 3.1 El conjunto de datos

**Fuente:** competencia *Give Me Some Credit* (Kaggle, 2011). Datos históricos
anonimizados de solicitantes reales, no sintéticos.

| Fichero | Filas | Columnas | Rol |
|---|---|---|---|
| `cs-training.csv` | 150,000 | 11 | Entrenamiento (con etiqueta) |
| `cs-test.csv` | 101,503 | 11 | Inferencia (etiqueta oculta por Kaggle) |
| `sampleEntry.csv` | 101,503 | 2 | Formato de entrega |
| `Data Dictionary.xls` | 11 filas | 3 | Definición de campos |

**Nota [IMPL]:** el índice de cada CSV numera desde 1 de forma independiente. La
fila 1 de `train` y la fila 1 de `test` son **personas distintas**. Se comprobó:
edad 45 / ingreso $9,120 frente a edad 43 / ingreso $5,700.

### 3.1.1 Diccionario de variables

| Variable | Definición | Tipo | Grupo conceptual |
|---|---|---|---|
| `SeriousDlqin2yrs` | Mora de 90+ días en 2 años | binaria | **objetivo** |
| `RevolvingUtilizationOfUnsecuredLines` | Saldo de crédito rotativo / suma de límites | ratio | Uso de crédito |
| `DebtRatio` | (Pagos de deuda + alimentos + gastos) / ingreso bruto | ratio | Endeudamiento |
| `MonthlyIncome` | Ingreso mensual | continua | Capacidad |
| `age` | Edad en años | entera | Perfil |
| `NumberOfDependents` | Dependientes excluido el titular | entera | Perfil |
| `NumberOfOpenCreditLinesAndLoans` | Líneas y préstamos abiertos | entera | Estructura |
| `NumberRealEstateLoansOrLines` | Hipotecas y líneas con garantía inmobiliaria | entera | Estructura |
| `NumberOfTime30-59DaysPastDueNotWorse` | Episodios de mora leve (2 años) | entera | Historial |
| `NumberOfTime60-89DaysPastDueNotWorse` | Episodios de mora media (2 años) | entera | Historial |
| `NumberOfTimes90DaysLate` | Episodios de mora grave | entera | Historial |

### 3.1.2 Alcance del análisis exploratorio [IMPL]

**Todas las cifras de este capítulo proceden de las 120,000 filas de
entrenamiento, no de las 150,000 del fichero.** `01_eda.py` ejecuta el split
—misma semilla y mismo `stratify` que `02_preprocessing.py`— *antes* de calcular
cualquier estadístico. Las 30,000 filas de validación no se inspeccionan en
ningún punto del análisis.

El motivo no es que el EDA ajuste parámetros: no ajusta ninguno. Es que el EDA es
**donde se toman las decisiones** —qué códigos son centinela, dónde cortar una
cola, qué indicadores crear— y varias de esas decisiones se apoyan en vistas que
involucran el objetivo (tasa de evento por decil, por patrón de faltantes). Si
esas vistas incluyeran las filas de validación, dichas filas habrían participado
en el diseño del modelo y dejarían de ser un juez independiente. La contaminación
no entra por un `.fit()`; entra por el analista.

Consecuencia práctica: los conteos de este capítulo son ~80% de los del fichero
completo. Las tasas y las conclusiones son estables — ningún umbral cambió al
recalcularlos sobre la partición de entrenamiento.

### 3.1.3 Balance del objetivo [MEDIDO]

$$\pi = P(Y=1) = \frac{8{,}021}{120{,}000} = 0.0668$$

Ratio de desbalance: 1 evento por cada 14 no-eventos. El estratificado preserva
la tasa exacta del fichero original (6.68%). Implicación directa en la elección
de métricas (§2.7.1).

---

## 3.2 Diagnóstico: los tres códigos de sistema

El hallazgo transversal del análisis exploratorio. Tres variables contenían
valores que **parecían** datos extremos y en realidad eran códigos de "no
disponible". **En los tres casos el riesgo real era el opuesto al que sugería el
número.**

### 3.2.1 Centinelas 96 y 98 en las variables de morosidad [MEDIDO]

```
NumberOfTime30-59DaysPastDueNotWorse:  ... 13(n=1), 96(n=2), 98(n=212)
NumberOfTimes90DaysLate:               ... 17(n=1), 96(n=2), 98(n=212)
NumberOfTime60-89DaysPastDueNotWorse:  ... 11(n=1), 96(n=2), 98(n=212)
```

| Grupo | n | Tasa de evento |
|---|---|---|
| Con centinela 96/98 | 214 | **54.67%** |
| Resto | 119,786 | 6.60% |

Verificación adicional: **las 214 filas tienen el mismo valor en las tres
columnas simultáneamente** (98, 98, 98). Nadie acumula exactamente 98 episodios
en tres categorías distintas: es un código de sistema.

#### Efecto sobre la matriz de correlaciones [MEDIDO]

| Par | Con centinelas | Sin centinelas |
|---|---|---|
| 30-59 ↔ 90+ | **0.984** | 0.221 |
| 30-59 ↔ 60-89 | **0.987** | 0.306 |
| 90+ ↔ 60-89 | **0.993** | 0.296 |

**214 filas (0.18% de la muestra) inflaron la correlación de ~0.27 a ~0.99.**
Sin investigar, la conclusión habría sido "las tres variables son redundantes,
elimino dos" — destruyendo información valiosa.

**Lección [RECOM]:** ningún estadístico agregado debe interpretarse antes de
revisar los outliers.

### 3.2.2 `RevolvingUtilizationOfUnsecuredLines > 100`

Ver la tabla completa en [§1.4.1.1](01_arquitectura.md). Resumen: entre 1 y 100
el riesgo es de **39.67%** (n=2,468) —sobregiros legítimos, el grupo más
predictivo del conjunto—; por encima de 100 se desploma a **3.03%** en la banda
100–1,000 (n=33) y vuelve a **7.19%** por encima de 1,000 (n=139), es decir
**indistinguible de la media global de 6.68%**. Deja de comportarse como deuda.

El histograma log-log confirma **bimodalidad**: una masa principal con pico justo
por debajo de 1, un valle casi vacío entre 10 y 100, y una segunda joroba
separada entre $10^2$ y $10^4$. El umbral cae exactamente en el valle.

### 3.2.3 `MonthlyIncome ≤ 1` [MEDIDO]

El caso mejor documentado de los tres, porque admite una prueba directa. Se
verifica en cuatro pasos (`01_eda.py`, sección 5e).

#### (1) Son valores discretos, no una banda continua

| Valor | n |
|---|---|
| `0.0` | 1,322 |
| `1.0` | 498 |
| `2.0`–`10.0` | 12 (dispersos) |
| estrictamente entre 0 y 1 | **0** |

Dos picos aislados sobre casi nada. **No existe ningún valor fraccionario**: no
hay una escala de ingresos bajos, hay dos códigos.

#### (2) Prueba del denominador — la evidencia decisiva

$\text{DebtRatio} = \text{deuda mensual} / \text{ingreso mensual}$. Si el sistema
de origen carecía de ingreso no pudo dividir, y lo esperable es que almacenara el
**numerador en bruto**. Se contrasta contra la deuda real de las filas con
ingreso normal, reconstruida como $\text{DebtRatio} \times \text{MonthlyIncome}$:

| Grupo | p25 | p50 | p75 | % con `DebtRatio > 2` |
|---|---|---|---|---|
| `MonthlyIncome = 0` | 101 | 914 | 2,150 | 92.9% |
| `MonthlyIncome = 1` | 104 | 562 | 1,324 | 91.4% |
| `MonthlyIncome` faltante | 123 | 1,163 | 2,384 | 93.3% |
| **deuda real** (ingreso > 100) | 639 | 1,657 | 2,901 | — |
| *un ratio auténtico* | ~0.1 | ~0.3 | ~0.5 | **1.0%** |

Los tres grupos sospechosos tienen `DebtRatio` en el orden de **cientos o miles
de dólares**, indistinguible de la deuda mensual real. Un ratio auténtico tiene
mediana 0.3 y solo el 1.0% supera 2. **No son ratios: son importes.**

#### (3) El riesgo descarta la interpretación de pobreza

| Grupo | n | Eventos | Tasa | IC 95% (Wilson) |
|---|---|---|---|---|
| `= 0` | 1,322 | 55 | **4.16%** | [3.21, 5.38] |
| `= 1` | 498 | 12 | **2.41%** | [1.38, 4.16] |
| `(1, 100]` | 49 | 1 | 2.04% | [0.36, 10.69] |
| Faltante | 23,675 | 1,319 | 5.57% | [5.29, 5.87] |
| `> 100` | 94,456 | 6,634 | 7.02% | [6.86, 7.19] |

Si fueran ingresos reales de \$0, estas personas serían las más pobres del
conjunto y deberían superar el **8.35%** del decil de menor ingreso. Están en
2–4%, por debajo incluso de los faltantes.

**Por qué dejarlo como número sería grave:** un valor de 0 colocaría a estas
filas en el extremo de mayor pobreza, donde el modelo espera **riesgo alto**,
cuando el riesgo real es 4.16%. **Señal invertida.**

#### (4) No son la misma población que los faltantes

| Grupo | edad mediana | líneas | `RevUtil` mediana | `NumberOfDependents` faltante |
|---|---|---|---|---|
| `MonthlyIncome = 0` | 47 | 6 | 0.108 | **0.0%** |
| `MonthlyIncome = 1` | 45 | 6 | 0.122 | **0.0%** |
| `MonthlyIncome` faltante | 57 | 6 | 0.081 | **13.2%** |
| `MonthlyIncome > 100` | 51 | 8 | 0.178 | 0.0% |

Comparten la patología del `DebtRatio` pero **son grupos distintos**: diez años
más jóvenes y, sobre todo, a estos registros nunca les falta
`NumberOfDependents`, mientras que al 13.2% de los faltantes sí.

**Consecuencia de diseño [IMPL]:** justifica mantener `flag_income_cero`
**separado** de `flag_income_faltante`. Parecen el mismo fenómeno pero proceden
de orígenes distintos.

#### (5) ¿Separar el `0` del `1`?

Las tasas puntuales difieren (4.16% frente a 2.41%, razón de momios 1.76), pero
**no son distinguibles**: prueba exacta de Fisher $p = 0.093$, e intervalos que
se solapan entre 3.21% y 4.16%. Con 12 eventos en el grupo del `1` no hay base
para fragmentar. Un solo indicador cubre ambos.

#### (6) Zona gris

Las 49 filas con $1 < \text{MonthlyIncome} \le 100$ se comportan como los
códigos (`DebtRatio` mediano 20.5, el 75.5% por encima de 2), pero 49 registros
no sostienen un cambio de umbral. Se documentan y se tratan como ingreso válido.

### 3.2.4 Síntesis del patrón

| Variable | Valor engañoso | Riesgo real | Riesgo que el modelo habría inferido |
|---|---|---|---|
| Morosidad | `96` / `98` | 54.67% | "98 episodios" → extremo |
| `RevolvingUtilization` | `> 100` | ~6.4% | "utilización altísima" → extremo |
| `MonthlyIncome` | `0` o `1` | ~3.7% | "ingreso nulo" → extremo |

**Tratamiento uniforme [IMPL]:** convertir a `NaN` + crear indicador. Nunca
capear, nunca eliminar la fila.

---

## 3.3 `DebtRatio`: dos poblaciones en una columna

**[MEDIDO]** Los percentiles revelan un acantilado:

```
p50:      0.37
p75:      0.86     ← todavía es un ratio
p90:  1,263.00     ← salto de tres órdenes de magnitud
p95:  2,441.00
```

| Corte | n | % de la muestra |
|---|---|---|
| $> 1$ | 27,987 | **23.32%** |
| $> 100$ | 19,444 | 16.20% |
| $> 1{,}000$ | 13,495 | 11.25% |

**Mecanismo identificado [MEDIDO]:** de las 27,987 filas con `DebtRatio > 1`,
**22,200 (79.3%) no tienen `MonthlyIncome`**. Sin denominador, el sistema
almacenó el numerador (deuda mensual bruta) en lugar del ratio.

De las 5,787 restantes que sí tienen ingreso, **1,691 tienen
`MonthlyIncome ≤ 1`** — mismo problema por otra vía.

### 3.3.1 Régimen válido [MEDIDO]

| Banda | n | Tasa de evento |
|---|---|---|
| 0 – 0.5 | 75,036 | 6.05% |
| 0.5 – 1 | 16,977 | 9.88% |
| **1 – 2** | 3,269 | **13.12%** ← máximo |
| 2 – 5 | 1,089 | 7.16% |
| 5 – 10 | 610 | 4.75% |
| > 1,000 | 13,495 | 4.89% |

De 0 a 2 la señal es monótona creciente; a partir de 2 se vuelve ruido que rebota
entre 4% y 8%. Ese patrón motivó un primer umbral `DebtRatio > 2`, que fue
sustituido después por el criterio directo (§3.3.2).

### 3.3.2 De un umbral de magnitud al criterio del denominador [MEDIDO]

El umbral `> 2` era un **proxy**. Lo que distingue un ratio de un importe no es
su tamaño sino si existía denominador: sin `MonthlyIncome` no hay división
posible y el campo guarda el numerador (§3.2.3, prueba del denominador).

Las dos reglas discrepan en dos zonas, y una de ellas costaba señal:

| Zona | n | Trato con el proxy | ¿Correcto? |
|---|---|---|---|
| Con ingreso válido y `DebtRatio > 2` | **955** | descartadas como dato corrupto | **no** |
| Sin denominador y `DebtRatio ≤ 2` | 1,732 | tratadas como ratio válido | inocuo |

Las 955 son ratios auténticos de personas con ingreso bajo, y **predicen por
encima de la media**:

| Tramo | n | Ingreso mediano | Deuda implícita | Tasa de evento |
|---|---|---|---|---|
| `DebtRatio` 2–5 | 700 | $1,300 | $3,678 | **9.43%** |
| `DebtRatio` 5–10 | 152 | $639 | $4,189 | 7.24% |
| `DebtRatio` 10–100 | 92 | $400 | $5,194 | 8.70% |

Las 1,732 de la otra zona son inocuas: el 82.4% tiene `DebtRatio` exactamente 0,
que vale lo mismo interpretado como ratio o como importe.

**Tratamiento [IMPL]:** se separa en dos columnas complementarias —
`DebtRatio_limpio` donde existía denominador, `deuda_bruta_proxy` donde no— en lugar
de capear:

```python
sin_denominador = (df["flag_income_faltante"] | df["flag_income_cero"]).astype(bool)
df["deuda_bruta_proxy"]      = df["DebtRatio"].where(sin_denominador, np.nan)
df["DebtRatio_limpio"] = df["DebtRatio"].where(~sin_denominador, np.nan)
```

**`flag_debtratio_invalido` fue eliminado.** Bajo esta regla equivaldría
exactamente a `flag_income_faltante | flag_income_cero`: una combinación lineal
de columnas existentes, es decir colinealidad perfecta — el mecanismo que
produjo [H1](07_auditoria.md#h1).

### 3.3.3 ¿Se comporta igual con y sin denominador? [MEDIDO]

Que el campo cambie de *significado* no bastaría por sí solo para separarlo: si
la magnitud fuese lo único distinto, reescalar resolvería el problema. La
pregunta que decide es si **ordena el riesgo** de la misma forma en ambos
grupos (`01_eda.py`, sección 5c).

**Las distribuciones apenas se solapan.** El percentil 10 del grupo sin
denominador (10.00) ya supera el percentil 99 del grupo con denominador (2.01):

| Percentil | Con ingreso | Sin ingreso |
|---|---|---|
| p10 | 0.02 | 10.00 |
| p50 | 0.29 | 1,131.00 |
| p90 | 0.71 | 3,751.80 |
| p99 | 2.01 | 8,156.08 |
| máx | 450 | 329,664 |

**El gradiente de riesgo es la prueba decisiva.** Se mide con la correlación de
Spearman entre el índice de decil y la tasa de evento — solo mira el orden, por
lo que es inmune al cambio de unidad:

| Grupo | n | Rango de tasas por decil | Spearman | $p$ |
|---|---|---|---|---|
| **Con denominador** | 94,505 | 5.05% → 12.15% | **+0.709** | 0.022 |
| **Sin denominador** | 25,495 | 4.10% – 8.43% (sin dirección) | **−0.042** | 0.907 |

Con denominador, `DebtRatio` ordena el riesgo de forma creciente en su mitad
superior: es apalancamiento y se comporta como tal. Sin denominador, el importe
**no ordena nada** — la tasa rebota sin dirección y el decil más alto (hasta
\$329,664 de deuda mensual) vuelve prácticamente a la media global.

**Confirmación independiente:** el Information Value de `deuda_bruta_proxy` es
**0.0136**, por debajo del umbral de 0.02 que Siddiqi clasifica como *inútil*.
Es la última variable del ranking de las 17.

**Consecuencia [IMPL]:** son dos variables distintas compartiendo columna.
Mezclarlas obligaría al modelo a aprender una relación monótona que solo existe
en una de las dos poblaciones. `deuda_bruta_proxy` se conserva en la rama de árboles
—no estorba y documenta la separación— pero no aporta señal; el modelo logístico
no la usa.

---

## 3.4 Valores faltantes

### 3.4.1 Magnitud y estructura [MEDIDO]

| Variable | n faltantes | % |
|---|---|---|
| `MonthlyIncome` | 23,675 | 19.73% |
| `NumberOfDependents` | 3,128 | 2.61% |

**Hallazgo estructural:** el **100%** de las filas sin `NumberOfDependents`
tampoco tiene `MonthlyIncome`. No es ausencia variable por variable: son
**registros incompletos** procedentes de un canal o sistema distinto.

### 3.4.2 El faltante es informativo — y en dirección contraintuitiva [MEDIDO]

| Grupo | n | Tasa de evento |
|---|---|---|
| Nada falta | 96,325 | **6.96%** |
| Solo falta `MonthlyIncome` | 20,547 | **5.75%** |
| Faltan ambos | 3,128 | **4.41%** |

**Gradación monótona:** cuanto más incompleto el registro, **menor** el riesgo.

**Implicaciones de diseño [IMPL]:**

1. Se crean **dos indicadores separados**, no uno. Un único indicador colapsaría
   5.75% y 4.41% en el mismo valor, perdiendo la gradación.
2. La mediana de `NumberOfDependents` es **0** (y la moda también, con 69,494 de
   116,872 observaciones). Imputar con la mediana y con cero son la misma
   operación; la objeción metodológica de "no asumir cero" es correcta en
   principio pero aquí resulta numéricamente vacua.

### 3.4.3 Por qué no se usa imputación sofisticada (KNN, MICE)

**[RECOM]** Tres razones:

1. **Dos de los tres modelos no la necesitan.** XGBoost y LightGBM manejan `NaN`
   nativamente y aquí el faltante es predictivo (§2.6.4).
2. **El mecanismo es MNAR y de nivel registro.** El 100% de solapamiento indica
   que la ausencia responde al canal de captura, no a las características del
   individuo. KNN imputaría buscando "vecinos parecidos" usando precisamente las
   variables que también están degradadas en esos registros.
3. **Coste computacional.** `KNNImputer` sobre $1.5\times10^5$ filas requiere del
   orden de $10^{10}$ cálculos de distancia.

**[RECOM]** La postura defendible es: línea base con mediana + indicador, y KNN
como **experimento medido** contra el AUC de validación, no como supuesto.

---

## 3.5 Relación variable–objetivo

### 3.5.1 Por qué la correlación de Pearson engaña

**[MEDIDO]** Para `RevolvingUtilizationOfUnsecuredLines`:

| Medida | Valor |
|---|---|
| Pearson (todos los datos) | **−0.0018** |
| Pearson (solo filas $\leq 1$) | 0.2494 |
| Spearman (por rangos) | 0.2404 |

Y el análisis por deciles:

$$2.52 \to 1.36 \to 1.40 \to 1.91 \to 2.37 \to 3.47 \to 5.25 \to 8.80 \to 16.59 \to 23.17\ \%$$

La variable **más predictiva del conjunto** aparecía como ruido bajo Pearson
(−0.002), únicamente porque 223 filas tienen valores absurdos.

**[TEORÍA]** Pearson mide asociación **lineal** y su estimador es
extraordinariamente sensible a outliers, porque los momentos de segundo orden
crecen con el cuadrado de la desviación. Por eso la herramienta estándar en
credit scoring es el análisis por bins, no la correlación.

### 3.5.2 Relaciones por valor exacto en las variables de conteo [MEDIDO]

| Valor | 30-59 días | 90+ días | 60-89 días |
|---|---|---|---|
| 0 | 4.00% | 4.63% | 5.10% |
| 1 | 15.03% | **33.66%** | 31.01% |
| 2 | 26.51% | 49.90% | 50.18% |
| 3 | 35.23% | 57.72% | 56.60% |
| 4 | 42.57% | 67.01% | 61.90% |
| 5 | 45.03% | 63.36% | 61.76% |

**Dos observaciones estructurales:**

1. **El salto 0→1 difiere radicalmente entre gravedades.** En mora leve el riesgo
   se multiplica por 3.8; en mora grave por **7.3**. Las tres variables aportan
   información **distinta**, no redundante — lo que justifica conservarlas.
2. **La relación satura.** A partir de 4–5 episodios la tasa se estanca en
   50%–65% con celdas de muy poca masa. Esto motiva agrupar el extremo superior
   en lugar de tratar el conteo como variable continua.

---

## 3.6 Ingeniería de variables

### 3.6.1 Las siete derivadas

Documentadas en [§1.4.1](01_arquitectura.md). Cada una responde a una pregunta
que un analista de riesgo formularía; ninguna es una combinación arbitraria.

### 3.6.2 Veredicto empírico: no aportaron nada [MEDIDO]

| Rama | Con las 7 derivadas | Sin ellas | Diferencia |
|---|---|---|---|
| Regresión logística (AUC val.) | 0.86300 | 0.86272 | −0.00028 |
| XGBoost (AUC early-stop) | 0.86350 | 0.86405 | **+0.00055** |

*(Medición del momento en que se tomó la decisión; el modelo lineal de referencia
tenía entonces 18 variables. La auditoría posterior lo redujo a 10 — ver
[E16](05_experimentos.md#e16--corrección-de-h1-coherencia-completa-de-signos).)*

En la rama lineal, además, **causaban 4 coeficientes con signo invertido**. En la
rama de árboles no rompían nada pero tampoco mejoraban.

**Lección [RECOM]:** la ingeniería de variables puede restar. Siete variables
construidas con lógica de negocio sólida no aportaron poder predictivo porque la
información ya estaba en las diez originales. El coste de descubrirlo fue bajo;
el coste de *no* descubrirlo habría sido un scorecard indefendible.

---

## 3.7 Estrategia de partición

### 3.7.1 Split único, dos ramas

```
raw (150,000)
      │
      ├─ split estratificado 80/20 (semilla 42) ──┐
      │                                            │
   train (120,000)                          validación (30,000)
      │                                            │
      ├── limpieza + variables (reglas fijas) ─────┤
      │                                            │
      ├─→ rama árboles ──→ trees/train,val,test    │
      └─→ rama WoE ─────→ woe/train,val,test       │
```

**Verificaciones [MEDIDO]:**

```
trees/train vs woe/train  →  índices idénticos: True
intersección train ∩ val  →  0 individuos
train + val               →  150,000 = 150,000
```

**Por qué un solo split [RECOM]:** con splits independientes por rama, cada
modelo se evaluaría sobre personas distintas y la comparación no significaría
nada.

**Por qué estratificado:** con $\pi = 6.68\%$, un split simple podría producir
proporciones distintas entre particiones. La estratificación garantiza
$\pi_{\text{train}} = \pi_{\text{val}} = 6.68\%$ exacto **[MEDIDO]**.

### 3.7.2 Sub-particiones internas

Cada script de entrenamiento aparta un conjunto adicional **del propio
entrenamiento** para tomar decisiones:

| Script | Ajuste | Decisión | Para qué |
|---|---|---|---|
| `03` | 96,000 | 24,000 | Elegir el subconjunto de variables |
| `04` | 96,000 | 24,000 | *Early stopping* + hiperparámetros |
| `05` | 96,000 | 24,000 | Ídem (mismo split, misma semilla) |

**El conjunto de validación (30,000) no participa en ninguna decisión.** Se toca
una sola vez, al final, para reportar.

**[IMPL]** `05` usa deliberadamente el mismo `random_state=42` que `04`, de modo
que cualquier diferencia de AUC entre XGBoost y LightGBM provenga del algoritmo
y no del reparto.

---

## 3.8 Diseño de los bins WoE

### 3.8.1 Variables de conteo — cortes manuales

```python
BINS_CONTEO = {
    "NumberOfTime30-59DaysPastDueNotWorse": [-0.1, 0, 1, 3, np.inf],
    "NumberOfTime60-89DaysPastDueNotWorse": [-0.1, 0, 1, np.inf],
    "NumberOfTimes90DaysLate":              [-0.1, 0, 1, 3, np.inf],
    "NumberOfDependents":                   [-0.1, 0, 1, 2, np.inf],
    ...
}
```

**Por qué no cuantiles [IMPL]:** `pd.qcut` falla cuando hay demasiados valores
repetidos. Con 94.4% de ceros en `NumberOfTimes90DaysLate`, no existen 10 cortes
distintos posibles y la función colapsa a un único bin. Es exactamente lo que
ocurrió en la primera versión del EDA.

**Estructura elegida:** $\{0\},\ \{1\},\ \{2,3\},\ \{4+\}$ — captura el salto
0→1 (el mayor) como bin propio, y agrupa la cola donde la relación satura.

### 3.8.2 Variables continuas — cortes derivados del análisis

```python
"RevolvingUtilizationOfUnsecuredLines": [-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, np.inf],
"age":                                  [-0.1, 30, 40, 45, 50, 55, 60, 65, 72, np.inf],
"MonthlyIncome":                        [-0.1, 2000, 3000, 4000, 5400, 7000, 9000, 12000, np.inf],
"DebtRatio_limpio":                     [-0.1, 0.2, 0.4, 0.6, 1.0, np.inf],
```

Los cortes se sitúan **donde cambia el riesgo**, no en percentiles regulares. Por
ejemplo, `RevolvingUtilization` tiene un corte explícito en 0.9 y otro en 1.0
porque ahí está el *mass point* de 11,778 personas al 98%–100% de su límite —un
grupo real, generado por el techo que impone el banco, con 19.31% de evento.

### 3.8.3 Tabla WoE resultante para la variable más predictiva [MEDIDO]

| Bin | n | % pobl. | Tasa evento | WoE |
|---|---|---|---|---|
| 0 – 0.1 | 51,687 | 43.07% | 1.84% | **+1.3401** |
| 0.1 – 0.3 | 22,660 | 18.88% | 3.11% | +0.8031 |
| 0.3 – 0.5 | 12,673 | 10.56% | 5.74% | +0.1609 |
| 0.5 – 0.7 | 9,125 | 7.60% | 9.62% | −0.3968 |
| 0.7 – 0.9 | 7,889 | 6.57% | 15.17% | −0.9155 |
| 0.9 – 1.0 | 13,326 | 11.10% | 19.31% | −1.2063 |
| > 1.0 | 2,468 | 2.06% | 39.67% | **−2.2171** |
| **FALTANTE** | 172 | 0.14% | 6.40% | **+0.0059** |

**El bin FALTANTE valida la decisión de umbral.** Su WoE es esencialmente cero:
las 172 filas invalidadas se comportan de forma indistinguible del promedio
poblacional, que es exactamente lo que se esperaba de ruido puro. Si se hubieran
capeado en lugar de invalidado, habrían caído en el bin `> 1.0` con WoE −2.22,
es decir etiquetadas como riesgo máximo teniendo riesgo medio.

### 3.8.4 Limitación del binning manual

**[RECOM]** Los cortes fueron elegidos por inspección del EDA, no por un
algoritmo de binning óptimo. Alternativas establecidas:

| Método | Ventaja | Desventaja |
|---|---|---|
| **Manual (implementado)** | Incorpora conocimiento del dominio; cortes interpretables | Subjetivo, no reproducible por otro analista |
| Árbol de decisión monótono | Automático, maximiza IV, garantiza monotonía | Puede producir cortes sin sentido de negocio |
| ChiMerge | Fundamento estadístico ($\chi^2$) | Sensible al umbral de significación |
| Binning óptimo (MILP) | Óptimo garantizado bajo restricciones | Coste computacional; requiere librería especializada |

**Recomendación para producción [RECOM]:** contrastar el binning manual contra
un binning monótono automático (`optbinning`) y documentar la diferencia. Si
coinciden, refuerza ambos; si difieren, obliga a justificar la elección.

---

## 3.9 Verificaciones automáticas del pipeline

`02_preprocessing.py` ejecuta **14 comprobaciones** al terminar:

| Categoría | Verificación |
|---|---|
| Integridad del split | Train y validación no comparten individuos |
| | No se pierde ninguna fila (120,000 + 30,000 = 150,000) |
| Consistencia entre ramas | trees y woe cubren los mismos individuos (train, val, test) |
| Contrato de columnas | Mismas columnas y mismo orden en las tres particiones, ambas ramas |
| Completitud de test | Ambas ramas conservan las 101,503 filas |
| Aptitud numérica | Conjunto WoE sin `NaN` ni infinitos |
| Anti-fuga | El objetivo no aparece entre las características |
| **Reproducibilidad** | **El artefacto recargado desde disco reproduce la transformación** |

La última es la más importante: convierte una promesa ("el artefacto sirve para
producción") en una aserción ejecutable.

---

[← Fundamentos](02_fundamentos_matematicos.md) · [Índice](README.md) · [Siguiente: Modelado →](04_modelado.md)

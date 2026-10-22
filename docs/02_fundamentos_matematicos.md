# 02 — Fundamentos matemáticos, estadísticos y financieros

[← Arquitectura](01_arquitectura.md) · [Índice](README.md) · [Siguiente: Pipeline de datos →](03_pipeline_datos.md)

---

## 2.1 El problema de credit scoring

### 2.1.1 Definición formal

Sea $\mathbf{X} \in \mathbb{R}^{p}$ el vector de características observables de
un solicitante en el momento de la decisión $t_0$, e

$$Y = \mathbb{1}\{\text{el solicitante alcanza 90+ días de mora en } (t_0,\, t_0+24\text{ meses}]\}$$

la variable objetivo. El sistema estima

$$\boxed{\ \text{PD}(\mathbf{x}) = P(Y=1 \mid \mathbf{X}=\mathbf{x})\ }$$

**Variables:**

| Símbolo | Significado | Dominio |
|---|---|---|
| $\mathbf{X}$ | Vector de características | $\mathbb{R}^{p}$, aquí $p = 10$ originales |
| $Y$ | Indicador de incumplimiento | $\{0, 1\}$ |
| $t_0$ | Momento de la decisión de crédito | — |
| $\text{PD}$ | Probability of Default | $[0,1]$ |

### 2.1.2 La distinción crítica: ventana de observación vs. ventana de desempeño

**[TEORÍA]** El credit scoring separa dos ventanas temporales:

```
   VENTANA DE OBSERVACIÓN            │      VENTANA DE DESEMPEÑO
   (de dónde salen las X)            │      (de dónde sale Y)
 ────────────────────────────────────┼──────────────────────────────►
   NumberOfTimes90DaysLate = 1       │   ¿Alcanzará 90+ días de mora
   RevolvingUtilization = 0.85       │    en los próximos 24 meses?
   age = 45                          │
                                    t₀
```

Esta distinción no es un tecnicismo: **determina si el modelo tiene sentido**.

Si $\mathbf{X}$ e $Y$ describieran el mismo evento, el problema sería trivial y
la estimación inútil. La verificación empírica **[MEDIDO]**:

| Comprobación | Resultado |
|---|---|
| Individuos con $Y=1$ | 8,021 |
| …de los cuales tenían `NumberOfTimes90DaysLate = 0` | **5,226 (65.2%)** |
| Individuos con `NumberOfTimes90DaysLate ≥ 1` (sin centinelas) | 6,492 |
| …de los cuales tuvieron $Y=0$ | **3,814 (58.7%)** |

Si ambas variables midieran lo mismo, ninguna de las dos cifras sería posible.
El 33.66% de default entre quienes tienen exactamente un episodio previo de 90+
días es una **tasa de reincidencia**, no una tautología.

**Implicación para el diseño [RECOM]:** el 65.4% de los eventos ocurre en
individuos con historial de 90+ días limpio. El valor del modelo reside
precisamente en detectarlos mediante otras señales — principalmente la
utilización de crédito, que capta deterioro financiero *antes* de que se
materialice en mora formal.

### 2.1.3 Correspondencia con la definición regulatoria de default

**[TEORÍA]** El Comité de Basilea define incumplimiento (BCBS, *Basel II
Framework*, §452) como la ocurrencia de al menos uno de:

1. El banco considera improbable que el deudor pague íntegramente sus
   obligaciones (*unlikeliness to pay*);
2. El deudor presenta **más de 90 días de mora** sobre una obligación material.

`SeriousDlqin2yrs` implementa el segundo criterio con horizonte de 24 meses. Es
por tanto un *proxy* razonable de default regulatorio, con dos salvedades
**[RECOM]**:

- No captura el criterio de *unlikeliness to pay* (quiebras, reestructuraciones
  forzosas sin mora).
- El horizonte de 24 meses excede el horizonte de 12 meses que Basilea III usa
  para la PD de capital regulatorio, aunque coincide con el concepto de PD
  *lifetime* de IFRS 9 para instrumentos de corto plazo.

---

## 2.2 Weight of Evidence

### 2.2.1 Definición

**[TEORÍA]** Para una variable $X$ discretizada en $J$ intervalos (*bins*), el
Weight of Evidence del intervalo $j$ es

$$\boxed{\ \text{WoE}_j = \ln\!\left(\frac{P(X \in B_j \mid Y = 0)}{P(X \in B_j \mid Y = 1)}\right) = \ln\!\left(\frac{n_{0j}/N_0}{n_{1j}/N_1}\right)\ }$$

**Variables:**

| Símbolo | Significado |
|---|---|
| $B_j$ | Intervalo $j$-ésimo |
| $n_{0j}$, $n_{1j}$ | No-eventos y eventos dentro de $B_j$ |
| $N_0$, $N_1$ | Totales de no-eventos y eventos |
| $\text{WoE}_j > 0$ | El intervalo concentra proporcionalmente más no-eventos → **menos riesgo** |
| $\text{WoE}_j < 0$ | El intervalo concentra proporcionalmente más eventos → **más riesgo** |

### 2.2.2 Derivación: por qué WoE está en escala de log-odds

Aplicando el teorema de Bayes a las odds posteriores dentro del intervalo $B_j$:

$$\frac{P(Y=0 \mid X \in B_j)}{P(Y=1 \mid X \in B_j)}
= \frac{P(X \in B_j \mid Y=0)}{P(X \in B_j \mid Y=1)} \cdot \frac{P(Y=0)}{P(Y=1)}$$

Tomando logaritmos:

$$\underbrace{\ln\frac{P(Y=0\mid B_j)}{P(Y=1\mid B_j)}}_{\text{log-odds posteriores}}
= \underbrace{\text{WoE}_j}_{\text{aporte del intervalo}}
+ \underbrace{\ln\frac{N_0}{N_1}}_{\text{log-odds a priori}}$$

**Interpretación:** el WoE es exactamente **cuánto desplaza ese intervalo los
log-odds respecto a la tasa base de la población**. Ésa es la razón por la que
encaja con la regresión logística, que también es lineal en log-odds (§2.3.2):
se le entregan al modelo variables ya expresadas en su propia escala natural.

### 2.2.3 Suavizado de Laplace

**[IMPL]** Si $n_{0j}=0$ o $n_{1j}=0$, el WoE diverge. La implementación aplica
una corrección aditiva con $\alpha = 0.5$:

$$\widehat{\text{WoE}}_j = \ln\!\left(\frac{(n_{0j} + \alpha)/N_0}{(n_{1j} + \alpha)/N_1}\right), \qquad \alpha = 0.5$$

```python
g["dist_eventos"]    = (g["eventos"]    + suavizado) / total_eventos
g["dist_no_eventos"] = (g["no_eventos"] + suavizado) / total_no_eventos
g["woe"] = np.log(g["dist_no_eventos"] / g["dist_eventos"])
```

$\alpha = 0.5$ corresponde al *prior* de Jeffreys para la binomial
$\text{Beta}(1/2, 1/2)$ **[TEORÍA]**. Alternativas: $\alpha=1$ (Laplace clásico,
más conservador) o suavizado proporcional al tamaño del bin.

**Nota de precisión [IMPL]:** los denominadores $N_0, N_1$ no se corrigen, luego
$\sum_j \widehat{P}_j$ excede ligeramente 1. El sesgo es de orden
$J\alpha/N \approx 5 \times 10^{-5}$ con $J{=}10$ y $N{=}10^5$ — despreciable,
pero es una desviación de la forma canónica.

### 2.2.4 Ventajas del WoE frente a alternativas

| Propiedad | WoE | One-hot | Valor crudo | Target encoding |
|---|---|---|---|---|
| Captura no-linealidad | ✅ | ✅ | ❌ | ✅ |
| Robusto a outliers | ✅ (solo cuenta el bin) | ✅ | ❌ | ✅ |
| Trata `NaN` sin imputar | ✅ (bin propio) | ✅ | ❌ | parcial |
| Dimensionalidad | 1 columna | $J$ columnas | 1 | 1 |
| Escala interpretable | ✅ log-odds | ❌ | ❌ | ❌ |
| Riesgo de sobreajuste | bajo | medio | bajo | **alto** |

**[RECOM]** El *target encoding* sin validación cruzada anidada sobreajusta
gravemente; el WoE mitiga ese riesgo al agrupar en bins con masa suficiente
antes de calcular la estadística.

---

## 2.3 Information Value

### 2.3.1 Definición

$$\boxed{\ \text{IV} = \sum_{j=1}^{J}\left(\frac{n_{0j}}{N_0} - \frac{n_{1j}}{N_1}\right)\text{WoE}_j\ }$$

**[TEORÍA]** El IV es la **divergencia de Jeffreys** entre las distribuciones
condicionales $P(X \mid Y=0)$ y $P(X \mid Y=1)$:

$$\text{IV} = D_{\text{KL}}\!\left(P_{X\mid Y=0} \,\|\, P_{X\mid Y=1}\right) + D_{\text{KL}}\!\left(P_{X\mid Y=1} \,\|\, P_{X\mid Y=0}\right)$$

Es simétrica y no negativa. Mide cuánto se separan las dos poblaciones a lo
largo de esa variable.

### 2.3.2 Escala convencional

**[TEORÍA]** (Siddiqi, *Credit Risk Scorecards*, 2006):

| IV | Interpretación |
|---|---|
| $< 0.02$ | Inútil |
| $0.02 – 0.10$ | Débil |
| $0.10 – 0.30$ | Medio |
| $0.30 – 0.50$ | Fuerte |
| $> 0.50$ | Sospechoso — investigar posible fuga |

### 2.3.3 Limitación fundamental del IV

**[TEORÍA]** El IV mide cada variable **de forma aislada**. Ignora por completo
la información compartida entre predictores.

**Manifestación en este proyecto [MEDIDO]:** las seis primeras posiciones del
ranking de IV están ocupadas por variables de morosidad que son
reempaquetamientos unas de otras:

| Variable | IV | Relación |
|---|---|---|
| `total_atrasos` | 1.4228 | $\sum$ de las tres originales |
| `peor_atraso` | 1.4052 | $\max$ de gravedad de las tres |
| `RevolvingUtilization` | 1.1463 | genuinamente distinta |
| `NumberOfTimes90DaysLate` | 0.8852 | original |
| `NumberOfTime30-59Days` | 0.7446 | original |
| `NumberOfTime60-89Days` | 0.6026 | original |

`flag_algun_atraso` coincide con $\mathbb{1}\{\text{total\_atrasos} > 0\}$ en
**120,000 de 120,000 filas (100%)**.

**Analogía:** cinco testigos entrevistados por separado que coinciden con gran
detalle puntúan altísimo cada uno — hasta que se descubre que todos lo oyeron de
la misma persona. No hay cinco testimonios; hay uno repetido cinco veces.

**Consecuencia práctica [RECOM]:** el IV nunca debe usarse como criterio único
de selección. Debe combinarse con un diagnóstico de colinealidad (VIF, §2.5) o
con métodos que evalúen conjuntos de variables (selección por pasos, LASSO).

### 2.3.4 La falsa alarma de IV > 0.50

Seis variables superan el umbral de "sospechoso". La regla es una heurística,
no un teorema. En este caso la investigación (§2.1.2) descarta la fuga: el
historial pasado predice el futuro **muy bien**, lo cual es distinto de
contenerlo.

**Lección metodológica [RECOM]:** cuando una regla heurística se dispara, se
investiga; no se obedece ciegamente ni se ignora.

---

## 2.4 Regresión logística

### 2.4.1 El modelo

**[TEORÍA]** Combinación lineal de las características:

$$z = \beta_0 + \sum_{i=1}^{p}\beta_i x_i = \beta_0 + \boldsymbol{\beta}^{\top}\mathbf{x}$$

Como $z \in \mathbb{R}$ y una probabilidad debe estar en $[0,1]$, se aplica la
función logística (sigmoide):

$$\boxed{\ \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad P(Y=1\mid\mathbf{x}) = \sigma\!\left(\beta_0 + \boldsymbol{\beta}^{\top}\mathbf{x}\right)\ }$$

**Propiedades de $\sigma$:**

$$\sigma: \mathbb{R} \to (0,1), \qquad \sigma(0) = \tfrac12, \qquad \sigma'(z) = \sigma(z)\left(1 - \sigma(z)\right)$$

**Por qué no regresión lineal:** $\beta_0 + \boldsymbol{\beta}^\top\mathbf{x}$
puede valer $-17$ o $340$. Predeciría "esta persona tiene 340% de probabilidad
de incumplir".

### 2.4.2 Linealidad en log-odds

Despejando:

$$\boxed{\ \operatorname{logit}(p) = \ln\!\frac{p}{1-p} = \beta_0 + \sum_i \beta_i x_i\ }$$

La cantidad $p/(1-p)$ son las **odds**. Su logaritmo, los **log-odds**.

**Interpretación de un coeficiente [TEORÍA]:** $\beta_i$ es el cambio en los
log-odds por unidad de $x_i$, manteniendo el resto constante. Equivalentemente,
$e^{\beta_i}$ es el **odds ratio**: multiplicar por $e^{\beta_i}$ las odds del
evento.

**En este proyecto** $x_i$ es un valor WoE, también en log-odds. Un
$\beta_i \approx -1$ significa que el modelo adopta el WoE prácticamente sin
reescalar — señal de que el binning captó bien la relación.

### 2.4.3 Estimación: máxima verosimilitud

La verosimilitud de la muestra $\{(\mathbf{x}_k, y_k)\}_{k=1}^{n}$ bajo
independencia condicional:

$$\mathcal{L}(\boldsymbol{\beta}) = \prod_{k=1}^{n} p_k^{\,y_k}\,(1-p_k)^{1-y_k},
\qquad p_k = \sigma\!\left(\boldsymbol{\beta}^{\top}\mathbf{x}_k\right)$$

Se maximiza el logaritmo, equivalentemente se minimiza la **log-loss**:

$$\boxed{\ \ell(\boldsymbol{\beta}) = -\sum_{k=1}^{n}\Big[y_k \ln p_k + (1-y_k)\ln(1-p_k)\Big]\ }$$

**Gradiente** (derivación): usando $\partial p_k/\partial\beta_i = p_k(1-p_k)x_{ki}$,

$$\frac{\partial \ell}{\partial \beta_i} = \sum_{k=1}^{n}\left(p_k - y_k\right)x_{ki}$$

El gradiente tiene la forma elegante *(predicción − realidad) × característica*.

**Matriz Hessiana:**

$$H_{ij} = \frac{\partial^2 \ell}{\partial\beta_i\partial\beta_j} = \sum_{k} p_k(1-p_k)\,x_{ki}x_{kj}
\quad\Longrightarrow\quad H = \mathbf{X}^{\top}\mathbf{W}\mathbf{X} \succeq 0$$

con $\mathbf{W} = \operatorname{diag}(p_k(1-p_k))$. Como $H$ es semidefinida
positiva, **la log-loss es convexa** y el óptimo es global **[TEORÍA]**.

**[IMPL]** scikit-learn usa el algoritmo L-BFGS (cuasi-Newton con memoria
limitada). Converge en decenas de iteraciones frente a un máximo de 2,000
**[MEDIDO]** — muy holgado, lo que confirma un problema bien condicionado.

### 2.4.4 Regularización

**[TEORÍA]** La penalización $L_2$ (ridge) minimiza

$$\ell_{\text{reg}}(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) + \frac{1}{2C}\|\boldsymbol{\beta}\|_2^2$$

donde $C > 0$ es el inverso de la fuerza de regularización: $C$ grande = poca
penalización.

**Resultado empírico [MEDIDO]:**

| $C$ | AUC validación | $\max\lvert\beta\rvert$ |
|---|---|---|
| 0.001 | 0.86013 | 0.584 |
| 0.01 | 0.86078 | 0.647 |
| 0.1 | 0.86024 | 0.817 |
| **1.0 (usado)** | **0.86059** | 0.840 |
| 10.0 | 0.86014 | 0.843 |
| 100.0 | 0.86014 | 0.843 |

La mejor configuración supera a la usada en $6.3\times10^{-4}$ — dentro del ruido
de muestreo, que es de $\pm 0.0036$ (§2.12.1).

**Por qué no importa [TEORÍA]:** con $n = 120{,}000$ y $p = 10$, hay **12,000
observaciones por parámetro**. El régimen asintótico domina y la varianza del
estimador es despreciable frente al sesgo que introduciría la regularización.
Además el AUC de entrenamiento (0.85719) es **menor** que el de validación
(0.86059), lo que descarta sobreajuste.

**[RECOM]** Saber qué hiperparámetros *no* ajustar ahorra tanto tiempo como
saber cuáles sí. Es habitual ver proyectos que dedican horas a barrer $C$
mientras arrastran un código centinela que invierte la señal.

---

## 2.5 Multicolinealidad y VIF

### 2.5.1 El problema

**[TEORÍA]** Si dos predictores son casi combinación lineal uno del otro, la
matriz $\mathbf{X}^\top\mathbf{W}\mathbf{X}$ se vuelve casi singular. Su inversa
—que da la matriz de covarianzas de $\hat{\boldsymbol{\beta}}$— tiene entradas
enormes: los coeficientes se vuelven inestables y pueden invertir de signo.

**Ejemplo canónico:** predecir el peso usando `altura_cm` y `altura_pulgadas`.
Infinitas combinaciones $(\beta_1, \beta_2)$ con
$\beta_1 + 2.54\beta_2 = \text{const}$ producen predicciones idénticas.

**Punto crítico:** la multicolinealidad **no degrada necesariamente el poder
predictivo**. Destruye la *interpretabilidad* y la *estabilidad*. En credit
scoring eso importa tanto como el acierto, porque hay que justificar cada
rechazo ante un supervisor y ante el cliente.

### 2.5.2 Variance Inflation Factor

$$\boxed{\ \text{VIF}_i = \frac{1}{1 - R_i^2}\ }$$

donde $R_i^2$ es el coeficiente de determinación al regresar $x_i$ contra todas
las demás variables.

**Interpretación:** $\text{VIF}_i$ es el factor por el que se infla la varianza
de $\hat\beta_i$ respecto al caso de predictores ortogonales.

| VIF | Lectura convencional |
|---|---|
| 1 | Ortogonal |
| > 5 | Preocupante |
| > 10 | Redundancia seria |

**[MEDIDO]** Con las 25 variables:

| Variable | $R^2$ | VIF |
|---|---|---|
| `woe_peor_atraso` | 0.9875 | **79.95** |
| `flag_algun_atraso` | 0.9825 | **57.01** |
| `woe_total_atrasos` | 0.9756 | **41.03** |
| `woe_NumberOfTime30-59Days` | 0.9207 | 12.61 |
| … | | |
| `woe_age` | 0.1549 | 1.18 |

Tras retirar las 7 variables derivadas, el VIF máximo baja a **8.8**.

**[IMPL]** El cálculo usa `LinearRegression` de scikit-learn, no `statsmodels`,
para no añadir dependencia. Es matemáticamente equivalente.

---

## 2.6 Gradient Boosting

### 2.6.1 Formulación general

**[TEORÍA]** (Friedman, 2001) El modelo aditivo por etapas:

$$F_M(\mathbf{x}) = \sum_{m=1}^{M} \nu\, f_m(\mathbf{x}), \qquad f_m \in \mathcal{F}$$

donde $\mathcal{F}$ es el espacio de árboles de regresión y $\nu$ la tasa de
aprendizaje. En cada etapa se ajusta $f_m$ al gradiente negativo de la pérdida.

### 2.6.2 XGBoost: expansión de segundo orden

**[TEORÍA]** (Chen & Guestrin, 2016) El objetivo regularizado en la iteración
$m$:

$$\mathcal{L}^{(m)} = \sum_{k=1}^{n} L\!\left(y_k,\, \hat y_k^{(m-1)} + f_m(\mathbf{x}_k)\right) + \Omega(f_m)$$

$$\Omega(f) = \gamma T + \tfrac{1}{2}\lambda\|\mathbf{w}\|^2$$

con $T$ = número de hojas y $\mathbf{w}$ = pesos de las hojas.

Aproximando con Taylor a segundo orden, con
$g_k = \partial_{\hat y} L$ y $h_k = \partial^2_{\hat y} L$:

$$\mathcal{L}^{(m)} \simeq \sum_k \left[g_k f_m(\mathbf{x}_k) + \tfrac12 h_k f_m^2(\mathbf{x}_k)\right] + \gamma T + \tfrac{\lambda}{2}\sum_{j=1}^T w_j^2$$

Para una estructura de árbol fija, el peso óptimo de la hoja $j$ (con $I_j$ el
conjunto de observaciones que caen en ella) se obtiene derivando e igualando a
cero:

$$\boxed{\ w_j^{\ast} = -\frac{\sum_{k\in I_j} g_k}{\sum_{k\in I_j} h_k + \lambda}\ }$$

y la ganancia de una división en

$$\boxed{\ \text{Gain} = \tfrac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma\ }$$

con $G = \sum g_k$, $H = \sum h_k$ en cada lado.

Para log-loss binaria: $g_k = p_k - y_k$ y $h_k = p_k(1-p_k)$.

**Conexión con `min_child_weight` [IMPL]:** ese hiperparámetro impone
$\sum_{k \in I_j} h_k \geq$ umbral. Como $h_k = p_k(1-p_k) \leq 0.25$, en un
problema desbalanceado donde $p_k \approx 0.07$ se tiene $h_k \approx 0.065$;
`min_child_weight = 20` exige por tanto del orden de **300 observaciones** por
hoja. Es un control de complejidad mucho más fuerte de lo que sugiere el número.

### 2.6.3 LightGBM: crecimiento por hojas

**[TEORÍA]** (Ke et al., 2017) La diferencia estructural:

```
XGBoost — crecimiento por NIVELES     LightGBM — crecimiento por HOJAS
          o                                     o
        /   \                                 /   \
       o     o   ← nivel completo             o     o
      / \   / \                              / \
     o   o o   o ← nivel completo            o   o
                                            / \
   árboles equilibrados                    o   o  ← parte donde más gane
   gobierna max_depth                      gobierna num_leaves
```

LightGBM divide siempre la hoja con mayor reducción de pérdida, esté donde esté.
Alcanza mayor precisión con menos hojas, pero sobreajusta con facilidad si
`num_leaves` es alto.

**Confirmación empírica [MEDIDO]:** ganó la configuración *más simple* probada:

| `num_leaves` | AUC (conjunto de *early stopping*) |
|---|---|
| **15** | **0.86569** |
| 31 | 0.86526 |
| 63 | 0.86294 |
| 127 | 0.86237 |

Además emplea *histogram binning* (agrupa valores continuos en 255 cubos antes
de buscar cortes), lo que explica el factor ~4× de velocidad observado.

### 2.6.4 Manejo nativo de faltantes

**[TEORÍA]** Ambas bibliotecas aprenden, para cada división, una **dirección por
defecto**: prueban enviar los `NaN` a izquierda y a derecha, y conservan la que
maximiza la ganancia.

**Por qué esto importa aquí [MEDIDO]:** el faltante es informativo (5.57% vs
6.96% de tasa de evento). Imputar con la mediana haría a esas 23,675 personas
indistinguibles de quienes efectivamente ganan la mediana, destruyendo señal.

---

## 2.7 Métricas de evaluación

### 2.7.1 Por qué accuracy es inútil

Con tasa de eventos $\pi = 6.68\%$, el clasificador trivial "nadie incumple"
alcanza $1-\pi = 93.32\%$ de accuracy sin detectar un solo evento.

**[MEDIDO]** El modelo con umbral 0.5 alcanza 93.75%: **0.43 puntos** por encima
del trivial, detectando apenas el 20.1% de los eventos.

### 2.7.2 Matriz de confusión

|  | Predice $\hat Y=0$ | Predice $\hat Y=1$ |
|---|---|---|
| **Real $Y=0$** | VN — se presta a un buen cliente | **FP** — se rechaza a un buen cliente |
| **Real $Y=1$** | **FN** — se presta a un moroso | VP — se detecta al moroso |

$$\text{Precision} = \frac{\text{VP}}{\text{VP}+\text{FP}}, \qquad
\text{Recall} = \text{TPR} = \frac{\text{VP}}{\text{VP}+\text{FN}}, \qquad
\text{FPR} = \frac{\text{FP}}{\text{FP}+\text{VN}}$$

### 2.7.3 ROC y AUC

La curva ROC traza $(\text{FPR}(t), \text{TPR}(t))$ recorriendo todos los
umbrales $t$.

**[TEORÍA]** Interpretación probabilística (equivalencia con el estadístico
$U$ de Mann–Whitney):

$$\boxed{\ \text{AUC} = P\!\left(\hat p(\mathbf{X}^+) > \hat p(\mathbf{X}^-)\right) + \tfrac12 P\!\left(\hat p(\mathbf{X}^+) = \hat p(\mathbf{X}^-)\right)\ }$$

donde $\mathbf{X}^+$ y $\mathbf{X}^-$ son observaciones aleatorias de las clases
positiva y negativa.

**Lectura del resultado [MEDIDO]:** con AUC $= 0.86628$, tomando al azar un
moroso y un no-moroso, el modelo asigna mayor riesgo al moroso el **86.6%** de
las veces.

**Propiedad clave:** el AUC es independiente del umbral, lo que lo hace apto
para comparar modelos. **Limitación:** también es independiente de la
calibración — dos modelos con AUC idéntico pueden tener probabilidades muy
distintas.

### 2.7.4 Gini

$$\text{Gini} = 2\,\text{AUC} - 1$$

Convención bancaria. Contiene exactamente la misma información. **[MEDIDO]**
Gini $= 0.7374$ para el modelo desplegado.

### 2.7.5 Estadístico KS

$$\boxed{\ \text{KS} = \max_{t}\,\big|F_0(t) - F_1(t)\big| = \max_t\left[\text{TPR}(t) - \text{FPR}(t)\right]\ }$$

Máxima separación vertical entre las distribuciones acumuladas de buenos y
malos. Es el estadístico de Kolmogorov–Smirnov de dos muestras aplicado a las
puntuaciones.

**[TEORÍA]** Referencia bancaria: $>0.30$ aceptable, $>0.40$ bueno.
**[MEDIDO]** KS $= 0.5865$.

### 2.7.6 Average Precision

$$\text{AP} = \sum_k \left(R_k - R_{k-1}\right) P_k$$

Área bajo la curva precision–recall. A diferencia del AUC, **sí** depende de la
prevalencia: la línea base del azar es $\pi = 0.0668$.

**[MEDIDO]** AP $= 0.4065$, seis veces la línea base.

**[RECOM]** Con clases muy desbalanceadas, la curva precision–recall es más
informativa que la ROC (Saito & Rehmsmeier, 2015), porque la ROC puede parecer
buena aunque la precisión sea baja: el denominador de FPR incluye los VN, que
son abrumadoramente mayoría.

---

## 2.8 Calibración

### 2.8.1 Definición

**[TEORÍA]** Un modelo está perfectamente calibrado si

$$P\!\left(Y = 1 \mid \hat p(\mathbf{X}) = q\right) = q \quad \forall q \in [0,1]$$

Es decir: entre los individuos a quienes asigna 8% de riesgo, exactamente el 8%
incumple.

**Por qué importa aquí:** la regla de decisión de negocio (§2.10) compara la PD
predicha contra un umbral derivado de parámetros económicos. Si las
probabilidades no son PD reales, la comparación carece de sentido. **Un modelo
que ordena bien pero no calibra sirve para hacer ranking, no para decidir cuánto
prestar.**

### 2.8.2 Medición

**[IMPL]** Se parte la validación en deciles de riesgo predicho y se compara la
media predicha contra la frecuencia observada:

$$\text{ECE} = \frac{1}{J}\sum_{j=1}^{J}\left|\bar{p}_j - \bar{y}_j\right| \times 100 \quad [\text{puntos porcentuales}]$$

**[MEDIDO]** Modelo desplegado (XGBoost):

| Decil | PD predicha | PD observada | Error |
|---|---|---|---|
| 0 | 0.57% | 0.47% | +0.10 |
| 5 | 2.69% | 2.17% | +0.52 |
| 8 | 11.37% | 11.80% | −0.43 |
| 9 | **36.51%** | **37.20%** | −0.69 |

Error absoluto medio: **0.34 pp**. El decil 9 —donde se concentra la decisión de
mayor impacto económico— es el más preciso.

**Observación contraintuitiva [MEDIDO]:** XGBoost calibra mejor (0.34 pp) que la
regresión logística (0.53 pp), pese a que ésta optimiza directamente la
verosimilitud. Con $n$ grande y `logloss` como objetivo, el *boosting* alcanza
buena calibración sin necesitar recalibrado posterior (Platt scaling, isotónica).

### 2.8.3 Por qué NO se usa `class_weight='balanced'`

**[MEDIDO]**

| Configuración | AUC | PD media predicha | PD real |
|---|---|---|---|
| Sin ajustar | 0.86854 | **6.68%** | 6.68% |
| `scale_pos_weight = 14` | 0.86793 | **33.48%** | 6.68% |

Reponderar las clases **no mejora el AUC y destruye la calibración**: las
probabilidades dejan de ser PD y se convierten en puntuaciones relativas. El
desbalance se gestiona eligiendo el umbral, no deformando el modelo.

---

## 2.9 SHAP

### 2.9.1 Valores de Shapley

**[TEORÍA]** (Shapley, 1953; Lundberg & Lee, 2017) Para un jugador $i$ en un
juego cooperativo con función de valor $v$:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\left[v(S \cup \{i\}) - v(S)\right]$$

Es la única asignación que satisface simultáneamente **eficiencia**
(las contribuciones suman el total), **simetría**, **jugador nulo** y
**aditividad**.

Aplicado a explicabilidad, $v(S)$ es la predicción usando solo el subconjunto de
características $S$, y se cumple la propiedad de **aditividad local**:

$$\boxed{\ f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{p}\phi_i(\mathbf{x})\ }$$

con $\phi_0 = \mathbb{E}[f(\mathbf{X})]$ el valor base.

**[IMPL]** `TreeSHAP` calcula estos valores en tiempo polinómico
$O(TLD^2)$ para ensambles de árboles, en lugar del $O(2^p)$ del cálculo directo.

### 2.9.2 Verificación empírica de la aditividad [MEDIDO]

```
Cliente #22556 — PD 89.4%
  base (log-odds medio de la población)          −2.6612
  RevolvingUtilization = 1.07                    +1.1184
  peor_atraso = 3                                +1.0523
  total_atrasos = 9                              +0.9926
  ...                                            ───────
  suma = +2.1341   →   σ(2.1341) = 89.4%  ✓
```

La reconciliación es exacta, como garantiza la propiedad de eficiencia.

### 2.9.3 SHAP frente a `feature_importances_`

**[MEDIDO]** Sobre los mismos datos:

| Variable | `feature_importances_` (XGBoost) | SHAP |
|---|---|---|
| `RevolvingUtilization` | **4.7%** | **30.91%** |
| `peor_atraso` | 34.2% | 31.94% |
| `flag_algun_atraso` | 27.8% | 4.48% |

Discrepancia de un factor 6.6× en `RevolvingUtilization`.

**Explicación [TEORÍA]:** `feature_importances_` con `importance_type="gain"`
acumula la reducción de pérdida en las divisiones donde se usó la variable.
Depende del orden en que el algoritmo la eligió y se reparte arbitrariamente
entre variables redundantes. SHAP mide cuánto **mueve la predicción**, que es la
magnitud de interés.

**[RECOM]** La importancia nativa no debe presentarse como "esto es lo que
predice la mora". La comparación XGBoost vs LightGBM en este proyecto lo
demuestra: `RevolvingUtilization` recibe 4.7% en uno y 24.1% en el otro con
datos idénticos.

---

## 2.10 La regla de decisión económica

### 2.10.1 Derivación

Sea $M$ el margen si el cliente paga y $L$ la pérdida si incumple. El valor
esperado de conceder crédito a un solicitante con probabilidad de incumplimiento
$p$:

$$\mathbb{E}[\text{VE}] = (1-p)\,M - p\,L$$

Conceder es óptimo mientras $\mathbb{E}[\text{VE}] > 0$:

$$(1-p)M > pL \;\Longleftrightarrow\; M > p(M+L) \;\Longleftrightarrow\;
\boxed{\ p < p^{\ast} = \frac{M}{M+L}\ }$$

**El umbral no es una elección estadística: lo dicta la economía.**

**[MEDIDO]** Con $M = \$800$ y $L = \$6{,}000$: $p^\ast = 800/6800 = 0.1176$.
El óptimo hallado por barrido empírico cayó en 0.1105 — la fórmula predice el
punto de corte sin necesidad de buscarlo.

### 2.10.2 Validación de que la fórmula generaliza [MEDIDO]

Buscar el umbral en el mismo conjunto donde se reporta el beneficio es
sobreajustar. Partiendo validación en dos mitades:

| Umbral | Origen | Beneficio en la mitad B (no vista) |
|---|---|---|
| 0.1184 | hallado empíricamente en la mitad A | $7,993,200 |
| **0.1176** | **fórmula $M/(M+L)$** | **$8,006,000** |

La fórmula supera al umbral ajustado a los datos. **El umbral se deriva, no se
aprende.**

### 2.10.3 Relación con la teoría de decisión bayesiana

**[TEORÍA]** El resultado es un caso particular de la regla de Bayes con
pérdida asimétrica. Con matriz de costes $c_{01}$ (rechazar a un bueno) y
$c_{10}$ (aceptar a un malo), el umbral óptimo es

$$p^{\ast} = \frac{c_{01}}{c_{01} + c_{10}}$$

Aquí $c_{01} = M$ (margen perdido) y $c_{10} = L$ (capital perdido).

---

## 2.11 Construcción del scorecard

### 2.11.1 Escala

**[TEORÍA]** (Siddiqi, 2006) El score es una transformación afín de los
log-odds, fijada por dos convenciones del banco:

- Un **score base** $S_0$ que corresponde a unas **odds base** $O_0$;
- El **PDO** (*Points to Double the Odds*): cuántos puntos duplican las odds.

$$\boxed{\ \text{Factor} = \frac{\text{PDO}}{\ln 2}, \qquad
\text{Offset} = S_0 - \text{Factor}\cdot\ln O_0\ }$$

$$\text{Score} = \text{Offset} + \text{Factor}\cdot\ln\!\left(\frac{1-p}{p}\right)$$

**[IMPL]** $S_0 = 600$, $O_0 = 50$, $\text{PDO} = 20$
$\Rightarrow$ Factor $= 28.8539$, Offset $= 487.1229$.

Verificación de la propiedad definitoria: sumar 20 puntos multiplica las odds
por $e^{20/28.8539} = e^{\ln 2} = 2$. ✓

### 2.11.2 Puntos por atributo

Repartiendo el intercepto entre las $n$ variables:

$$\boxed{\ \text{Puntos}_{ij} = -\left(\text{WoE}_{ij}\cdot\beta_i + \frac{\beta_0}{n}\right)\cdot\text{Factor} + \frac{\text{Offset}}{n}\ }$$

El signo negativo invierte la convención: WoE alto (seguro) → coeficiente
negativo → **más puntos**.

### 2.11.3 Verificación de reconciliación [MEDIDO]

La prueba que valida todo el scorecard: sumar los puntos de cada característica
debe reproducir exactamente el score derivado de la PD del modelo.

```
Diferencia máxima : 0.0000000000 puntos
Diferencia media  : 0.0000000000 puntos
```

Si no cuadrara, la tabla sería decorativa.

### 2.11.4 Equivalencia score ↔ PD [MEDIDO]

| Score | PD |
|---|---|
| 400 | 95.34% |
| 500 | 39.02% |
| **545** | **11.76%** ← umbral de negocio |
| 600 | 1.96% |
| 650 | 0.35% |

---

## 2.12 Inferencia estadística sobre las comparaciones

### 2.12.1 Por qué un solo split no basta

**[MEDIDO]** La variabilidad del AUC entre particiones:

$$\text{AUC}_{\text{CV}} = 0.85794, \qquad s = 0.00364 \quad (\text{LogReg, 5 particiones})$$

Un intervalo aproximado del 95% abarca $[0.85033,\ 0.86489]$: **1.46 puntos de
amplitud**. Comparar dos modelos con un único número cada uno es, en este
régimen, indistinguible del azar.

### 2.12.2 Contraste emparejado

**[IMPL]** Como los folds son idénticos para los tres modelos, se usa la
diferencia emparejada $d_k = \text{AUC}^A_k - \text{AUC}^B_k$:

$$\text{significativo si} \quad |\bar d| > 2\,\frac{s_d}{\sqrt{K}}$$

**[MEDIDO]**

| Comparación | $\bar d$ | Veredicto |
|---|---|---|
| LogReg vs XGBoost | −0.00685 | **real** |
| LogReg vs LightGBM | −0.00640 | **real** |
| XGBoost vs LightGBM | +0.00045 | **dentro del ruido** |

**[RECOM]** El emparejamiento reduce sustancialmente la varianza frente a un
contraste no emparejado, porque elimina la variabilidad debida a la dificultad
intrínseca de cada fold. Es la elección correcta cuando los folds coinciden.

### 2.12.3 Validación cruzada anidada

**Sospecha:** los hiperparámetros se eligieron con un subconjunto del
entrenamiento que después aparece en los folds de test de la CV. Eso induciría
optimismo.

**[MEDIDO]** Repitiendo toda la selección dentro de cada fold externo:

| | Media | Desv. |
|---|---|---|
| CV reportada | 0.86628 | 0.00306 |
| **Nested CV** | **0.86661** | 0.00337 |

Diferencia: −0.00028. **El sesgo no es detectable.** La rejilla es pequeña (11
configuraciones) y el conjunto grande, lo que limita la capacidad de sobreajuste
por selección.

---

[← Arquitectura](01_arquitectura.md) · [Índice](README.md) · [Siguiente: Pipeline de datos →](03_pipeline_datos.md)

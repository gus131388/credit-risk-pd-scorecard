# 08 — Comparación metodológica

[← Auditoría](07_auditoria.md) · [Índice](README.md) · [Siguiente: Guía del desarrollador →](09_guia_desarrollador.md)

---

Este capítulo contrasta cada decisión metodológica del proyecto con las
alternativas empleadas en academia e industria, y evalúa si la elección
implementada es la más apropiada.

---

## 8.1 Tratamiento de valores atípicos

| Método | Fundamento | Ventajas | Desventajas | Complejidad |
|---|---|---|---|---|
| **Invalidar + indicador (implementado)** | Distinguir dato corrupto de valor extremo | Preserva la señal de los extremos legítimos; no invierte el riesgo | Requiere análisis por variable | $O(n)$ |
| Winsorización a percentil | Reducir influencia de la cola | Automático, sin decisiones | **Aplasta señal real**; asigna riesgo máximo a datos corruptos | $O(n\log n)$ |
| Truncamiento (eliminar filas) | Simplicidad | Elimina el problema | **Inviable en inferencia**; sesga si la corrupción no es aleatoria | $O(n)$ |
| Transformación log/Box-Cox | Comprimir la escala | Reduce asimetría | No resuelve códigos de sistema; irrelevante para árboles | $O(n)$ |
| Detección multivariante (Isolation Forest, Mahalanobis) | Outliers en el espacio conjunto | Detecta combinaciones anómalas | No distingue "raro" de "corrupto"; caja negra | $O(n\log n)$ – $O(n^2)$ |

**Veredicto [RECOM]:** la elección implementada es **la correcta para este
conjunto**, y la evidencia es contundente: winsorizar `RevolvingUtilization` en 1
habría invalidado 2,640 registros con 37.50% de evento. Ningún método automático
habría distinguido "sobregiro real de 1.5" de "código corrupto de 50,708" — solo
el análisis de la tasa de evento por banda lo hace.

**Coste de la elección:** requiere análisis manual por variable, lo que la hace
menos escalable a conjuntos con cientos de predictores.

---

## 8.2 Tratamiento de valores faltantes

| Método | Supuesto sobre el mecanismo | Ventajas | Desventajas |
|---|---|---|---|
| **`NaN` nativo + indicador (implementado, árboles)** | Ninguno | Aprovecha que el faltante es informativo; sin supuestos | Solo funciona con algoritmos que lo soporten |
| **Bin `FALTANTE` en WoE (implementado, lineal)** | Ninguno | Mide el riesgo del grupo faltante; sin imputación | Genera colinealidad con indicadores (ver [H1](07_auditoria.md#h1)) |
| Media/mediana + indicador | MCAR o MAR | Simple, estándar | Distorsiona la distribución; el indicador recupera parte |
| KNN | MAR | Preserva relaciones multivariantes | $O(n^2)$; falla bajo MNAR de nivel registro |
| MICE (imputación múltiple encadenada) | MAR | **Propaga la incertidumbre** de la imputación | Costoso; complica el pipeline de inferencia |
| Máxima verosimilitud (FIML) | MAR | Eficiente, sin descartar información | Requiere modelo paramétrico completo |

**Veredicto [RECOM]:** correcta. La evidencia del proyecto muestra un mecanismo
**MNAR de nivel registro** (100% de solapamiento entre los faltantes de
`MonthlyIncome` y `NumberOfDependents`), donde KNN y MICE —que asumen MAR—
serían inapropiados: imputarían a partir de las mismas variables que están
degradadas en esos registros.

**[RECOM]** MICE sería la alternativa correcta si el mecanismo fuera MAR y se
necesitara cuantificar la incertidumbre de la imputación. Aquí no aplica.

---

## 8.3 Codificación de variables

| Método | Referencia | Ventajas | Desventajas |
|---|---|---|---|
| **WoE (implementado)** | Siddiqi (2006) | Escala log-odds; robusto a outliers; NaN como bin; interpretable | Requiere binning; pierde granularidad intra-bin |
| One-hot | Estándar | Sin supuestos de orden | Alta dimensionalidad; no ordena los niveles |
| Valor crudo | — | Máxima granularidad | Asume linealidad en logit; sensible a outliers |
| Target encoding | Micci-Barreca (2001) | Compacto, captura no-linealidad | **Sobreajusta** sin CV anidada |
| Splines / GAM | Hastie & Tibshirani (1990) | No-linealidad suave sin discretizar | Menos interpretable que una tabla de puntos |
| Binning + one-hot | Alternativa a WoE | Cada bin con coeficiente propio | Más parámetros; no comparable entre variables |

**Veredicto [RECOM]:** para un **scorecard**, WoE es la elección correcta y
estándar de la industria. Su ventaja decisiva es que produce una tabla de puntos
auditable.

**Alternativa a considerar seriamente:** *binning + one-hot* con restricción de
monotonía es equivalente en poder y **evita el problema de colinealidad con los
indicadores** que causó [H1](07_auditoria.md#h1), porque el bin faltante sería
una columna explícita en lugar de un valor dentro de otra columna.

---

## 8.4 Diseño de bins

| Método | Fundamento | Ventajas | Desventajas |
|---|---|---|---|
| **Manual guiado por EDA (implementado)** | Conocimiento del dominio | Cortes con sentido de negocio; incorpora hallazgos como el *mass point* en 0.9–1.0 | **Subjetivo**; no reproducible por otro analista |
| Cuantiles iguales | Simplicidad | Automático, bins equilibrados | **Falla con masas puntuales** (94% de ceros) |
| Árbol de decisión monótono | Maximiza reducción de impureza | Automático; garantiza monotonía | Cortes sin sentido de negocio |
| ChiMerge | Test $\chi^2$ | Fundamento estadístico | Sensible al nivel de significación |
| Binning óptimo (MILP) | Optimización con restricciones | Óptimo garantizado bajo restricciones explícitas | Coste computacional; librería especializada |
| MDLP (Fayyad & Irani, 1993) | Longitud de descripción mínima | Automático, principio de parsimonia | No garantiza monotonía |

**Veredicto [RECOM]:** **la decisión más discutible del proyecto** (ver
[H6](07_auditoria.md#h6)).

A favor de lo implementado: los cuantiles demostradamente fallan aquí, y el
binning manual incorporó hallazgos que ningún algoritmo habría producido —el
corte explícito en 1.0 de `RevolvingUtilization` separa "dentro del límite" de
"sobregiro", una distinción de negocio, no estadística.

En contra: no es reproducible por un tercero, que es un requisito en entorno
regulado.

**Recomendación:** contrastar contra `optbinning` con restricción de monotonía y
documentar la diferencia. Si coinciden, ambos se refuerzan; si difieren, obliga a
justificar explícitamente la elección.

---

## 8.5 Selección de variables

| Método | Ventajas | Desventajas |
|---|---|---|
| **IV + VIF + restricción de signo (implementado)** | Combina poder predictivo, colinealidad e interpretabilidad | Manual; el conjunto de candidatos es finito y elegido a mano |
| Solo IV | Simple, estándar en scorecards | **Ignora la correlación** — premia variables redundantes |
| Stepwise (forward/backward) | Automático, clásico | Inflación de error tipo I; inestable |
| LASSO ($L_1$) | Selección y regularización simultáneas | Selección inestable con predictores correlacionados |
| Elastic Net | Estable con grupos correlacionados | Dos hiperparámetros que ajustar |
| Boruta / importancia por permutación | Basado en el modelo, capta interacciones | Costoso; no garantiza signos coherentes |

**Veredicto [RECOM]:** el criterio **"signos coherentes primero, AUC después"**
es correcto para un scorecard y no es habitual verlo implementado
explícitamente — la mayoría de los proyectos maximizan AUC y descubren los signos
invertidos al presentar el modelo.

**Limitación reconocida:** solo se evaluaron 6 subconjuntos definidos a mano.
Elastic Net con restricción de signo (disponible en `scikit-learn` vía
`positive=True` sobre variables transformadas) sería más sistemático.

---

## 8.6 Gestión del desbalance de clases

| Método | Efecto en AUC | Efecto en calibración | Veredicto |
|---|---|---|---|
| **Nada + umbral económico (implementado)** | Referencia | **Preservada** (PD media 6.68% = real) | ✅ |
| `class_weight='balanced'` | −0.0008 **[MEDIDO]** | **Destruida** (PD media 33.48%) | ❌ |
| SMOTE | Variable | Distorsionada | ❌ para PD |
| Submuestreo de la mayoría | Similar, más varianza | Distorsionada (recalibrable) | 🟡 |
| Aprendizaje sensible al coste | Similar | Depende de la implementación | 🟡 |

**Veredicto [RECOM]:** correcta y **medida, no asumida**. La demostración
empírica de que reponderar no mejora el AUC y destruye la calibración es un
resultado que muchos proyectos omiten.

**[RECOM]** SMOTE es especialmente inapropiado aquí: genera observaciones
sintéticas interpolando entre vecinos, lo que carece de sentido con variables
como "número de episodios de mora" (¿qué significa 1.7 episodios?) y con `NaN`
informativos.

---

## 8.7 Elección de la familia de modelos

| Familia | AUC esperado en credit scoring | Interpretabilidad | Adopción regulatoria |
|---|---|---|---|
| **Regresión logística + WoE** | Referencia | **Total** | Universal |
| **Gradient boosting** | +0.005 a +0.02 | Baja (requiere SHAP) | Creciente, con reservas |
| Random Forest | Similar a boosting | Baja | Poco frecuente |
| Redes neuronales | Similar o inferior con datos tabulares | Muy baja | Rara vez aprobada |
| SVM | Inferior con $n$ grande | Baja | No usada |

**[TEORÍA]** La literatura de benchmarking (Lessmann et al., 2015, *European
Journal of Operational Research*) sobre 41 conjuntos de credit scoring encuentra
que los métodos de ensamble superan consistentemente a la regresión logística,
pero **con márgenes modestos**. Este proyecto reproduce ese resultado: +0.0069 de
AUC.

**Veredicto [RECOM]:** entrenar las tres familias y compararlas con validación
cruzada es la práctica correcta. La conclusión —el *booster* gana de forma
estadísticamente real pero por poco— es coherente con la literatura.

**El intercambio de fondo:** 0.0069 de AUC (~$5 por solicitud) contra la
capacidad de explicar cada rechazo con una tabla de puntos. En banca esa decisión
frecuentemente se resuelve a favor de la interpretabilidad.

---

## 8.8 Explicabilidad

| Método | Garantías teóricas | Coste | Fidelidad |
|---|---|---|---|
| **Scorecard (implementado, lineal)** | Exacto por construcción | Nulo | **Exacta** |
| **SHAP / TreeSHAP (implementado, árboles)** | Axiomas de Shapley; aditividad | $O(TLD^2)$ | **Exacta** para árboles |
| `feature_importances_` | Ninguna | Nulo | **No fiable** (ver [E9](05_experimentos.md#e9--importancia-de-variables-inestabilidad-entre-algoritmos)) |
| LIME | Fidelidad local aproximada | Alto (remuestreo) | Aproximada, inestable |
| Partial Dependence Plots | Efecto marginal medio | Medio | Sesgado con predictores correlacionados |
| Counterfactual explanations | — | Alto | Útil para explicar al cliente |

**Veredicto [RECOM]:** correcta. Usar SHAP para el modelo desplegado y el
scorecard para el lineal cubre ambos casos con métodos de fidelidad exacta.

**[RECOM]** Las *explicaciones contrafactuales* ("con un score 30 puntos mayor
habría sido aprobado") serían un complemento valioso para la comunicación al
cliente, y el scorecard ya proporciona la base para generarlas.

---

## 8.9 Estrategia de validación

| Método | Sesgo | Varianza | Coste |
|---|---|---|---|
| Split único 80/20 | Bajo | **Alta** (±0.003 medido) | Mínimo |
| **CV $k$-fold (implementado, $k=5$)** | Bajo | Media | $k\times$ |
| CV anidada (aplicada como verificación) | **Mínimo** | Media | $k \times m \times$ |
| Bootstrap .632+ | Bajo | Baja | Alto |
| **Out-of-time** | Mide lo correcto | Depende | Requiere datos temporales |

**Veredicto [RECOM]:** la validación cruzada con contraste emparejado es la
elección correcta, y la verificación con CV anidada es una práctica de rigor poco
habitual en proyectos de este tipo.

**Deficiencia:** ausencia de validación temporal
([H2](07_auditoria.md#h2)) — la única que mide lo que realmente importa para un
modelo de crédito.

---

## 8.10 Determinación del umbral

| Método | Fundamento | Ventaja | Desventaja |
|---|---|---|---|
| **$M/(M+L)$ (implementado)** | Teoría de decisión bayesiana | Óptimo demostrable; **no requiere datos** | Depende de estimar $M$ y $L$ |
| Maximizar F1 | Equilibrio precision/recall | Automático | Ignora la asimetría económica |
| Índice de Youden ($\max$ TPR−FPR) | Punto de máximo KS | Simple | Supone costes simétricos |
| Barrido empírico sobre validación | Directo | Aparentemente óptimo | **Sobreajusta** — demostrado en [E12](05_experimentos.md#e12--debe-buscarse-el-umbral-en-los-datos) |
| Fijar tasa de aprobación objetivo | Restricción operativa | Alineado con el plan comercial | No optimiza beneficio |

**Veredicto [RECOM]:** correcta, y **demostrada empíricamente superior** al
barrido: el umbral teórico (0.1176) rinde mejor en datos no vistos que el hallado
empíricamente (0.1184). Es una verificación que rara vez se realiza.

---

## 8.11 Síntesis

| Decisión | Veredicto | Comentario |
|---|---|---|
| Invalidar + indicador para atípicos | ✅ Óptima | Demostrada con evidencia contundente |
| `NaN` nativo / bin FALTANTE | ✅ Óptima | El mecanismo MNAR descarta KNN/MICE |
| WoE para el modelo lineal | ✅ Estándar | Considerar binning + one-hot para evitar H1 |
| **Binning manual** | 🟡 **Discutible** | Contrastar con binning automático (H6) |
| Signos primero, AUC después | ✅ Superior a la práctica común | Poco frecuente y correcto |
| Sin reponderación de clases | ✅ Óptima | Medida, no asumida |
| Tres familias comparadas con CV | ✅ Correcta | Coherente con Lessmann et al. (2015) |
| SHAP + scorecard | ✅ Correcta | Añadir contrafactuales |
| CV emparejada + anidada | ✅ Rigurosa | Falta out-of-time (H2) |
| Umbral $M/(M+L)$ | ✅ Óptima | Verificada empíricamente |

**Conclusión general [RECOM]:** las decisiones metodológicas son, en su mayoría,
las apropiadas y en varios casos superiores a la práctica habitual. La excepción
es el binning manual, que es defendible por su calidad pero deficiente en
reproducibilidad.

---

[← Auditoría](07_auditoria.md) · [Índice](README.md) · [Siguiente: Guía del desarrollador →](09_guia_desarrollador.md)

# 10 — Referencias y niveles de evidencia

[← Guía del desarrollador](09_guia_desarrollador.md) · [Índice](README.md)

---

## 10.1 Aviso sobre la verificación de fuentes

**Declaración de transparencia.** Las referencias de esta sección se citan de
memoria del autor de la documentación y **no han sido verificadas contra las
publicaciones originales durante la redacción de este manual**. Los datos
bibliográficos (año, volumen, páginas) deben confirmarse antes de cualquier uso
formal —una tesis, un informe de validación regulatoria o una publicación—.

Las afirmaciones etiquetadas **[MEDIDO]** a lo largo del manual **sí** están
verificadas: proceden de ejecuciones del código de este repositorio y son
reproducibles con los scripts incluidos.

Esta distinción es deliberada. Confundir "lo he leído" con "lo he comprobado" es
una fuente habitual de error en informes técnicos.

---

## 10.2 Algoritmos y métodos

### Gradient boosting

- **Friedman, J. H.** (2001). *Greedy Function Approximation: A Gradient Boosting
  Machine*. Annals of Statistics, 29(5), 1189–1232.
  → Formulación original del boosting como descenso de gradiente en espacio de
  funciones. Base de [§2.6.1](02_fundamentos_matematicos.md#261-formulación-general).

- **Chen, T. & Guestrin, C.** (2016). *XGBoost: A Scalable Tree Boosting System*.
  Proceedings of KDD '16.
  → Expansión de Taylor de segundo orden, término de regularización, algoritmo de
  cortes aproximados y manejo de dispersión (*sparsity-aware split finding*),
  que es el mecanismo de tratamiento nativo de `NaN`.
  Base de [§2.6.2](02_fundamentos_matematicos.md#262-xgboost-expansión-de-segundo-orden).

- **Ke, G. et al.** (2017). *LightGBM: A Highly Efficient Gradient Boosting
  Decision Tree*. Proceedings of NeurIPS 2017.
  → Crecimiento por hojas, GOSS (*Gradient-based One-Side Sampling*) y EFB
  (*Exclusive Feature Bundling*).
  Base de [§2.6.3](02_fundamentos_matematicos.md#263-lightgbm-crecimiento-por-hojas).

### Explicabilidad

- **Shapley, L. S.** (1953). *A Value for n-Person Games*. En Kuhn & Tucker
  (eds.), Contributions to the Theory of Games II. Princeton University Press.
  → Axiomas de eficiencia, simetría, jugador nulo y aditividad.

- **Lundberg, S. M. & Lee, S.-I.** (2017). *A Unified Approach to Interpreting
  Model Predictions*. Proceedings of NeurIPS 2017.
  → Aplicación de los valores de Shapley a explicabilidad de modelos.

- **Lundberg, S. M. et al.** (2020). *From Local Explanations to Global
  Understanding with Explainable AI for Trees*. Nature Machine Intelligence, 2(1).
  → TreeSHAP: cálculo en tiempo polinómico para ensambles de árboles.

### Codificación y binning

- **Micci-Barreca, D.** (2001). *A Preprocessing Scheme for High-Cardinality
  Categorical Attributes in Classification and Prediction Problems*. ACM SIGKDD
  Explorations, 3(1).
  → Target encoding y su riesgo de sobreajuste.

- **Fayyad, U. & Irani, K.** (1993). *Multi-Interval Discretization of
  Continuous-Valued Attributes for Classification Learning*. IJCAI-93.
  → Discretización por MDLP.

- **Navas-Palencia, G.** (2020). *Optimal Binning: Mathematical Programming
  Formulation*. arXiv:2001.08025.
  → Fundamento de la librería `optbinning`, alternativa recomendada al binning
  manual en [H6](07_auditoria.md#h6).

### Métricas

- **Hanley, J. A. & McNeil, B. J.** (1982). *The Meaning and Use of the Area
  under a Receiver Operating Characteristic (ROC) Curve*. Radiology, 143(1).
  → Equivalencia entre AUC y el estadístico U de Mann–Whitney.

- **Saito, T. & Rehmsmeier, M.** (2015). *The Precision-Recall Plot Is More
  Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced
  Datasets*. PLoS ONE, 10(3).
  → Base de la recomendación de [§2.7.6](02_fundamentos_matematicos.md#276-average-precision).

- **Niculescu-Mizil, A. & Caruana, R.** (2005). *Predicting Good Probabilities
  with Supervised Learning*. Proceedings of ICML '05.
  → Calibración de modelos; Platt scaling e isotónica.

---

## 10.3 Credit scoring y riesgo de crédito

- **Siddiqi, N.** (2006). *Credit Risk Scorecards: Developing and Implementing
  Intelligent Credit Scoring*. Wiley.
  → Referencia canónica de la industria. Fuente de: convención de signo del WoE,
  escala del IV, y construcción del scorecard (Factor/Offset/PDO).
  Base de [§2.2](02_fundamentos_matematicos.md#22-weight-of-evidence),
  [§2.3](02_fundamentos_matematicos.md#23-information-value) y
  [§2.11](02_fundamentos_matematicos.md#211-construcción-del-scorecard).

- **Thomas, L. C., Edelman, D. B. & Crook, J. N.** (2002). *Credit Scoring and
  Its Applications*. SIAM.
  → Tratamiento formal, incluida la teoría de reject inference.

- **Lessmann, S., Baesens, B., Seow, H.-V. & Thomas, L. C.** (2015).
  *Benchmarking State-of-the-Art Classification Algorithms for Credit Scoring: An
  Update of Research*. European Journal of Operational Research, 247(1).
  → Comparación sobre 41 conjuntos: los ensambles superan a la regresión
  logística con márgenes modestos. Contexto de
  [§8.7](08_comparacion_metodologica.md#87-elección-de-la-familia-de-modelos).

- **Baesens, B. et al.** (2003). *Benchmarking State-of-the-Art Classification
  Algorithms for Credit Scoring*. Journal of the Operational Research Society,
  54(6).

---

## 10.4 Regulación

- **BCBS** (2006). *International Convergence of Capital Measurement and Capital
  Standards: A Revised Framework — Comprehensive Version* (Basilea II).
  Bank for International Settlements.
  → §452: definición de default. §272 y siguientes: fórmula IRB de capital y
  correlación de activos. Base de
  [§6.6](06_negocio_y_regulacion.md#66-basilea-iiiii-capital-regulatorio).

- **BCBS** (2005). *Studies on the Validation of Internal Rating Systems*.
  Working Paper No. 14.
  → Requisitos de validación: discriminación, calibración, estabilidad.
  Base de [§6.6.3](06_negocio_y_regulacion.md#663-requisitos-de-validación-bajo-basilea)
  y de [H2](07_auditoria.md#h2).

- **BCBS** (2017). *Basel III: Finalising Post-Crisis Reforms*.
  → Suelos de parámetros (*input floors*) y restricciones al uso de modelos
  internos.

- **IASB** (2014). *IFRS 9 Financial Instruments*.
  → Modelo de pérdida crediticia esperada; los tres estadios; criterio de
  deterioro significativo (SICR).
  Base de [§6.5](06_negocio_y_regulacion.md#65-expected-credit-loss-ecl).

- **Federal Reserve / OCC** (2011). *Supervisory Guidance on Model Risk
  Management* (SR 11-7 / OCC 2011-12).
  → Gobierno del riesgo de modelo: desarrollo, validación independiente y uso.
  Base de la observación de proceso en [§7.5](07_auditoria.md#75-opinión-de-validación).

- **CFPB / Regulation B** — Equal Credit Opportunity Act, 12 CFR Part 1002.
  → Uso de la edad en decisiones de crédito; requisito de sistema
  *empirically derived, demonstrably and statistically sound*.
  Base de [§6.7.1](06_negocio_y_regulacion.md#671-variables-potencialmente-sensibles).

---

## 10.5 Estadística y aprendizaje automático

- **Hastie, T., Tibshirani, R. & Friedman, J.** (2009). *The Elements of
  Statistical Learning* (2.ª ed.). Springer.
  → Regresión logística, regularización, validación cruzada, boosting.

- **Hosmer, D. W., Lemeshow, S. & Sturdivant, R. X.** (2013). *Applied Logistic
  Regression* (3.ª ed.). Wiley.
  → Estimación por máxima verosimilitud, diagnóstico, multicolinealidad.

- **Little, R. J. A. & Rubin, D. B.** (2019). *Statistical Analysis with Missing
  Data* (3.ª ed.). Wiley.
  → Taxonomía MCAR/MAR/MNAR. Base del argumento de
  [§8.2](08_comparacion_metodologica.md#82-tratamiento-de-valores-faltantes).

- **Varma, S. & Simon, R.** (2006). *Bias in Error Estimation when Using
  Cross-Validation for Model Selection*. BMC Bioinformatics, 7(91).
  → Justificación de la validación cruzada anidada
  ([E11](05_experimentos.md#e11--validación-cruzada-anidada)).

- **Cawley, G. C. & Talbot, N. L. C.** (2010). *On Over-fitting in Model
  Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR, 11.
  → Sesgo por selección; fundamento de apartar un conjunto de selección
  independiente de validación.

---

## 10.6 Documentación oficial de las herramientas

| Herramienta | Documentación |
|---|---|
| scikit-learn | https://scikit-learn.org/stable/ |
| XGBoost | https://xgboost.readthedocs.io/ |
| LightGBM | https://lightgbm.readthedocs.io/ |
| SHAP | https://shap.readthedocs.io/ |
| pandas | https://pandas.pydata.org/docs/ |

---

## 10.7 Origen de los datos

**Give Me Some Credit** — competencia de Kaggle, 2011.
https://www.kaggle.com/c/GiveMeSomeCredit

**[MEDIDO]** Lo verificado sobre el origen:

- Datos históricos **anonimizados de solicitantes reales**, no sintéticos. Lo
  respalda la naturaleza del ruido observado: códigos centinela, faltantes
  estructurados y outliers de sistema son característicos de datos operacionales
  bancarios.
- 150,000 registros etiquetados y 101,503 sin etiqueta, con proporciones de
  anomalías prácticamente idénticas entre ambos (19.82% vs 19.81% de
  `MonthlyIncome` faltante; 23.42% vs 23.23% de `DebtRatio > 1`), lo que confirma
  que proceden de la misma población.

**No verificado:** la institución de origen (anónima por diseño) y la metodología
exacta de partición train/test. Lo más probable es un muestreo aleatorio simple
—práctica estándar en competencias de la época— pero **no se encontró
confirmación documental**.

---

## 10.8 Resumen de niveles de evidencia usados en el manual

| Etiqueta | Significado | Verificable cómo |
|---|---|---|
| **[TEORÍA]** | Resultado establecido en la literatura o normativa | Consultando la fuente citada |
| **[IMPL]** | Cómo está implementado en este repositorio | Leyendo el código |
| **[MEDIDO]** | Número obtenido ejecutando el código | Re-ejecutando los scripts |
| **[RECOM]** | Recomendación u opinión del autor de la documentación | No verificable — es juicio profesional |

**Advertencia final.** Las afirmaciones **[RECOM]** son las que más cuidado
requieren: representan criterio profesional, no hechos establecidos. En un
informe de validación formal deberían someterse a contraste independiente.

---

[← Guía del desarrollador](09_guia_desarrollador.md) · [Índice](README.md)

# Give Me Some Credit — Modelo de Probabilidad de Default (PD)

Modelo de credit scoring que predice la probabilidad de que un solicitante
entre en mora grave (90+ días de atraso) durante los **dos años siguientes**.

Dataset: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
(Kaggle, 2011) — 150,000 solicitantes con etiqueta, 101,503 sin etiqueta.

> **Estado:** completo — tres modelos entrenados, comparados y submission generada.

## Documentación

El manual técnico completo está en [`docs/`](docs/README.md): arquitectura,
fundamentos matemáticos, pipeline, modelado, experimentos, impacto de negocio y
regulación, auditoría independiente, comparación metodológica y guía del
desarrollador.

> **Nota de la auditoría.** La revisión independiente documentada en
> [`docs/07_auditoria.md`](docs/07_auditoria.md) concluye que el sistema **no es
> desplegable en producción bancaria** en su estado actual.
>
> - **H1** — atribución incorrecta en el scorecard: ✅ **corregido**. El
>   scorecard usa ahora solo las 10 variables originales; 0 coeficientes
>   incoherentes, VIF máximo 1.3.
> - **H2** — sin validación *out-of-time*: ⛔ **abierto**. No subsanable: el
>   conjunto de datos carece de marca temporal.
> - **H3** — sin marco de monitoreo: ⛔ **abierto**. Requiere datos temporales.
>
> Como investigación aplicada sí es completo y reproducible.

## Resultados

| Modelo | ROC-AUC (CV, 5 folds) | KS | Error de calibración | Valor de negocio |
|---|---|---|---|---|
| Logistic Regression (WoE, 10 vars) | 0.85794 ± 0.00359 | 0.5720 | 0.72 pp | $172.59 / solicitud |
| **XGBoost** ← elegido | **0.86628 ± 0.00306** | **0.5832** | **0.35 pp** | $180.37 / solicitud |
| LightGBM | 0.86619 ± 0.00343 | 0.5885 | 0.36 pp | $182.21 / solicitud |

Comparación emparejada fold a fold:

```
LogReg  vs XGBoost    −0.00835   diferencia REAL
LogReg  vs LightGBM   −0.00826   diferencia REAL
XGBoost vs LightGBM   +0.00009   dentro del ruido
```

Los dos boosters superan a la regresión logística de forma consistente; **entre
ellos son estadísticamente indistinguibles**. El desempate se hizo por
calibración, porque de ella depende que la regla de negocio funcione.

Frente a no usar modelo, el elegido aporta **$5.4M sobre 30,000 solicitudes**
(ver [evaluación económica](#el-modelo-en-dólares)).

---

## El problema, en una frase

Las variables describen el **pasado** del solicitante (ventana de observación);
el target describe su **futuro** (ventana de desempeño):

```
   VENTANA DE OBSERVACIÓN          │      VENTANA DE DESEMPEÑO
   (las variables)                 │      (el target)
 ──────────────────────────────────┼────────────────────────────────►
   NumberOfTimes90DaysLate = 1     │   ¿Caerá 90+ días en mora
   RevolvingUtilization = 0.85     │    durante los próximos 2 años?
   age = 45                        │
                              punto de decisión
```

Esta distinción es la clave del proyecto y descarta la sospecha de fuga de datos:
**el 65.4% de quienes cayeron en mora tenían historial de 90+ días en cero**, y
el 58.8% de quienes sí tenían historial no reincidieron. Ahí está el verdadero
reto — detectar riesgo en personas con historial aparentemente limpio.

---

## Estructura

```
Probability_of_Default/
├── 01_eda.py                  Análisis exploratorio (datos crudos, sin recortar)
├── 02_preprocessing.py        Orquestador: split único → dos ramas → artefactos
├── 03_train_logistic.py       Regresión logística sobre WoE (baseline)
├── 04_train_xgboost.py        XGBoost sobre valores crudos
├── 05_train_lightgbm.py       LightGBM sobre valores crudos
├── 06_evaluation.py           Comparación final, elección y submission
├── 07_business_evaluation.py  Traducción a dólares y sensibilidad
├── 08_scorecard.py            Tabla de puntos (el entregable bancario)
├── predict.py                 Puntuar solicitantes nuevos (entrada de producción)
│
├── src/
│   ├── preprocessing.py       Limpieza por reglas + feature engineering
│   ├── woe.py                 Weight of Evidence e Information Value
│   ├── evaluation.py          Métricas, calibración y regla de negocio
│   └── artifacts.py           Persistencia del transformador entrenado
│
├── datasets/processed/
│   ├── trees/                 train / val / test  (XGBoost, LightGBM)
│   └── woe/                   train / val / test  (Logistic Regression)
│
├── artifacts/
│   ├── preprocessors/         El transformador congelado (.joblib)
│   ├── models/                Los tres modelos entrenados
│   ├── metrics/               Resultados publicados (versionados en git)
│   ├── plots/                 Gráficas de evaluación
│   └── submissions/           Predicciones sobre cs-test.csv
├── outputs/                   Gráficas del EDA
├── docs/                      Manual técnico completo (11 capítulos)
├── tests/                     91 pruebas
│   ├── test_preprocessing.py    invariantes del pipeline (21)
│   ├── test_woe.py              WoE, IV y auditor de binning (35)
│   └── test_evaluation.py       métricas y regla de negocio (35)
└── .gitignore                 datos y modelos son derivados: no van a git
```

### Cómo ejecutarlo

```bash
pip install -r requirements.txt

python 01_eda.py                  # gráficas exploratorias → outputs/
python 02_preprocessing.py        # datasets + transformador persistido
python 03_train_logistic.py       # baseline interpretable
python 04_train_xgboost.py        # booster 1
python 05_train_lightgbm.py       # booster 2 + comparación CV
python 06_evaluation.py           # elección final + SHAP + submission
python 07_business_evaluation.py  # traducción a dólares
python 08_scorecard.py            # tabla de puntos

python predict.py --demo          # puntuar solicitantes nuevos

pytest tests/ -v                  # 91 pruebas (independientes del resto)
```

Cambiar algo obliga a re-ejecutar todo lo que dependa de ello: los umbrales de
`src/` afectan a toda la cadena; los supuestos económicos solo al `07`.

---

## Decisiones de diseño

### Un solo split, dos ramas de transformación

```
raw ──► split único (80/20 estratificado) ──┬── rama árboles ──► trees/
                                            └── rama WoE ─────► woe/
```

Ambas ramas contienen a **las mismas personas** en cada partición, para que la
comparación entre modelos sea justa. Cada familia recibe el tratamiento que le
corresponde:

| | Árboles (XGBoost, LightGBM) | Logistic Regression |
|---|---|---|
| Outliers | sin capear | binning |
| Faltantes | `NaN` nativo | bin propio |
| Escalado | no | no (WoE ya normaliza) |
| Codificación | valores crudos | Weight of Evidence |

**Por qué no capear para árboles:** un árbol solo mira el *orden*. Un corte en
`RevUtil > 1.5` separa las mismas filas tanto si el máximo es 2 como si es 50,708.
Capear no aporta nada y elimina cortes posibles.

**Por qué no imputar para árboles:** XGBoost y LightGBM aprenden en cada división
hacia qué lado enviar los `NaN`. Y aquí el faltante *es* predictivo (5.57% de
default frente a 6.96%), así que imputar destruiría señal.

### Códigos de sistema disfrazados de números

El hallazgo transversal del EDA. Tres variables tenían valores que **parecían**
datos extremos pero eran códigos de "no disponible" — y en los tres casos el
riesgo real era el **opuesto** al que sugería el número:

| Variable | Valor engañoso | Riesgo real | Riesgo que el modelo habría asumido |
|---|---|---|---|
| Morosidad | `96` / `98` | 54.67% | "98 atrasos" → extremo |
| `RevolvingUtilization` | `> 100` | ~6.4% | "utilización altísima" → extremo |
| `MonthlyIncome` | `0` o `1` | ~3.7% | "ingreso nulo" → extremo |

En los tres: convertir a `NaN` + crear un flag. Nunca capear ni dejar el número.

**El error que casi cometemos:** capear `RevolvingUtilization` en 1 habría
invalidado 2,640 filas con **37.50% de default** — el grupo más predictivo del
dataset. Los valores entre 1 y 100 son sobregiros legítimos, no errores.

### Prevención de fuga de datos

Toda estadística se aprende **solo** con el conjunto de entrenamiento:

```python
tablas = aprender_woe(X_train, y_train)   # mira el target: SOLO train
X_val  = aplicar_woe(X_val,  tablas)      # no mira el target
X_test = aplicar_woe(X_test, tablas)      # usa lo aprendido en train
```

El script verifica 14 invariantes al terminar, incluida una prueba de ida y
vuelta: recargar el artefacto desde disco y aplicarlo a datos crudos debe
reproducir exactamente el mismo resultado.

### Por qué se persiste el transformador

Guardar solo los `.parquet` no basta: son el *resultado* de la transformación,
no la transformación. Para puntuar a un cliente nuevo hay que aplicarle los
mismos cortes y los mismos valores WoE del día del entrenamiento. El artefacto
guarda tablas WoE, cortes, umbrales de limpieza, orden de columnas y metadatos
de trazabilidad.

---

## Hallazgos del EDA

| Hallazgo | Dato |
|---|---|
| Desbalance del target | 6.68% de defaults (1 por cada 14) |
| Variable más predictiva | `RevolvingUtilization`: 1.84% → 39.76% de default |
| Grupo de mayor riesgo | filas con centinela 96/98: **54.67%** |
| Faltantes informativos | sin ingreso → **menos** riesgo (5.57% vs 6.96%) |
| Correlación 0.99 espuria | 214 filas (0.18%) inflaban la correlación real de 0.22 a 0.98 |
| Pearson engañosa | `RevolvingUtilization` daba −0.002; por deciles, 1.4% → 23.2% |

**Sobre el EDA:** las gráficas usan datos **crudos y sin recortar**. Un histograma
recortado al percentil 99 se ve mejor pero esconde justo lo que el EDA debe
encontrar. Si la gráfica se ve aplastada por un outlier, esa fealdad *es* el
hallazgo. Las bimodalidades de `RevolvingUtilization` y `DebtRatio` —dos
poblaciones separadas por un valle vacío— solo son visibles así.

---

## El modelo en dólares

Un ROC-AUC no le dice nada a un comité de crédito. La pregunta real es dónde
poner el punto de corte, y eso lo dicta la economía, no la estadística:

```
Valor esperado de prestar a alguien con probabilidad de mora p:
        (1 − p)·M − p·L  >  0        →        p  <  M / (M + L)
```

Con un margen de $800 y una pérdida de $6,000 (LGD 60% sobre $10,000), el
umbral óptimo es **11.76%** — y el óptimo empírico sobre validación cayó en
11.05%, confirmando la fórmula.

| Política | Beneficio (30,000 solicitudes) |
|---|---|
| Aprobar a todos (sin modelo) | $10,366,000 |
| Con modelo, umbral 11.76% | **$15,777,600** |

**Pero el modelo no siempre compensa.** Si las pérdidas son baratas respecto al
margen (ratio < 1), lo óptimo es prestarle a todo el mundo y el modelo solo
resta valor rechazando gente que sí habría pagado:

| Ratio pérdida/ganancia | Sin modelo | Con modelo | Ganancia |
|---|---|---|---|
| 0.5× | $21,594,000 | $21,580,800 | **−$13,200** |
| 7.5× | $10,366,000 | $15,777,600 | +$5,411,600 |
| 15× | **−$1,664,000** | $12,489,600 | +$14,153,600 |

Un modelo con AUC 0.87 puede ser económicamente inútil. A partir de ~15× es la
diferencia entre un negocio viable y uno que pierde dinero.

## El scorecard

El entregable característico del credit scoring: cada rango de cada variable
vale una cantidad fija de puntos, y el score del cliente es la suma.

```
Escala:  600 puntos ↔ odds 50:1     PDO = 20 (cada 20 puntos duplican las odds)

RevolvingUtilization   0.0 – 0.1   →  +79.9 pts   (referencia)
                       0.5 – 0.7   →  +49.2 pts   (−30.7 vs normal)
                       0.9 – 1.0   →  +34.9 pts   (−45.0 vs normal)
                       > 1.0       →  +17.1 pts   (−62.8 vs normal)
```

El scorecard usa **solo las 10 variables originales** tras la corrección del
hallazgo H1: los indicadores de calidad de dato resultaron perfectamente
colineales con los bins `FALTANTE` del WoE.

**Verificación exacta:** sumar los puntos de un cliente reproduce el score
derivado de la PD del modelo con una diferencia máxima de `0.0000000000`. Si no
cuadrara, la tabla sería decorativa.

| Banda de score | n | % población | default |
|---|---|---|---|
| < 500 | 944 | 3.15% | **54.98%** |
| 500–550 | 3,331 | 11.10% | 21.98% |
| 550–600 | 14,785 | 49.28% | 4.53% |
| 600–650 | 10,940 | 36.47% | **0.77%** |

El umbral de negocio (PD < 11.76%) equivale a **545 puntos**: se aprobaría al
88.2% de los solicitantes.

Y así se explica un rechazo concreto:

```
Cliente #11724 — score 402 — PD 95.1%
  RevolvingUtilization = 1.07   −62.8 pts
  NumberOfTimes90DaysLate = 3   −56.3 pts
  NumberOfTime60-89Days = 2     −32.3 pts
  NumberOfTime30-59Days = 5     −32.2 pts
```

## Explicabilidad del modelo elegido (SHAP)

El scorecard solo existe para la regresión logística. Como el modelo elegido fue
XGBoost, hacía falta cerrar esa brecha: SHAP reparte cada predicción individual
entre sus variables, y las contribuciones **suman exactamente** el resultado del
modelo.

```
Cliente #22556 — PD 89.4%
  base (log-odds medio de la población)          −2.6612
  RevolvingUtilization = 1.07                    +1.1184
  peor_atraso = 3                                +1.0523
  total_atrasos = 9                              +0.9926
  ...                                            ───────
  suma = +2.1341   →   PD = 89.4%  ✓
```

Además, SHAP **contradice a `feature_importances_`**: da a
`RevolvingUtilization` un 30.91% de impacto, mientras la importancia nativa de
XGBoost le daba 4.7%. Otra razón para no leer esa tabla sin más.

## Tres lecciones del proyecto

**1. Los códigos de sistema invierten la señal.** Tres variables tenían valores
que *parecían* extremos y en realidad eran códigos de "no disponible" — y en los
tres casos el riesgo real era el opuesto al que sugería el número. Capear
`RevolvingUtilization` en 1 habría invalidado 2,640 filas con 37.50% de default,
el grupo más predictivo del dataset.

**2. Añadir variables puede restar — dos veces.** Las 7 variables derivadas
causaban multicolinealidad y coeficientes con el signo invertido en la regresión
logística. Y los 7 indicadores de calidad de dato resultaron **perfectamente
colineales con los bins `FALTANTE`** del WoE, dando puntos a favor al grupo con
54.67% de morosidad. El scorecard final usa **solo las 10 variables originales**:
cuesta 0.0019 de AUC y a cambio el VIF baja de 8.8 a 1.3.

**3. La importancia de variables es del modelo, no de los datos.**
`RevolvingUtilization` recibe 4.7% de importancia en XGBoost y **24.1%** en
LightGBM — mismos datos, misma información. Con variables redundantes cada
algoritmo reparte el crédito a su manera. Presentar esa tabla como "esto es lo
que predice la mora" sería presentar un artefacto.

## Notas metodológicas

- **Un solo split no basta.** La variabilidad entre particiones es ±0.003 de
  AUC. Las comparaciones entre modelos se hacen con validación cruzada de 5
  particiones idénticas y prueba emparejada fold a fold.
- **Validación nunca participa en decisiones.** Cada script aparta un
  sub-conjunto del *train* para seleccionar variables o hiperparámetros. Si se
  eligiera mirando validación, el AUC reportado estaría inflado.
- **Sin `class_weight='balanced'` ni `scale_pos_weight`.** No mejoran el AUC y
  destruyen la calibración (la PD media pasa de 6.7% a 33%). El desbalance se
  maneja en el umbral, no deformando el modelo.
- **Reentrenamiento final con las 150,000 filas.** Una vez tomadas todas las
  decisiones, esconder la validación ya no aporta nada.
- **Una comprobación con exenciones puede ocultar defectos.** Eximir a los
  indicadores del control de signos permitió que tres llegaran al modelo con la
  dirección invertida. La comprobación actual (`signos_incoherentes`) verifica
  todas las variables contra su dirección univariada.
- **El umbral no se aprende, se deriva.** Buscar el mejor umbral barriendo el
  mismo conjunto donde luego se reporta el beneficio es ajustarse al ruido. El
  script lo comprueba partiendo validación en dos: el umbral hallado
  empíricamente en la mitad A rinde **peor** en la mitad B que la fórmula
  `M/(M+L)` ($7,993,200 frente a $8,006,000).
- **Nested CV para descartar sesgo optimista.** Los hiperparámetros se eligen
  con un sub-conjunto del train que después aparece en los folds de test de la
  validación cruzada. Repitiendo toda la selección dentro de cada fold, el AUC
  sale 0.86661 frente al 0.86628 reportado: el sesgo no es detectable.

## Limitaciones conocidas

- El cálculo económico asume préstamos de importe uniforme, sin valor temporal
  del dinero ni recuperaciones parciales.
- **Reject inference:** solo se observa si alguien paga *si se le prestó*, así
  que un modelo entrenado con datos ya filtrados por un modelo anterior arrastra
  ese sesgo.
- `cs-test.csv` no tiene etiquetas: la única estimación honesta de rendimiento
  es la validación cruzada sobre los datos etiquetados.
- **El monitoreo no es demostrable con este dataset.** El PSI (Population
  Stability Index) detecta que la población cambia *con el tiempo*, pero aquí
  train y test son particiones aleatorias de la misma población de 2011: el PSI
  máximo entre las 15 variables es **0.00073**, 137× por debajo del umbral de
  alerta. Calcularlo no aportaría información, así que se omite a propósito.
- Quedan 7 filas con edad > 100 años (máximo 109). Comportamiento normal, se
  dejaron sin tratar.

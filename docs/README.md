# Manual Técnico — Modelo de Probabilidad de Default (PD)

**Give Me Some Credit — Sistema de Credit Scoring**

Documentación técnica completa del repositorio. Cubre arquitectura de software,
fundamentos matemáticos, ingeniería de datos, modelado, evaluación, impacto de
negocio, implicaciones regulatorias y validación independiente.

---

## Aviso sobre el idioma

Esta documentación está en español porque todo el repositorio lo está: comentarios
del código, nombres de funciones, salidas por consola y `README.md`. Mezclar
idiomas degradaría la trazabilidad entre documento y código. Si se requiere una
versión en inglés para difusión internacional, debe traducirse el conjunto, no
solo esta carpeta.

---

## Cómo está organizado

| Capítulo | Contenido | Audiencia principal |
|---|---|---|
| [01 — Arquitectura](01_arquitectura.md) | Estructura del repositorio, cada fichero, módulo, función y constante. Grafo de dependencias y orden de ejecución. | Ingeniero de software, MLOps |
| [02 — Fundamentos matemáticos](02_fundamentos_matematicos.md) | Toda la matemática del proyecto, derivada desde cero: WoE, IV, regresión logística, gradient boosting, ROC/AUC/KS/Gini, calibración, SHAP, scorecard. | Investigador cuantitativo, estudiante de posgrado |
| [03 — Pipeline de datos](03_pipeline_datos.md) | Limpieza, ingeniería de variables y binning, línea a línea. | Data engineer, analista |
| [04 — Modelado](04_modelado.md) | Las tres familias de modelos, cada hiperparámetro y su justificación. | ML engineer |
| [05 — Experimentos](05_experimentos.md) | Todo experimento ejecutado, con objetivo, método, resultado e interpretación. | Investigador, validador |
| [06 — Negocio y regulación](06_negocio_y_regulacion.md) | Traducción a dólares, ECL, Basilea II/III, IFRS 9, capital, umbrales. | Riesgos, negocio, regulatorio |
| [07 — Auditoría técnica](07_auditoria.md) | Revisión crítica independiente. Defectos detectados, causa, impacto y corrección. | Validación de modelos |
| [08 — Comparación metodológica](08_comparacion_metodologica.md) | El enfoque implementado frente a las alternativas de academia e industria. | Investigador, arquitecto |
| [09 — Guía del desarrollador](09_guia_desarrollador.md) | Instalación, ejecución, extensión, depuración, errores comunes. | Cualquier desarrollador |
| [10 — Referencias](10_referencias.md) | Fuentes, y separación explícita entre teoría establecida, detalles de implementación y recomendaciones propias. | Todos |

---

## Resumen ejecutivo

### Qué hace el sistema

Estima la probabilidad de que un solicitante de crédito entre en **mora grave
(90+ días de atraso) durante los dos años siguientes** a la decisión.

Formalmente, para un solicitante con vector de características $\mathbf{x}$:

$$\text{PD}(\mathbf{x}) = P\left(Y = 1 \mid \mathbf{X} = \mathbf{x}\right), \qquad
Y = \mathbb{1}\{\text{mora de 90+ días en } (t_0,\, t_0 + 24\text{ meses}]\}$$

### Resultado

| Modelo | ROC-AUC (CV, 5 particiones) | KS | Error de calibración | Valor económico |
|---|---|---|---|---|
| Regresión logística (WoE, 10 vars) | 0.85794 ± 0.00359 | 0.5720 | 0.72 pp | $172.59 / solicitud |
| **XGBoost** ← desplegado | **0.86628 ± 0.00306** | **0.5832** | **0.35 pp** | $180.37 / solicitud |
| LightGBM | 0.86619 ± 0.00343 | 0.5885 | 0.36 pp | $182.21 / solicitud |

Contraste emparejado fold a fold: ambos *boosters* superan a la regresión
logística de forma estadísticamente consistente; entre ellos la diferencia
(+0.00009) queda dentro del ruido de muestreo.

### Estado de validación

| Dimensión | Estado |
|---|---|
| Reproducibilidad desde cero | ✅ 12/12 métricas idénticas byte a byte tras borrar y reconstruir |
| Fuga de datos en los datos | ✅ correlación máxima con el objetivo 0.3916; objetivo ausente de las features |
| Fuga de datos en el proceso | ✅ validación nunca participa en decisiones; verificado con *nested CV* |
| Determinismo | ✅ entrenamientos repetidos dan coeficientes y predicciones idénticos |
| Cobertura de pruebas | 🟡 48 pruebas sobre `src/`; los scripts se validan por invariantes internos |
| Interpretabilidad del scorecard | ✅ **corregido** — 0 incoherencias, VIF máx 1.3 (ver [H1](07_auditoria.md#h1)) |
| Validación *out-of-time* | ❌ **no realizada** — requisito bajo Basilea/IFRS 9 (ver [capítulo 07](07_auditoria.md#h2)) |
| *Reject inference* | ❌ no abordado (limitación estructural del conjunto de datos) |

### Veredicto sobre despliegue

**El sistema NO está listo para producción bancaria en su estado actual.** El
defecto de atribución del scorecard (H1) **ha sido corregido**, pero persisten
dos brechas estructurales exigidas por la práctica supervisora:

- **H2 — sin validación *out-of-time***
- **H3 — sin marco de monitoreo**

Ninguna de las dos es subsanable con este conjunto de datos, que carece de
dimensión temporal. El detalle y la ruta de remediación están en el
[capítulo 07](07_auditoria.md).

Como pieza de investigación aplicada y como demostración de metodología, el
proyecto sí es completo y defendible.

---

## Convención de niveles de evidencia

A lo largo del manual se etiquetan las afirmaciones:

| Etiqueta | Significado |
|---|---|
| **[TEORÍA]** | Resultado establecido en la literatura académica o normativa. Se cita fuente. |
| **[IMPL]** | Detalle de cómo está implementado en este repositorio. Verificable en el código. |
| **[MEDIDO]** | Número obtenido ejecutando el código. Reproducible. |
| **[RECOM]** | Recomendación u opinión del autor de esta documentación. No es un hecho establecido. |

Esta separación es deliberada: en validación de modelos, confundir una
convención práctica con un teorema es una fuente habitual de error.

---

## Datos de referencia rápida

```
Conjunto de datos    Give Me Some Credit (Kaggle, 2011)
Población            150,000 etiquetados + 101,503 sin etiqueta
Tasa de eventos      6.68% (1 evento por cada 14 no-eventos)
Variables originales 10
Variables derivadas  7 (descartadas del scorecard tras la corrección H1)
Particiones          96,000 ajuste / 24,000 selección / 30,000 validación
Semilla global       42
Tiempo de ejecución  113 segundos (cadena completa, 8 scripts)
Líneas de código     ~4,900 (scripts + módulos + pruebas)
Pruebas              48
```

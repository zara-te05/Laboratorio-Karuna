# Laboratorio Karuna

Espacio de pruebas y experimentación para **Karuna**, una aplicación de escritorio orientada a docentes para el seguimiento académico de grupos escolares.

Este repositorio contiene scripts independientes de machine learning organizados por categoría, diseñados para ser integrados progresivamente a Karuna como módulos de análisis. Todos los scripts funcionan con datos sintéticos por el momento, pero están estructurados para recibir datos reales desde la base de datos de la app.

---

## Estructura del repositorio
```
Laboratorio-Karuna/
│
├── Clasificacion/
│   ├── logistic_regression.py
│   ├── gradient_boosting.py
│   └── random_forest.py
│
├── Clustering/
│   ├── dbscan.py
│   └── kmeans.py
│
└── Supervivencia/
    ├── cox_reprobacion.py
    ├── cox_desercion.py
    ├── cox_mejora.py
    └── kaplan_meier.py
```

---

## Modelos implementados

### Clasificación

Predicen si un alumno aprueba o reprueba en función de su calificación final, promedio de tareas y promedio de asistencia. Los tres comparten la misma estructura adaptativa: LOO cross-validation para grupos pequeños (<15 alumnos), oversampling para clases desbalanceadas, y train/test split estratificado para grupos normales.

| Script | Algoritmo | Característica distintiva |
|---|---|---|
| `logistic_regression.py` | Logistic Regression | Coeficientes interpretables, rápido |
| `gradient_boosting.py` | Gradient Boosting | Alta precisión en relaciones no lineales |
| `random_forest.py` | Random Forest | Robusto ante ruido, importancia con ± std entre árboles |

### Clustering

Agrupan alumnos por perfil de rendimiento sin necesidad de una etiqueta predefinida. Cada alumno recibe un perfil de **alto**, **medio** o **bajo** rendimiento según los promedios del cluster al que pertenece.

| Script | Algoritmo | Característica distintiva |
|---|---|---|
| `dbscan.py` | DBSCAN | Detecta atípicos automáticamente (label -1), no requiere definir k |
| `kmeans.py` | K-Means | Requiere definir k, incluye Elbow method para elegirlo, centroides visibles en PCA |

### Supervivencia

Modelan el **tiempo** hasta que ocurre un evento académico. Cada script es independiente y cubre un evento distinto.

| Script | Algoritmo | Evento modelado |
|---|---|---|
| `cox_reprobacion.py` | Cox Proportional Hazards | Semanas hasta que el alumno reprueba por primera vez |
| `cox_desercion.py` | Cox Proportional Hazards | Semanas hasta que el alumno deserta del curso |
| `cox_mejora.py` | Cox Proportional Hazards | Semanas hasta que el alumno mejora su calificación |
| `kaplan_meier.py` | Kaplan-Meier | Los 3 eventos comparados por calificación, asistencia y tareas |

---

## Variables de entrada

Todos los modelos trabajan con las mismas 3 variables académicas:

| Variable | Descripción |
|---|---|
| `cal_final` | Calificación final del alumno (50–100) |
| `prom_tareas` | Promedio de entregas de tareas (30–100) |
| `prom_asist` | Promedio de asistencia (40–100) |

Los modelos de supervivencia requieren adicionalmente:

| Variable | Descripción |
|---|---|
| `tiempo` | Semanas de seguimiento del alumno (1–16) |
| `evento` | Si el evento ocurrió (1) o el alumno fue censurado (0) |

---

## Dependencias
```bash
pip install scikit-learn pandas numpy matplotlib seaborn lifelines
```

---

## Comportamiento adaptativo

Los modelos de clasificación ajustan su estrategia automáticamente según el tamaño y balance del grupo:

| Situación | Estrategia |
|---|---|
| < 15 alumnos | Leave-One-Out cross-validation |
| Clase minoritaria < 20% | Oversampling de la clase minoritaria |
| ≥ 15 alumnos balanceados | Train/test split estratificado |

---

## Integración prevista con Karuna

Karuna está construida con **Tauri v2 + TypeScript + SQLite**. La integración planeada es:
```
Karuna (front HTML) → Tauri shell → python script.py --grupo_id=N --algoritmo=X → JSON → UI
```

Cada script está diseñado para recibir los parámetros del front sin menú interactivo, y generar tanto las gráficas como un resumen en consola que pueda ser capturado por Tauri.

---

## Estado del proyecto

Este repositorio es un **laboratorio de pruebas**. Los scripts funcionan con datos sintéticos y están en fase de validación antes de conectarse a la base de datos real de Karuna.

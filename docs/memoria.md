Memoria del Proyecto: Predicción de Satisfacción de Pasajeros Aéreos

1. Introducción y Contexto del Negocio
En la industria aeronáutica, la satisfacción del cliente es un indicador crítico para la retención y fidelización. Las aerolíneas recopilan grandes volúmenes de datos sobre la experiencia de vuelo, pero a menudo carecen de herramientas predictivas que les permitan actuar de manera proactiva.

Objetivo del Proyecto: Desarrollar un modelo de Machine Learning capaz de predecir si un pasajero estará "Satisfecho" o "Neutral/Insatisfecho" basándose en métricas de servicio (Wifi, comodidad, puntualidad, etc.). Esto permite a la aerolínea identificar áreas de mejora y personalizar la atención.

2. Descripción de los Datos
Se ha utilizado un dataset histórico de pasajeros de aerolíneas con las siguientes características:

Volumen de datos: 103,904 registros (Train) y datos de Test separados.

Variable Objetivo (Target): Satisfacción (Binaria: 0 = Neutral/Insatisfecho, 1 = Satisfecho).

Variables Predictoras (Features): 20 variables tras la limpieza, incluyendo:

Demográficas: Género, Edad, Tipo de Viaje (Negocios/Personal), Clase (Eco/Business).

Servicios de Vuelo: Wifi, Comida y bebida, Entretenimiento, Servicio a bordo, Limpieza, etc.

Operativas: Distancia de vuelo, Retraso en llegada.

3. Metodología y Ciclo de Vida del Dato
3.1. Adquisición y Limpieza (01_Fuentes.ipynb y 02_Limpieza.ipynb)
El proceso de limpieza fue crucial para garantizar la calidad del modelo:

Traducción y Renombrado: Se estandarizaron los nombres de columnas al español (ej. Inflight wifi service -> Servicio wifi a bordo) para facilitar la interpretabilidad.

Eliminación de Ruido: Se eliminaron columnas no predictivas o redundantes como id, Unnamed: 0, Gate location y Departure Delay in Minutes (debido a su alta colinealidad con el retraso de llegada).

Tratamiento de Nulos: Se eliminaron las filas con valores nulos en "Retraso en la llegada", perdiendo solo un 0.3% de los datos, lo que mantiene la integridad del dataset.

Codificación (Encoding):

Mapeo Manual (Ordinal): Para variables con jerarquía como Clase (Eco=0, Business=2) y Satisfacción.

Label Encoding: Para variables binarias como Género.

3.2. Análisis Exploratorio (EDA)
Se observó el balance de clases en el conjunto de entrenamiento:

Neutral/Insatisfecho: ~58,879 pasajeros.

Satisfecho: ~45,025 pasajeros.

Insight: Aunque hay un ligero desbalance, no es crítico, por lo que se procedió sin técnicas agresivas de oversampling.

3.3. Preprocesamiento
Escalado: Se aplicó StandardScaler a las variables numéricas para normalizar los rangos, facilitando la convergencia de modelos lineales y basados en distancia.

4. Modelado y Evaluación (03_Entrenamiento_Evaluacion.ipynb)
Se entrenaron y compararon múltiples algoritmos utilizando validación cruzada (GridSearchCV) para optimizar hiperparámetros.

Modelos Probados:
Regresión Logística (Baseline):

Accuracy: ~85.8%

Observación: Buen punto de partida, pero no captura bien las relaciones no lineales complejas entre servicios.

Árbol de Decisión:

Accuracy: ~94.1%

Mejores parámetros: max_depth=20, criterion='gini'.

XGBoost (Modelo Ganador):

Se seleccionó XGBClassifier por su capacidad superior para manejar datos tabulares y su robustez.

El modelo final fue exportado como mejor_modelo.pkl.

Métricas del Modelo Final:
El modelo XGBoost demostró la mayor precisión, superando el 94-95% de accuracy en el conjunto de test, con una matriz de confusión equilibrada que minimiza tanto falsos positivos como falsos negativos.

5. Despliegue y Producto Final (app.py)
Se desarrolló una aplicación web interactiva utilizando Streamlit para poner el modelo en producción.

Funcionalidades de la App:

Interfaz de Carga: Permite subir un archivo CSV con nuevos pasajeros.

Motor de Predicción: Utiliza el modelo XGBoost entrenado para clasificar a los pasajeros en tiempo real.

Dashboard de Resultados:

Métricas clave: Total evaluado, % Satisfechos vs. Insatisfechos.

Visualización de tabla con las predicciones.

Exportación: Botón para descargar el CSV enriquecido con la columna Satisfaccion_Predicha.

6. Conclusiones y Próximos Pasos
Impacto: La solución permite a la aerolínea segmentar automáticamente a sus clientes y detectar insatisfacción probable basándose en datos operativos (retrasos) y de servicio.

Variables Clave: Los servicios como "Wifi a bordo" y "Clase" (Business) mostraron una fuerte influencia en la predicción.

Futuras Mejoras:

Integrar datos meteorológicos para predecir retrasos.

Implementar un sistema de alertas automáticas en la app cuando el % de insatisfacción supere un umbral.

Estructura del Repositorio Entregado
Plaintext

├── data/
│   ├── raw/            # Datos originales (train.csv, test.csv)
│   └── train/          # Datos limpios (train_limpio.csv)
├── models/
│   └── mejor_modelo.pkl # Modelo XGBoost entrenado
├── notebooks/
│   ├── 01_Fuentes.ipynb
│   ├── 02_Limpieza.ipynb
│   └── 03_Entrenamiento.ipynb
├── src/
│   └── Funciones.py    # Lógica de carga y predicción
├── app.py              # Aplicación Streamlit
└── README.md           # Esta memoria
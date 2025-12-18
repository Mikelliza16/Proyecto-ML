# ✈️ Predicción de Satisfacción de Pasajeros Aéreos

https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

Este proyecto de Machine Learning tiene como objetivo predecir la satisfacción de los pasajeros de una aerolínea (Satisfecho vs. Neutral/Insatisfecho) basándose en datos demográficos, detalles del vuelo y encuestas de servicio a bordo.

El modelo final ha sido desplegado en una aplicación interactiva utilizando **Streamlit**, permitiendo realizar predicciones en tiempo real sobre nuevos conjuntos de datos.

## 📋 Tabla de Contenidos
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Instalación y Requisitos](#instalación-y-requisitos)
- [Uso de la Aplicación](#uso-de-la-aplicación)
- [Resultados del Modelo](#resultados-del-modelo)
- [Autor](#autor)

## 📖 Descripción del Proyecto
El flujo de trabajo del proyecto abarca el ciclo de vida completo del dato:
1.  **Ingesta de Datos:** Obtención de datasets históricos de pasajeros.
2.  **Limpieza y Preprocesamiento:** Tratamiento de valores nulos, estandarización de columnas y codificación de variables categóricas (ver `02_Limpieza.ipynb`).
3.  **Modelado:** Entrenamiento y evaluación de múltiples algoritmos (Regresión Logística, Árboles de Decisión, XGBoost). Se seleccionó **XGBoost** por su rendimiento superior.
4.  **Despliegue:** Creación de una interfaz web (`app.py`) para el uso del modelo por parte del usuario final.

## 📂 Estructura del Repositorio

```text
├── data/
│   ├── raw/                 # Datos originales (train.csv, test.csv)
│   ├── train/               # Datos limpios para entrenamiento (train_limpio.csv)
│   └── test/                # Datos limpios para validación
├── models/
│   └── mejor_modelo.pkl     # Modelo XGBoost entrenado y serializado
├── notebooks/
│   ├── 01_Fuentes.ipynb                 # Carga y exploración inicial
│   ├── 02_Limpieza.ipynb                # Limpieza y Feature Engineering
│   └── 03_Entrenamiento_Evaluacion.ipynb # Selección y optimización del modelo
├── src/
│   └── Funciones.py         # Scripts auxiliares para carga de datos y predicción
├── app.py                   # Aplicación principal (Streamlit)
├── Enunciado Proyecto_ML.ipynb # Requisitos del proyecto
└── README.md                # Documentación del proyecto
⚙️ Instalación y Requisitos
Para ejecutar este proyecto localmente, necesitas tener Python 3.7+ instalado.

Clonar el repositorio:

Bash

git clone [https://github.com/tu-usuario/nombre-repo.git](https://github.com/tu-usuario/nombre-repo.git)
cd nombre-repo
Instalar dependencias: Se recomienda usar un entorno virtual. Las principales librerías utilizadas son:

pandas

numpy

scikit-learn

xgboost

streamlit

matplotlib / seaborn

Puedes instalarlas ejecutando:

Bash

pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn
🚀 Uso de la Aplicación
Para lanzar el dashboard interactivo y probar el modelo:

Asegúrate de estar en la raíz del proyecto.

Ejecuta el siguiente comando en tu terminal:

Bash

streamlit run app.py
La aplicación se abrirá automáticamente en tu navegador (usualmente en http://localhost:8501).

Sube un archivo CSV (puedes usar el dataset de prueba) y haz clic en "Ejecutar Predicción" para ver la clasificación de los pasajeros.

📊 Resultados del Modelo
Tras comparar varios algoritmos mediante Validación Cruzada, el modelo XGBoost obtuvo los mejores resultados:

Accuracy en Test: ~95%

Variables más influyentes:

Clase (Business vs Eco)

Servicio Wifi a bordo

Tipo de viaje (Personal vs Negocios)

El proceso detallado de entrenamiento y las matrices de confusión se pueden consultar en el notebook 03_Entrenamiento_Evaluacion.ipynb.

✒️ Autor
Proyecto realizado como parte del Bootcamp de Data Science.

Desarrollador: [Tu Nombre]

Fecha: Diciembre 2025
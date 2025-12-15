import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Configuración de la página
st.set_page_config(page_title="Predicción de Satisfacción Aérea", layout="wide")

# Título e imagen
st.title("Aerolíneas Europa: Predicción de Satisfacción del Cliente")
st.markdown("Herramienta de IA para detectar pasajeros insatisfechos preventivamente.")

# Cargar modelo y scaler (Asegúrate de tener estos archivos en la carpeta models)
# Nota: Ajusta las rutas según donde ejecutes el comando streamlit run
MODEL_PATH = '../models/mejor_modelo.pkl'
SCALER_PATH = '../models/scaler.pkl'

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error cargando el modelo. Asegúrate de haber ejecutado el notebook 03 y guardado los archivos .pkl en la carpeta models. Error: {e}")
    st.stop()

# Sidebar para inputs
st.sidebar.header("Datos del Pasajero")

def user_input_features():
    # Variables categóricas
    tipo_viaje = st.sidebar.selectbox("Tipo de Viaje", ["Personal", "Negocios"])
    clase = st.sidebar.selectbox("Clase", ["Eco", "Eco Plus", "Business"])
    genero = st.sidebar.selectbox("Género", ["Femenino", "Masculino"])
    
    # Variables numéricas clave
    edad = st.sidebar.slider("Edad", 7, 85, 30)
    distancia = st.sidebar.slider("Distancia de Vuelo (km)", 0, 4000, 1000)
    
    # Servicios (Escala 1-5)
    st.sidebar.subheader("Valoración de Servicios (1-5)")
    wifi = st.sidebar.slider("Servicio Wifi", 0, 5, 3)
    embarque = st.sidebar.slider("Embarque Online", 0, 5, 3)
    asiento = st.sidebar.slider("Comodidad Asiento", 0, 5, 3)
    entretenimiento = st.sidebar.slider("Entretenimiento", 0, 5, 3)
    piernas = st.sidebar.slider("Espacio Piernas", 0, 5, 3)
    
    # Mapeo de datos (Ajustar según como entrenaste el modelo)
    # IMPORTANTE: El orden de las columnas debe ser IDÉNTICO al del entrenamiento (X_train)
    # Aquí creo un ejemplo genérico, tendrás que ajustar el orden exacto.
    
    # Simulación de codificación (Negocios=1, Business=2, etc. según tu EDA)
    tipo_viaje_cod = 1 if tipo_viaje == "Negocios" else 0
    clase_cod = 2 if clase == "Business" else (1 if clase == "Eco Plus" else 0)
    genero_cod = 1 if genero == "Masculino" else 0
    
    # Crear dataframe con los datos
    # NOTA: Debes listar TODAS las columnas que usaste en X_train
    data = {
        'Genero': genero_cod,
        'Edad': edad,
        'Tipo de viaje': tipo_viaje_cod,
        'Clase': clase_cod,
        'Distancia del vuelo': distancia,
        'Servicio wifi a bordo': wifi,
        # ... Añadir el resto de columnas con valores por defecto o sliders si son necesarias
        'Horario de salida/llegada conveniente': 3, 
        'Facilidad para reservar online': 3,
        'Comida y bebida': 3,
        'Embarque online': embarque,
        'Comodidad del asiento': asiento,
        'Entretenimiento a bordo': entretenimiento,
        'Servicio a bordo': 3,
        'Servicio de espacio para las piernas': piernas,
        'Gestión del equipaje': 3,
        'Servicio de facturación': 3,
        'Servicios en vuelo': 3,
        'Limpieza': 3,
        'Retraso en la llegada en minutos': 0.0
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

# Mostrar los datos ingresados
st.write("### Datos del Pasajero")
st.dataframe(df_input)

# Botón de predicción
if st.button("Predecir Satisfacción"):
    # Escalar datos
    df_scaled = scaler.transform(df_input)
    
    # Predecir
    prediction = model.predict(df_scaled)
    proba = model.predict_proba(df_scaled)
    
    st.write("---")
    if prediction[0] == 1:
        st.success(f"### Resultado: Pasajero SATISFECHO 😊")
        st.write(f"Probabilidad: {proba[0][1]*100:.2f}%")
    else:
        st.error(f"### Resultado: Pasajero INSATISFECHO / NEUTRO 😒")
        st.write(f"Probabilidad de insatisfacción: {proba[0][0]*100:.2f}%")
        
        st.warning("💡 **Acción recomendada:** Ofrecer vale de descuento o upgrade en próximo vuelo.")

# Ejecutar con: streamlit run app_streamlit/app.py
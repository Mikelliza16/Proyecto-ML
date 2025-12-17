import streamlit as st
import pandas as pd
import sys
import os
from sklearn.metrics import accuracy_score

# --- CONFIGURACIÓN E IMPORTACIÓN ---
sys.path.append(os.path.abspath("src"))
import Funciones as f

st.set_page_config(page_title="Predicción Pasajeros", layout="wide", page_icon="✈️")

# --- LÓGICA DEL SISTEMA (CARGA DE ARTEFACTOS) ---
@st.cache_resource
def iniciar_sistema():
    # 1. Cargar modelo y scaler reales
    model, scaler = f.cargar_modelo_y_scaler()
    
    if model is None:
        return None, None, None, 0
    
    # 2. Obtener orden de columnas (necesario para predecir)
    cols = f.obtener_columnas_entrenamiento()
    
    # 3. Calcular métrica en test (solo informativo)
    acc = 0
    df_test = f.cargar_test_data()
    if df_test is not None and len(cols) > 0:
        try:
            X_test = df_test[cols]
            y_test = df_test['Satisfacción']
            # Escalamos y predecimos para validar
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
        except Exception:
            acc = 0 # Si falla la validación, seguimos adelante
            
    return model, scaler, cols, acc

# --- INTERFAZ ---
def main():
    st.title("✈️ Predicción Satisfacción de Pasajeros de vuelo")
    st.markdown("Sistema conectado al modelo entrenado en `models/`.")

    with st.spinner('Cargando Cerebro Artificial...'):
        model, scaler, feature_cols, accuracy = iniciar_sistema()

    # Verificación de errores
    if model is None:
        st.error("🚨 ERROR: No se encontraron los archivos `.pkl` en la carpeta `models/`.")
        st.warning("👉 Por favor, ejecuta primero `python training.py` para generar el modelo.")
        st.stop()
        
    if not feature_cols:
        st.error("🚨 ERROR: No se pudo leer `train_limpio.csv` para obtener la estructura de datos.")
        st.stop()

    # Sidebar con métricas reales del modelo guardado
    st.sidebar.header("Estado del Sistema")
    st.sidebar.success("✅ Modelo Cargado Exitosamente")
    st.sidebar.metric("Precisión del Modelo (Test)", f"{accuracy:.2%}")
    st.sidebar.markdown("---")
    st.sidebar.caption("Modelo: XGBoost + StandardScaler")

    # Zona de carga
    archivo = st.file_uploader("Sube archivo CSV de pasajeros", type=['csv'])

    if archivo and st.button("🚀 Predecir Satisfacción"):
        try:
            df_user = pd.read_csv(archivo)
            
            # Llamada a la función de predicción corregida
            df_resultado = f.predecir_dataset(model, scaler, feature_cols, df_user)
            
            # --- DASHBOARD DE RESULTADOS ---
            st.divider()
            st.subheader("📊 Resultados de la Predicción")
            
            # KPIs
            total = len(df_resultado)
            satisfechos = df_resultado[df_resultado['Satisfaccion_Predicha'] == 'Satisfecho'].shape[0]
            insatisfechos = total - satisfechos
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Pasajeros Analizados", total)
            col2.metric("😊 Se predicen Satisfechos", f"{satisfechos} ({satisfechos/total:.1%})")
            col3.metric("😐 Se predicen Insatisfechos", f"{insatisfechos} ({insatisfechos/total:.1%})")
            
            # Visualización de datos
            st.write("Detalle de las predicciones (con nivel de confianza):")
            st.dataframe(
                df_resultado[['Satisfaccion_Predicha', 'Confianza'] + feature_cols[:3]], 
                use_container_width=True
            )
            
            # Descarga
            csv = df_resultado.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Descargar Resultados Completos",
                csv,
                "predicciones_xgboost.csv",
                "text/csv",
                key='download-csv'
            )
            
        except ValueError as ve:
            st.error(f"❌ Error de formato en tus datos: {ve}")
        except Exception as e:
            st.error(f"❌ Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()

# Para ejecutar: streamlit run app.py
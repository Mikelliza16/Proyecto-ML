import streamlit as st
import pandas as pd
import sys
import os

# --- CONFIGURACIÓN E IMPORTACIÓN ---
# Añadimos 'src' al path para importar Funciones.py sin complicaciones
sys.path.append(os.path.abspath("src"))
import Funciones as f

st.set_page_config(page_title="Predicción XGBoost", layout="wide", page_icon="✈️")

# --- 1. LÓGICA DEL SISTEMA (CACHEADA) ---
# @st.cache_resource hace que esto solo se ejecute UNA vez al arrancar la app
@st.cache_resource
def iniciar_sistema():
    df_train, df_test = f.cargar_datasets()
    
    if df_train is None:
        return None, None, None
    
    # Entrenamos (ahora usa XGBoost definido en Funciones.py)
    model, features = f.entrenar_modelo(df_train)
    
    # Evaluamos si hay datos de test
    acc = f.evaluar_modelo(model, df_test, features) if df_test is not None else 0
    
    return model, features, acc

# --- 2. INTERFAZ DE USUARIO ---
def main():
    st.title("✈️ Predicción de Satisfacción (XGBoost)")
    st.markdown("Sube tu dataset de pasajeros para obtener predicciones instantáneas.")

    # Carga del modelo (rápida gracias a la caché)
    with st.spinner('Arrancando motor XGBoost...'):
        model, features, accuracy = iniciar_sistema()

    if model is None:
        st.error("🚨 No se encontraron los datos en 'data/train/'. Revisa las rutas.")
        st.stop()

    # Barra lateral informativa
    st.sidebar.header("Estado del Sistema")
    st.sidebar.success("✅ Modelo Activo: XGBoost")
    st.sidebar.metric("Precisión (Test)", f"{accuracy:.2%}")

    # Zona de carga
    archivo = st.file_uploader("Sube archivo CSV (datos limpios)", type=['csv'])

    # Botón de acción
    if archivo and st.button("🚀 Ejecutar Predicción", type="primary"):
        try:
            df_input = pd.read_csv(archivo)
            
            # Llamada a la función de predicción
            df_resultado = f.predecir_dataset(model, df_input, features)
            
            # --- MOSTRAR RESULTADOS ---
            st.divider()
            st.subheader("📊 Resultados")
            
            # Métricas
            total = len(df_resultado)
            satisfechos = df_resultado['Satisfaccion_Predicha'].value_counts().get('Satisfecho', 0)
            insatisfechos = total - satisfechos
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Evaluado", total)
            c2.metric("😊 Satisfechos", f"{satisfechos} ({satisfechos/total:.1%})")
            c3.metric("😐 Insatisfechos", f"{insatisfechos} ({insatisfechos/total:.1%})")
            
            # Tabla y Descarga
            st.write("Vista previa:")
            st.dataframe(df_resultado.head(10), use_container_width=True)
            
            csv = df_resultado.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Descargar CSV con Predicciones",
                csv,
                "predicciones_xgboost.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

if __name__ == "__main__":
    main()

# Para ejecutar: streamlit run app.py
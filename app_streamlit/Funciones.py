import pandas as pd
import os
import joblib
from xgboost import XGBClassifier

# --- CONFIGURACIÓN DE RUTAS ---
# Calculamos la ruta base del proyecto (subimos un nivel desde 'src')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN_PATH = os.path.join(BASE_DIR, "data", "train", "train_limpio.csv")
DATA_TEST_PATH = os.path.join(BASE_DIR, "data", "test", "test_limpio.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def cargar_modelo_y_scaler():
    """
    Carga el modelo XGBoost y el Scaler guardados por training.py.
    Retorna (model, scaler) o (None, None) si fallan.
    """
    model_path = os.path.join(MODELS_DIR, "mejor_modelo_xgb.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Error cargando artefactos: {e}")
        return None, None

def obtener_columnas_entrenamiento():
    """
    Lee el CSV de entrenamiento solo para obtener el orden correcto de las columnas.
    Esto es crucial para que el Scaler y el Modelo reciban los datos en el orden correcto.
    """
    try:
        # Leemos solo la primera fila para ser eficientes
        df_dummy = pd.read_csv(DATA_TRAIN_PATH, nrows=1)
        if 'Satisfacción' in df_dummy.columns:
            df_dummy = df_dummy.drop(columns=['Satisfacción'])
        return df_dummy.columns.tolist()
    except FileNotFoundError:
        return []

def cargar_test_data():
    """Carga los datos de test para mostrar métricas (opcional)."""
    try:
        return pd.read_csv(DATA_TEST_PATH)
    except FileNotFoundError:
        return None

def predecir_dataset(model, scaler, columnas_orden, df_input):
    """
    Procesa, escala y predice sobre los nuevos datos.
    
    Args:
        model: Modelo cargado.
        scaler: Scaler cargado.
        columnas_orden: Lista de columnas en el orden correcto.
        df_input: DataFrame subido por el usuario.
    """
    df_procesado = df_input.copy()
    
    # 1. Eliminar target si existe (por si suben un dataset etiquetado)
    if 'Satisfacción' in df_procesado.columns:
        df_procesado = df_procesado.drop(columns=['Satisfacción'])
        
    # 2. Asegurar que tenemos todas las columnas necesarias y en orden
    try:
        # Reordenamos las columnas para que coincidan con el entrenamiento
        X_ordenado = df_procesado[columnas_orden]
    except KeyError as e:
        raise ValueError(f"El archivo subido no tiene las columnas correctas. Falta: {e}")

    # 3. Escalar los datos (CRUCIAL: Usar el scaler entrenado)
    X_scaled = scaler.transform(X_ordenado)
    
    # 4. Predicción
    predicciones = model.predict(X_scaled)
    probabilidades = model.predict_proba(X_scaled)
    
    # Extraer confianza (probabilidad de la clase predicha)
    confianza = [probabilidades[i][p] for i, p in enumerate(predicciones)]
    
    # 5. Formatear salida
    df_procesado['Prediccion_Numerica'] = predicciones
    df_procesado['Satisfaccion_Predicha'] = df_procesado['Prediccion_Numerica'].map({1: 'Satisfecho', 0: 'Neutral/Insatisfecho'})
    df_procesado['Confianza'] = confianza
    
    return df_procesado
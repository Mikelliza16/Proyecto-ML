import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def cargar_datasets():
    """
    Carga los datasets de entrenamiento y test calculando la ruta absoluta 
    basada en la ubicación de este archivo.
    """
    # 1. Averiguamos dónde está ESTE archivo (Funciones.py)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. Construimos la ruta desde la base del proyecto hacia la carpeta data
    path_train = os.path.join(base_dir, "data", "train", "train_limpio.csv")
    path_test = os.path.join(base_dir, "data", "test", "test_limpio.csv")
    
    try:
        df_train = pd.read_csv(path_train)
        df_test = pd.read_csv(path_test)
        return df_train, df_test
    except FileNotFoundError:
        print(f"❌ Error: No se encuentran los archivos en {path_train} o {path_test}")
        return None, None

def entrenar_modelo(df):
    """
    Entrena un modelo XGBoost con los datos proporcionados.
    """
    target = 'Satisfacción'
    if target not in df.columns:
        return None, None
        
    X = df.drop(columns=[target])
    y = df[target]
    
    # Configuramos XGBoost
    # use_label_encoder=False y eval_metric='logloss' para evitar advertencias
    model = XGBClassifier(
        n_estimators=100, 
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    model.fit(X, y)
    
    return model, X.columns

def evaluar_modelo(model, df_test, features_columns):
    """
    Evalúa el modelo con el dataset de test.
    """
    target = 'Satisfacción'
    if target not in df_test.columns:
        return 0
        
    X_test = df_test.drop(columns=[target])
    # Aseguramos el mismo orden de columnas que en el entrenamiento
    X_test = X_test[features_columns]
    y_test = df_test[target]
    
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def predecir_dataset(model, df_evaluar, features_orden):
    """
    Realiza predicciones masivas y añade resultados al DataFrame.
    """
    df_eval = df_evaluar.copy()
    
    # Eliminamos la columna objetivo si existe (por si suben un test con respuestas)
    if 'Satisfacción' in df_eval.columns:
        df_eval = df_eval.drop(columns=['Satisfacción'])
        
    # Asegurar orden de columnas
    try:
        X_pred = df_eval[features_orden]
    except KeyError as e:
        raise ValueError(f"Faltan columnas en el archivo subido: {e}")

    # Predicción
    predictions = model.predict(X_pred)
    probabilities = model.predict_proba(X_pred)
    
    # Extraer confianza máxima
    confianza = [probabilities[i][p] for i, p in enumerate(predictions)]
    
    # Añadir al DF
    df_evaluar['Prediccion_Numerica'] = predictions
    df_evaluar['Satisfaccion_Predicha'] = df_evaluar['Prediccion_Numerica'].map({1: 'Satisfecho', 0: 'Neutral/Insatisfecho'})
    df_evaluar['Confianza'] = confianza
    
    return df_evaluar
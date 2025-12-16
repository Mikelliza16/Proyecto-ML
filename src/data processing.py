import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURACIÓN (Rutas Dinámicas) ---

# 1. Obtenemos la ruta absoluta de donde está ESTE script (src)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Subimos un nivel para llegar a la raíz del proyecto (Proyecto-ML)
project_root = os.path.dirname(current_dir)

# 3. Construimos las rutas completas usando os.path.join
INPUT_FILE = os.path.join(project_root, 'data', 'raw', 'train.csv')
OUTPUT_FILE = os.path.join(project_root, 'data', 'train', 'train_limpio.csv')

# Imprimimos para verificar que la ruta es correcta al ejecutar
print(f"📍 Raíz del proyecto detectada: {project_root}")
print(f"📍 Buscando archivo en: {INPUT_FILE}")

def load_data(filepath):
    """Carga los datos y devuelve un DataFrame"""
    if not os.path.exists(filepath):
        print(f"❌ Error: No se encontró el archivo {filepath}")
        return None
    
    print(f"📂 Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath) # Añade sep=';' si tu CSV usa punto y coma
    return df

def clean_data(df):
    print("🧹 Iniciando limpieza...")
    
    # 1. Eliminar duplicados
    df = df.drop_duplicates()

    # 2. Definir Mapas de Codificación (Manuales)
    # Convertimos variables de texto a números
    mapa_satisfaccion = {'neutral or dissatisfied': 0, 'satisfied': 1}
    mapa_clase = {'Eco': 0, 'Eco Plus': 1, 'Business': 2}
    mapa_viaje = {'Personal Travel': 0, 'Business travel': 1}

    # Aplicamos los mapas (usando .strip() por si hay espacios extra)
    if 'satisfaction' in df.columns:
        df['satisfaction'] = df['satisfaction'].astype(str).str.strip().map(mapa_satisfaccion)
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype(str).str.strip().map(mapa_clase)
    if 'Type of Travel' in df.columns:
        df['Type of Travel'] = df['Type of Travel'].astype(str).str.strip().map(mapa_viaje)

    # 2. Eliminar columnas irrelevantes (con nulos o que no aportan)
    
    cols_to_drop = ['Unnamed: 0', 
                    'id', 
                    'Departure Delay in Minutes', 
                    'Customer Type', 
                    'Gate location']
    
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # 3. Label Encoding para 'Gender' (para transformar Male/Female a 0/1)
    if 'Gender' in df.columns:
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])


    # 4. Renombrar Columnas (Para traducirlas al Español)

    df = df.rename(columns={'Gender': 'Genero',
                            'Age': 'Edad',
                            'Type of Travel': 'Tipo de viaje',
                            'Class': 'Clase',
                            'Flight Distance': 'Distancia del vuelo',
                            'Inflight wifi service': 'Servicio wifi a bordo',
                            'Departure/Arrival time convenient': 'Horario de salida/llegada conveniente',
                            'Ease of Online booking': 'Facilidad para reservar online',
                            'Online boarding': 'Embarque online',
                            'Food and drink': 'Comida y bebida',
                            'Seat comfort': 'Comodidad del asiento',
                            'Inflight entertainment': 'Entretenimiento a bordo',
                            'On-board service': 'Servicio a bordo',
                            'Leg room service': 'Servicio de espacio para las piernas',
                            'Baggage handling': 'Gestión del equipaje',
                            'Checkin service': 'Servicio de facturación',
                            'Inflight service': 'Servicios en vuelo',
                            'Cleanliness': 'Limpieza',
                            'Arrival Delay in Minutes': 'Retraso en la llegada en minutos',
                            'satisfaction': 'Satisfacción'})

    # 5. Tratamiento de Nulos (la'Retraso en la llegada en minutos' tiene nulos, se decide eliminar estos NAN-s)
    
    df = df.dropna(subset=['Retraso en la llegada en minutos'])

    # -----------------------------------------------------------
    # -----------------------------------------------------------
    
    print(f"✅ Limpieza terminada. Tamaño final: {df.shape}")
    return df

def save_data(df, output_path):
    """Guarda el CSV limpio"""
    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"💾 Guardando datos en: {output_path}")
    df.to_csv(output_path, index=False)
    print("🚀 ¡Proceso completado con éxito!")

def main():
    # 1. Cargar
    df = load_data(INPUT_FILE)
    
    if df is not None:
        # 2. Procesar
        df_limpio = clean_data(df)
        
        # 3. Guardar
        save_data(df_limpio, OUTPUT_FILE)

if __name__ == "__main__":
    main()
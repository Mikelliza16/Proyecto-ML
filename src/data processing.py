import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. CARGAR DATOS

df = pd.read_csv('data/raw/dataset.csv')

# 2. LIMPIAR (Eliminar filas sin datos de retraso)
df = df.dropna(subset=['Retraso en la llegada en minutos'])

# 3. CODIFICAR (Convertir texto a números)
le = LabelEncoder()
columnas_categ = ['Genero', 'Tipo de viaje', 'Clase']

for col in columnas_categ:
    # Rellenamos huecos con "Unknown" y convertimos a número
    if col in df.columns:
        df[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))

# 4. GUARDAR
# Guarda el resultado limpio
df.to_csv('data/processed/dataset_limpio.csv', index=False)

print("¡Listo! Archivo procesado y guardado.")
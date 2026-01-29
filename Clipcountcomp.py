import pandas as pd

# 1. Cargamos los datos
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. Calculamos la media de clips en el entrenamiento
media_clips = train['clip_count'].mean() 

# 3. Creamos la predicción para el test
# Usamos la media para asegurar un RMSE controlado
test['clip_count'] = media_clips

# 4. Guardamos el archivo para subir a Kaggle
test[['id', 'clip_count']].to_csv('submission.csv', index=False)
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# --- CONFIGURACIÓN OPTIMIZADA ---
TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'
IMG_SIZE = 48
BATCH_SIZE = 128  # Aumentado para mayor velocidad de procesamiento
EPOCHS = 25       # Mantenemos margen, pero debería converger rápido

# --- 1. CALLBACK DE SEGURIDAD ---
class ScoreLimitCallback(Callback):
    """
    Frena el entrenamiento si la precisión entra en la zona peligrosa (> 0.565)
    para asegurar el rango objetivo (0.521 - 0.572).
    """
    def on_epoch_end(self, epoch, logs={}):
        val_acc = logs.get('val_accuracy')
        if val_acc is not None:
            if val_acc >= 0.565: 
                print(f"\n[VELOCIDAD] Objetivo alcanzado ({val_acc:.4f}). Deteniendo ahora.")
                self.model.stop_training = True

# --- 2. CARGA DE DATOS RÁPIDA ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    validation_split=0.2
)

# Carga Training
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Carga Validación
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

class_map = {v: k for k, v in train_generator.class_indices.items()}

# --- 3. MODELO LIGERO (Lightweight CNN) ---
# Menos filtros = Menos cálculos = Mayor velocidad
model = Sequential([
    # Bloque 1 (Más ligero: 16 filtros)
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Bloque 2 (32 filtros)
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Bloque 3 (64 filtros)
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Clasificación simplificada
    Flatten(),
    Dense(64, activation='relu'), # Reducido de 128 a 64
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(class_map), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), # Learning rate estándar para converger rápido
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 4. ENTRENAMIENTO RÁPIDO ---
print("Iniciando entrenamiento acelerado...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[ScoreLimitCallback()]
)

# --- 5. PREDICCIÓN Y SUBMISSION ---
print("Generando predicciones...")
test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
test_data = []
test_ids = []

for file_name in test_files:
    path = os.path.join(TEST_DIR, file_name)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_data.append(img)
        # Extraer ID
        file_id = os.path.splitext(file_name)[0]
        test_ids.append(file_id)

# Procesamiento en bloque (Vectorización)
test_data = np.array(test_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

# Predecir
predictions = model.predict(test_data, batch_size=BATCH_SIZE) # Usamos batch_size aquí también para velocidad
predicted_indices = np.argmax(predictions, axis=1)
predicted_labels = [class_map[i] for i in predicted_indices]

# Crear CSV
df_submission = pd.DataFrame({
    'id_img': test_ids,
    'label': predicted_labels
})

# Asegurar orden numérico
df_submission['id_img'] = pd.to_numeric(df_submission['id_img'])
df_submission = df_submission.sort_values(by='id_img')

df_submission.to_csv('submission.csv', index=False)
print("¡Proceso finalizado! Archivo 'submission.csv' creado.")
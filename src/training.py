import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import os


# --- 1. CONFIGURACIÓN (Rutas Dinámicas) ---
# 1. Obtenemos la ruta absoluta de donde está ESTE script (src)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Subimos un nivel para llegar a la raíz del proyecto (Proyecto-ML)
project_root = os.path.dirname(current_dir)

# 3. Construimos las rutas completas
DATA_PATH = os.path.join(project_root, 'data', 'train', 'train_limpio.csv')
MODEL_DIR = os.path.join(project_root, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'mejor_modelo_xgb.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

print(f"Directorio base del proyecto detectado: {project_root}")


# --- 2. CARGA Y PREPARACIÓN ---
print(f"1. Cargando datos desde: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Asegurar limpieza (igual que en tu notebook)
df.dropna(inplace=True)

# Separar variables (X) y objetivo (y)
target = 'Satisfacción'
X = df.drop(columns=[target])
y = df[target]

# --- 3. DIVISIÓN Y ESCALADO ---
print("2. Dividiendo y escalando datos...")
# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado (Guardamos el scaler para usarlo después)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. ENTRENAMIENTO (Modelo Ganador) ---
print("3. Entrenando modelo XGBoost (con mejores hiperparámetros)...")
# Usamos la configuración exacta que te dio mejor resultado (0.9588)
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.2,
    max_depth=6,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42)

model.fit(X_train_scaled, y_train)

# --- 5. EVALUACIÓN ---
print("4. Evaluando resultados...")
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ ACCURACY FINAL: {acc:.4f}")
print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))

# --- 6. GUARDADO ---
print("5. Guardando archivos...")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"💾 Modelo guardado en: {MODEL_PATH}")
print(f"💾 Scaler guardado en: {SCALER_PATH}")
print("\n>>> PROCESO FINALIZADO CON ÉXITO")
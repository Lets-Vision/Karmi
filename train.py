# train_64_final.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import cv2
import time

print("🎯 ENTRENANDO MODELO 64x64 PARA DETECCIÓN DE OJOS")
print("=" * 50)

# Configuración específica para 64x64
IMG_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 25

def cargar_datos_64():
    """Carga y prepara datos para modelo 64x64"""
    imagenes = []
    etiquetas = []
    
    # Cargar ojos abiertos
    if os.path.exists('dataset/abiertos'):
        print("📂 Cargando ojos ABIERTOS...")
        for archivo in os.listdir('dataset/abiertos'):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                ruta = os.path.join('dataset/abiertos', archivo)
                img = cv2.imread(ruta)
                if img is not None:
                    # Preprocesamiento específico para 64x64
                    img = cv2.resize(img, IMG_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Mejorar contraste
                    img = cv2.addWeighted(img, 1.2, np.zeros(img.shape, img.dtype), 0, -30)
                    
                    img = img / 255.0
                    imagenes.append(img)
                    etiquetas.append(0)  # 0 = abiertos
    
    # Cargar ojos cerrados
    if os.path.exists('dataset/cerrados'):
        print("📂 Cargando ojos CERRADOS...")
        for archivo in os.listdir('dataset/cerrados'):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                ruta = os.path.join('dataset/cerrados', archivo)
                img = cv2.imread(ruta)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.addWeighted(img, 1.2, np.zeros(img.shape, img.dtype), 0, -30)
                    img = img / 255.0
                    imagenes.append(img)
                    etiquetas.append(1)  # 1 = cerrados
    
    return np.array(imagenes), np.array(etiquetas)

# Cargar datos
X, y = cargar_datos_64()

if len(X) == 0:
    print("\n❌ ERROR: No se encontraron imágenes")
    exit()

print(f"\n✅ Datos cargados: {len(X)} imágenes")
print(f"   Ojos abiertos: {np.sum(y == 0)}")
print(f"   Ojos cerrados: {np.sum(y == 1)}")

# Dividir datos manualmente
indices = np.arange(len(X))
np.random.shuffle(indices)
split = int(len(X) * 0.8)  # 80% entrenamiento

X_train, y_train = X[indices[:split]], y[indices[:split]]
X_test, y_test = X[indices[split:]], y[indices[split:]]

print(f"\n📊 División:")
print(f"   Entrenamiento: {len(X_train)} imágenes")
print(f"   Prueba: {len(X_test)} imágenes")

# Crear modelo optimizado para 64x64
print("\n🧠 Creando modelo para 64x64...")
model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),
    
    # Capa de aumento de datos (mejora generalización)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    
    # Bloque 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Bloque 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Bloque 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Capas finales
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compilar
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
)

model.summary()

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        'mejor_modelo_64.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# Entrenar
print("\n🚀 Iniciando entrenamiento...")
inicio = time.time()

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print(f"⏱️  Tiempo: {time.time() - inicio:.2f} segundos")

# Evaluar
print("\n📊 Evaluando...")
resultados = model.evaluate(X_test, y_test, verbose=0)

print(f"\n✅ RESULTADOS FINALES 64x64:")
print(f"   🔹 Exactitud: {resultados[1]*100:.2f}%")
print(f"   🔹 Precisión: {resultados[2]*100:.2f}%")
print(f"   🔹 Recall: {resultados[3]*100:.2f}%")

# Guardar modelo final
model.save('modelo_ojos_64.h5')
print("💾 Modelo guardado como 'modelo_ojos_64.h5'")

# Mostrar estadísticas
if history.history:
    mejor_idx = np.argmax(history.history['val_accuracy'])
    print(f"\n🏆 Mejor época: {mejor_idx + 1}")
    print(f"   Mejor exactitud validación: {history.history['val_accuracy'][mejor_idx]*100:.2f}%")

print("\n🎉 ¡Entrenamiento completado!")
print("\n📌 Para usar el sistema completo:")
print("   python sistema_completo_64.py")+''
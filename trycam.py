# train_64_final.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import time

# --- 1. CONFIGURACIÓN DE HARDWARE (CRÍTICO PARA QUE NO SE TRABE) ---
# Evita que TensorFlow acapare toda la memoria de la GPU si tienes una
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Configuración de memoria GPU optimizada.")
    except RuntimeError as e:
        print(e)

print("🎯 ENTRENANDO MODELO 64x64 (OPTIMIZADO PARA MEMORIA)")
print("=" * 50)

# Configuración
IMG_SIZE = (64, 64)
BATCH_SIZE = 16 # Mantenemos bajo para ahorrar memoria
EPOCHS = 25
DATA_DIR = 'dataset' # Asegúrate de que tus carpetas 'abiertos' y 'cerrados' estén aquí dentro

# --- 2. CARGAR DATOS COMO STREAM (NO CARGAR TODO A RAM) ---
# Usamos image_dataset_from_directory para leer del disco poco a poco
print("📂 Configurando pipeline de datos...")

try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        class_names=['abiertos', 'cerrados'] # 0=abiertos, 1=cerrados
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        class_names=['abiertos', 'cerrados']
    )
except ValueError as e:
    print(f"\n❌ ERROR: No se encontró la estructura de carpetas correcta en '{DATA_DIR}'.")
    print("Asegúrate de tener: dataset/abiertos/ y dataset/cerrados/")
    exit()

# --- 3. PREPROCESAMIENTO OPTIMIZADO EN TENSORFLOW ---
# Replicamos tu lógica de OpenCV (contraste y normalización) pero usando TF
# para que se ejecute en la GPU/CPU de forma eficiente.

def procesar_imagen(img, label):
    # Convertir a float
    img = tf.cast(img, tf.float32)
    
    # 1. Ajustar brillo (equivalente a tu beta -30)
    # Restamos 30 valores de pixel aprox
    img = tf.image.adjust_brightness(img, -30.0/255.0)
    
    # 2. Ajustar contraste (equivalente a tu alpha 1.2)
    img = tf.image.adjust_contrast(img, 1.2)
    
    # 3. Normalizar entre 0 y 1
    img = img / 255.0
    
    # Asegurar rango válido por si el contraste se pasó
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img, label

# Aplicar preprocesamiento y optimización de carga
AUTOTUNE = tf.data.AUTOTUNE

# map: aplica la función. prefetch: carga el siguiente lote mientras entrena el actual
train_ds = train_ds.map(procesar_imagen, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(procesar_imagen, num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)

print("\n✅ Pipeline de datos listo. Los datos se leerán bajo demanda.")

# --- 4. MODELO (IGUAL PERO INTEGRADO) ---
print("\n🧠 Creando modelo para 64x64...")

model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),
    
    # Aumento de datos
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
    ),
    # Callback para limpiar basura de memoria al final de cada época
    keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: tf.keras.backend.clear_session())
]

# --- 5. ENTRENAR ---
print("\n🚀 Iniciando entrenamiento...")
inicio = time.time()

# Nota: Ya no pasamos X_train, y_train. Pasamos el dataset 'train_ds'
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print(f"⏱️  Tiempo: {time.time() - inicio:.2f} segundos")

# --- 6. EVALUACIÓN Y GUARDADO ---
print("\n📊 Evaluando...")
resultados = model.evaluate(val_ds, verbose=0)

print(f"\n✅ RESULTADOS FINALES 64x64:")
print(f"   🔹 Exactitud: {resultados[1]*100:.2f}%")
print(f"   🔹 Precisión: {resultados[2]*100:.2f}%")
print(f"   🔹 Recall: {resultados[3]*100:.2f}%")

model.save('modelo_ojos_64.h5')
print("💾 Modelo guardado como 'modelo_ojos_64.h5'")

if history.history:
    mejor_idx = np.argmax(history.history['val_accuracy'])
    print(f"\n🏆 Mejor época: {mejor_idx + 1}")
    print(f"   Mejor exactitud validación: {history.history['val_accuracy'][mejor_idx]*100:.2f}%")

print("\n🎉 ¡Entrenamiento completado sin desbordar memoria!")
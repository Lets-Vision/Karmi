import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
import numpy as np
import cv2
import os

# --- CONFIGURACIÓN ---
# Reemplaza con la ruta real de tu modelo y una imagen de prueba
MODEL_PATH = 'modelo_ojos_64.h5' 
IMG_PATH = 'C:/Users/Mikel/model/dataset/cerrados/s0018_01076_0_0_0_0_1_01.png' # Usa una imagen clara
OUTPUT_FILENAME = 'visualizacion_cnn_isef.png'
IMG_SIZE = (64, 64)
OUTPUT_DIR = 'visualizacion_isef'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. Cargar Modelo e Imagen
print("📂 Cargando modelo...")
model = load_model(MODEL_PATH)

print("🖼️ Procesando imagen...")
img = cv2.imread(IMG_PATH)
img_res = cv2.resize(img, (64, 64))
# Aplicamos tu preprocesamiento: Contraste + Normalización
img_cont = cv2.addWeighted(img_res, 1.2, np.zeros(img_res.shape, img_res.dtype), 0, -30)
cv2.imwrite(f'{OUTPUT_DIR}/0_entrada_procesada.jpg', img_cont) # Guardar entrada

img_tensor = cv2.cvtColor(img_cont, cv2.COLOR_BGR2RGB) / 255.0
img_tensor = np.expand_dims(img_tensor, axis=0)

# 2. Extraer capas de activación (Conv2D)
capas_conv = [layer.name for layer in model.layers if 'conv2d' in layer.name]
model_visualizacion = Model(inputs=model.inputs, outputs=[model.get_layer(name).output for name in capas_conv])

activaciones = model_visualizacion.predict(img_tensor)

# 3. Generar imágenes de cada bloque
for i, mapa in enumerate(activaciones):
    # Promediamos los filtros para ver la "activación general" de ese bloque
    heatmap = np.mean(mapa[0], axis=-1)
    
    # Normalizar para que sea visible (0 a 255)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    
    # Aplicar color "Jet" (Azul es frío, Rojo es donde la IA más se fija)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Redimensionar para que no se vea tan pequeña en el póster (ej. 300x300)
    heatmap_final = cv2.resize(heatmap_color, (300, 300), interpolation=cv2.INTER_CUBIC)
    
    nombre_archivo = f'{OUTPUT_DIR}/bloque_{i+1}_activacion.jpg'
    cv2.imwrite(nombre_archivo, heatmap_final)
    print(f"✅ Guardado: {nombre_archivo}")

print(f"\n🚀 ¡Listo! Revisa la carpeta '{OUTPUT_DIR}' para tus imágenes del póster.")
# Simulación de la predicción
prediccion = model.predict(img_tensor)[0][0] # Devuelve un valor entre 0 y 1

# Crear una imagen de "Resultado"
resultado_img = np.zeros((300, 500, 3), dtype=np.uint8) + 255 # Fondo blanco

# Lógica de clasificación
estado = "CERRADO" if prediccion > 0.5 else "ABIERTO"
color = (0, 0, 255) if prediccion > 0.5 else (0, 200, 0) # Rojo o Verde

# Escribir en la imagen
cv2.putText(resultado_img, f"Score: {prediccion:.4f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
cv2.putText(resultado_img, f"Estado: {estado}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

cv2.imwrite(f'{OUTPUT_DIR}/4_resultado_final.jpg', resultado_img)
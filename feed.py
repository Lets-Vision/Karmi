import os
import shutil

# --- CONFIGURACIÓN ---
# Ruta donde descomprimiste el dataset MRL (puede tener subcarpetas dentro)
ruta_origen = "mrlEyes_2018_01" 

# Rutas de destino para el entrenamiento
ruta_destino_base = "dataset"
carpeta_abiertos = os.path.join(ruta_destino_base, "abiertos")
carpeta_cerrados = os.path.join(ruta_destino_base, "cerrados")

# Crear carpetas si no existen
os.makedirs(carpeta_abiertos, exist_ok=True)
os.makedirs(carpeta_cerrados, exist_ok=True)

print("Organizando imágenes... esto puede tardar un poco.")
contador_abiertos = 0
contador_cerrados = 0

# Recorrer todas las subcarpetas del dataset
for root, dirs, files in os.walk(ruta_origen):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):
            # El formato es: subject_id_gender_glasses_eyeState_...
            # Separamos el nombre por guiones bajos
            partes = file.split("_")
            
            # El estado del ojo suele ser el 5º elemento (índice 4)
            # 0 = cerrado, 1 = abierto
            try:
                estado_ojo = partes[4] 
                
                ruta_completa_origen = os.path.join(root, file)
                
                if estado_ojo == "1":
                    # Mover (o copiar) a abiertos
                    shutil.copy(ruta_completa_origen, os.path.join(carpeta_abiertos, file))
                    contador_abiertos += 1
                elif estado_ojo == "0":
                    # Mover (o copiar) a cerrados
                    shutil.copy(ruta_completa_origen, os.path.join(carpeta_cerrados, file))
                    contador_cerrados += 1
                    
            except IndexError:
                print(f"Archivo con nombre extraño ignorado: {file}")

print("¡Terminado!")
print(f"Imágenes de ojos ABIERTOS procesadas: {contador_abiertos}")
print(f"Imágenes de ojos CERRADOS procesadas: {contador_cerrados}")
print(f"Ahora puedes ejecutar el script 'entrenar.py'")
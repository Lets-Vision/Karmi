import cv2
import numpy as np
import tensorflow as tf
import time
import os

# --- CARGAR MODELO DE IA ---
try:
    model = tf.keras.models.load_model('karmi1.1M_64.h5')
    print("Modelo IA cargado correctamente.")
except:
    print("ADVERTENCIA: No se encontró 'modelo_ojos_64.h5'. Modo simulación.")
    model = None

# --- CARGAR CASCADAS (Detectores) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Intentamos cargar la nariz desde la carpeta local
archivo_nariz = 'haarcascade_mcs_nose.xml'
nose_cascade = cv2.CascadeClassifier(archivo_nariz)

# Verificación de seguridad para la nariz
usar_nariz = not nose_cascade.empty()
if not usar_nariz:
    print(f"ATENCIÓN: No se encontró '{archivo_nariz}'.")
    print("--> Descárgalo y ponlo en la misma carpeta para mayor precisión.")
    print("--> El sistema funcionará en modo 'Estimación' (Línea Amarilla) por ahora.\n")
else:
    print(f"Detector de nariz '{archivo_nariz}' cargado EXITOSAMENTE. (Modo Línea Azul activo)")

# --- CONFIGURACIÓN ---
nombre_ventana = 'Sistema de Rastreo Anatomico (Robust)'
cv2.namedWindow(nombre_ventana)

def nada(x): pass
cv2.createTrackbar('Umbral IA %', nombre_ventana, 60, 100, nada)
cv2.createTrackbar('Padding Ojo %', nombre_ventana, 20, 100, nada)

memoria_ojos = {"izq": None, "der": None}
cap = cv2.VideoCapture(0) 

ultimo_tiempo_medicion = time.time()
intervalo_medicion = 1.0 

estado_actual = {
    "izq": {"estado": "...", "pred": 0.0, "color": (100,100,100)},
    "der": {"estado": "...", "pred": 0.0, "color": (100,100,100)}
}

def procesar_para_ia(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    img_resized = cv2.resize(gray_clahe, (64, 64)) 
    img_norm = img_resized.astype('float32') / 255.0
    img_3ch = cv2.merge([img_norm, img_norm, img_norm])
    return img_resized, np.expand_dims(img_3ch, axis=0)

while True:
    tiempo_actual = time.time()
    toca_medir = False
    if (tiempo_actual - ultimo_tiempo_medicion) >= intervalo_medicion:
        toca_medir = True
        ultimo_tiempo_medicion = tiempo_actual

    umbral_sensibilidad = cv2.getTrackbarPos('Umbral IA %', nombre_ventana) / 100.0
    padding_perc = cv2.getTrackbarPos('Padding Ojo %', nombre_ventana) / 100.0

    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Añadimos minSize para descartar basurilla pequeña directamente
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(100, 100))

    # --- FILTRO: SOLO UN ROSTRO (EL MÁS GRANDE) ---
    if len(faces) > 0:
        # Ordenamos los rostros por área (ancho * alto) y nos quedamos con el primero
        rostro_principal = max(faces, key=lambda rect: rect[2] * rect[3])
        faces = [rostro_principal]

    ojos_visualizables = [] 

    for (x, y, w, h) in faces:
        # Asegurar coordenadas dentro del frame
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_frame - x), min(h, h_frame - y)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 50), 1)
        
        roi_face_gray = gray_frame[y:y+h, x:x+w]
        if roi_face_gray.size == 0: continue

        # --- DETECCIÓN DE NARIZ ---
        nariz_detectada = False
        ny_real_en_cara = 0
        nh_local = 0
        
        y_limite_inferior = int(y + (h * 0.55)) # Fallback
        y_limite_superior = int(y + (h * 0.20)) # Fallback

        if usar_nariz:
            # Buscar en el tercio inferior central
            y_nose_start = int(h * 0.3)
            roi_face_bottom = roi_face_gray[y_nose_start:h, 0:w]
            
            if roi_face_bottom.size > 0:
                noses = nose_cascade.detectMultiScale(roi_face_bottom, 1.1, 5)
                if len(noses) > 0:
                    nariz_detectada = True
                    # Nariz más grande
                    nx_loc, ny_loc, nw_loc, nh_loc = sorted(noses, key=lambda n: n[2]*n[3], reverse=True)[0]
                    
                    ny_real_en_cara = ny_loc + y_nose_start
                    nh_local = nh_loc
                    
                    # Dibujar nariz
                    cv2.rectangle(frame, (x+nx_loc, y+ny_real_en_cara), 
                                  (x+nx_loc+nw_loc, y+ny_real_en_cara+nh_loc), (0, 255, 0), 1)

        # --- CÁLCULO DE LÍMITES ---
        if nariz_detectada:
            # ANATOMÍA: Ojos encima de la nariz
            y_limite_inferior = y + ny_real_en_cara + int(nh_local * 0.2) 
            distancia = int(nh_local * 1.5)
            y_limite_superior = max(y, y_limite_inferior - distancia)
            cv2.line(frame, (x, y_limite_inferior), (x+w, y_limite_inferior), (255, 0, 0), 2) # Azul = Preciso
        else:
            # FALLBACK: Estimación
            cv2.line(frame, (x, y_limite_inferior), (x+w, y_limite_inferior), (0, 255, 255), 1) # Amarillo = Estimado

        # Validar y Extraer
        if y_limite_inferior > y_limite_superior:
            roi_franja = gray_frame[y_limite_superior:y_limite_inferior, x:x+w]
            
            if roi_franja.size > 0:
                # OPTIMIZACIÓN: Verificamos si ambos ojos ya están memorizados y cerrados
                ambos_cerrados = (estado_actual["izq"]["estado"] == "CERRADO" and 
                                  estado_actual["der"]["estado"] == "CERRADO" and
                                  memoria_ojos["izq"] is not None and 
                                  memoria_ojos["der"] is not None)
                
                # Si ambos están cerrados, no corremos el cascade en absoluto (ahorra CPU)
                if not ambos_cerrados:
                    eyes = eye_cascade.detectMultiScale(roi_franja, 1.1, 8, minSize=(20, 20))
                    
                    if len(eyes) >= 2:
                        eyes = sorted(eyes, key=lambda e: e[0])
                        
                        # Guardamos la posición SOLO si el ojo está ABIERTO o si es la primera vez
                        if memoria_ojos["izq"] is None or estado_actual["izq"]["estado"] != "CERRADO":
                            memoria_ojos["izq"] = eyes[0]
                            
                        if memoria_ojos["der"] is None or estado_actual["der"]["estado"] != "CERRADO":
                            memoria_ojos["der"] = eyes[1]
                
                posiciones = []
                if memoria_ojos["izq"] is not None and memoria_ojos["der"] is not None:
                    posiciones = [("izq", memoria_ojos["izq"]), ("der", memoria_ojos["der"])]

                for (lado, (ex, ey, ew, eh)) in posiciones:
                    # Coord absolutas
                    abs_x = x + ex
                    abs_y = y_limite_superior + ey
                    
                    # Padding
                    pw, ph = int(ew*padding_perc), int(eh*padding_perc)
                    nx = max(0, abs_x - pw)
                    ny = max(0, abs_y - ph)
                    nw_f = min(w_frame, abs_x + ew + pw) - nx
                    nh_f = min(h_frame, abs_y + eh + ph) - ny
                    
                    roi_ojo = frame[ny:ny+nh_f, nx:nx+nw_f]
                    
                    if roi_ojo.size > 0:
                        img_vis, tensor = procesar_para_ia(roi_ojo)
                        
                        if toca_medir and model:
                            pred = model.predict(tensor, verbose=0)[0][0]
                            est = "CERRADO" if pred > umbral_sensibilidad else "ABIERTO"
                            col = (0,0,255) if est == "CERRADO" else (0,255,0)
                            estado_actual[lado].update({"pred": pred, "estado": est, "color": col})
                            print(f"[{time.strftime('%H:%M:%S')}] {lado.upper()}: {est} ({pred:.2f})")

                        info = estado_actual[lado]
                        cv2.rectangle(frame, (nx, ny), (nx+nw_f, ny+nh_f), info["color"], 2)
                        
                        # Display
                        disp = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
                        disp = cv2.resize(disp, (100, 100), interpolation=cv2.INTER_NEAREST)
                        cv2.putText(disp, info["estado"], (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, info["color"], 1)
                        ojos_visualizables.append(disp)

    # UI Displays
    if len(ojos_visualizables) > 0: frame[10:110, 10:110] = ojos_visualizables[0]
    if len(ojos_visualizables) > 1: frame[10:110, w_frame-110:w_frame-10] = ojos_visualizables[1]

    # UI Barra
    seg_rest = intervalo_medicion - (time.time() - ultimo_tiempo_medicion)
    if seg_rest < 0: seg_rest = 0
    bar_w = int(w_frame * (1 - seg_rest/intervalo_medicion))
    col_bar = (255, 255, 255) if toca_medir else (200, 100, 0)
    cv2.rectangle(frame, (0, h_frame-5), (bar_w, h_frame), col_bar, -1)

    cv2.imshow(nombre_ventana, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
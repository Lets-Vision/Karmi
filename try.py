import cv2
import numpy as np
import tensorflow as tf
import time
import os

# --- CARGAR MODELO DE IA ---
model = None
for modelo_path in ['karmi1.1M_64.h5', 'modelo_ojos_64.h5', 'karmi64.h5']:
    try:
        model = tf.keras.models.load_model(modelo_path)
        print(f"✅ Modelo '{modelo_path}' cargado correctamente.")
        break
    except:
        continue

if model is None:
    print("❌ ERROR: No se encontró ningún modelo .h5 compatible.")

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
memoria_relativa = {"izq": None, "der": None} # dx, dy respecto al centro de la CARA
face_smooth = None # Para suavizado de cara
cap = cv2.VideoCapture(0) 

def ema_smoothing(viejo, nuevo, alpha=0.3):
    if viejo is None: return nuevo
    return tuple(int(alpha * n + (1-alpha) * v) for v, n in zip(viejo, nuevo))

ultimo_tiempo_medicion = time.time()
intervalo_medicion = 1.0 

estado_actual = {
    "izq": {"estado": "...", "pred": 0.0, "color": (100,100,100)},
    "der": {"estado": "...", "pred": 0.0, "color": (100,100,100)}
}

def procesar_para_ia(roi):
    """Sincronizado al 100% con el entrenamiento (RGB + Contraste)"""
    # 1. Redimensión a 64x64
    img = cv2.resize(roi, (64, 64))
    # 2. BGR a RGB (Fundamental)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. Mejora de contraste (idéntica al entrenamiento: 1.2, -30)
    img_cont = cv2.addWeighted(img_rgb, 1.2, np.zeros(img_rgb.shape, img_rgb.dtype), 0, -30)
    # 4. Normalización [0, 1]
    img_norm = img_cont.astype('float32') / 255.0
    # 5. Formato Tensor
    tensor = np.expand_dims(img_norm, axis=0)
    return img_cont, tensor

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
    
    # --- FILTRO: SOLO UN ROSTRO (el más grande) ---
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(100, 100))
    if len(faces) > 0:
        face_principal = max(faces, key=lambda rect: rect[2] * rect[3])
        # Suavizado de cara
        face_smooth = ema_smoothing(face_smooth, face_principal, 0.4)
        faces = [face_smooth]
    else:
        # Resetear memorias si no hay cara
        memoria_ojos = {"izq": None, "der": None}
        memoria_relativa = {"izq": None, "der": None}
        face_smooth = None

    ojos_visualizables = [] 

    for (x, y, w, h) in faces:
        # Asegurar coordenadas dentro del frame
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_frame - x), min(h, h_frame - y)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 50), 1)
        
        roi_face_gray = gray_frame[y:y+h, x:x+w]
        if roi_face_gray.size == 0: continue

        # --- DETECCIÓN DE NARIZ ---
        # Solo intentamos detectar si cargó el archivo XML correctamente
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

        # --- CÁLCULO DE LÍMITES (Fijos por la CARA para mayor estabilidad) ---
        y_limite_superior = int(y + (h * 0.18))
        y_limite_inferior = int(y + (h * 0.52))
        
        # Dibujar línea verde de referencia (basada en la cara)
        cv2.line(frame, (x, y_limite_inferior), (x+w, y_limite_inferior), (0, 255, 0), 2)

        # Validar y Extraer
        if y_limite_inferior > y_limite_superior:
            roi_franja = gray_frame[y_limite_superior:y_limite_inferior, x:x+w]
            if roi_franja.size > 0:
                eyes = eye_cascade.detectMultiScale(roi_franja, 1.1, 6, minSize=(18, 18))
                
                # --- FILTRO: EXACTAMENTE DOS OJOS (mejor par) ---
                if len(eyes) >= 2:
                    eyes_s = sorted(eyes, key=lambda e: e[0])
                    mejor_par = None
                    max_sep = 0
                    for i in range(len(eyes_s)):
                        for j in range(i+1, len(eyes_s)):
                            cx_i = eyes_s[i][0] + eyes_s[i][2]//2
                            cx_j = eyes_s[j][0] + eyes_s[j][2]//2
                            sep = cx_j - cx_i
                            # Anatomía: ojos separados entre 15% y 75% del ancho de la cara
                            if w * 0.15 < sep < w * 0.75:
                                if sep > max_sep:
                                    max_sep = sep
                                    mejor_par = (eyes_s[i], eyes_s[j])
                    
                    if mejor_par:
                        # Si detectamos ojos, guardamos su posición relativa al ROSTRO
                        memoria_ojos["izq"] = ema_smoothing(memoria_ojos["izq"], mejor_par[0])
                        memoria_ojos["der"] = ema_smoothing(memoria_ojos["der"], mejor_par[1])
                        
                        # Guardar anclaje respecto al centro de la cara (regido por la cara/línea verde)
                        cx_f, cy_f = x + w//2, y + h//2
                        for lado in ["izq", "der"]:
                            ex, ey, ew, eh = memoria_ojos[lado]
                            abs_x, abs_y = x + ex, y_limite_superior + ey
                            memoria_relativa[lado] = (abs_x - cx_f, abs_y - cy_f, ew, eh)
                
                # --- FALLBACK: FIJAR POR EL ROSTRO SI SE PIERDEN ---
                elif memoria_relativa["izq"] is not None:
                    # Usar el centro de la cara actual como ancla (altura por la cara)
                    cx_f, cy_f = x + w//2, y + h//2
                    for lado in ["izq", "der"]:
                        dx, dy, ew, eh = memoria_relativa[lado]
                        # Re-construir posición del ojo basada en la cara actual
                        abs_x, abs_y = cx_f + dx, cy_f + dy
                        # Actualizar memoria_ojos (relativo a la franja actual)
                        memoria_ojos[lado] = (abs_x - x, abs_y - y_limite_superior, ew, eh)
                
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
                        # Convertimos de vuelta a BGR solo para que cv2.imshow lo muestre bien
                        disp = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
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
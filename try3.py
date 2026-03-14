import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from collections import deque
import socket
import os
import json
import asyncio
import sys

# DEBUG: Imprimir info del entorno
print("--- DEBUG INFO ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Path: {sys.path}")
print("------------------")

try:
    import websockets
except ImportError:
    print("❌ ERROR: websockets no encontrado. Intentando instalar...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets
    print("✅ websockets instalado dinámicamente.")

# ====================================================================
#  CONFIGURACIÓN GLOBAL Y CARGA DE MODELO
# ====================================================================
tf.get_logger().setLevel('ERROR')

model_name = 'karmi1.1M_64.h5'
try:
    model = tf.keras.models.load_model(model_name)
    # Warm-up (evita lag inicial)
    dummy = np.zeros((1, 64, 64, 3), dtype='float32')
    model.predict(dummy, verbose=0)
    print(f"✅ Modelo '{model_name}' cargado y listo.")
except Exception as e:
    print(f"❌ ADVERTENCIA: No se pudo cargar '{model_name}':", e)
    model = None

# Detectores
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
usar_nariz = not nose_cascade.empty()

# Comunicación (UDP y WebSocket)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
DEST = ('127.0.0.1', 5005)

# --- WebSocket Server ---
CLIENTS = set()
WS_LOOP = None

async def ws_handler(websocket):
    CLIENTS.add(websocket)
    try:
        async for _ in websocket: pass
    finally:
        CLIENTS.remove(websocket)

async def start_ws():
    async with websockets.serve(ws_handler, "localhost", 8765):
        await asyncio.Future()  # run forever

def run_ws_server():
    global WS_LOOP
    WS_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(WS_LOOP)
    WS_LOOP.run_until_complete(start_ws())

threading.Thread(target=run_ws_server, daemon=True).start()

# --- Gesture States ---
GESTURE_TIMES = {"Ambos": None, "Derecho": None, "Izquierdo": None}
GESTURE_FLAGS = {"Ambos": False, "Derecho": False, "Izquierdo": False}
LAST_REPEAT = {"Ambos": 0, "Derecho": 0, "Izquierdo": 0}
GESTURE_GRACE = {"Ambos": 0, "Derecho": 0, "Izquierdo": 0}

# --- Thresholds (Sincronizados con el Frontend) ---
MIN_WINK_TIME = 0.3   # 300ms para confirmar intención
REPEAT_DELAY = 1.0    # 1s entre movimientos si se mantiene pulsado
LONG_BLINK_TIME = 1.0 # 1s para Escape/Cerrar
GRACE_TIME = 0.15      # 150ms de tolerancia a parpadeos del ojo opuesto

def detect_gestures(open_eyes):
    global GESTURE_TIMES, GESTURE_FLAGS, LAST_REPEAT, GESTURE_GRACE
    now = time.time()
    events = []
    
    der_open = "Derecho" in open_eyes
    izq_open = "Izquierdo" in open_eyes
    
    def check_gesture(key, is_active):
        if is_active:
            GESTURE_GRACE[key] = now 
            if GESTURE_TIMES[key] is None:
                GESTURE_TIMES[key] = now
                LAST_REPEAT[key] = now
                return False 
            
            dur = now - GESTURE_TIMES[key]
            
            # Primer disparo
            if dur >= MIN_WINK_TIME and not GESTURE_FLAGS[key]:
                GESTURE_FLAGS[key] = True
                LAST_REPEAT[key] = now
                return True
            
            # Auto-repetición
            if GESTURE_FLAGS[key] and (now - LAST_REPEAT[key]) >= REPEAT_DELAY:
                LAST_REPEAT[key] = now
                return True
        else:
            if GESTURE_TIMES[key] is not None:
                if (now - GESTURE_GRACE[key]) > GRACE_TIME:
                    GESTURE_TIMES[key] = None
                    GESTURE_FLAGS[key] = False
        return False

    # Ambos cerrados (Parpadeo / Enter)
    if check_gesture("Ambos", not der_open and not izq_open):
        is_long = (now - GESTURE_TIMES["Ambos"]) >= LONG_BLINK_TIME
        events.append({"action": "parpadeo", "long_press": is_long})
        print(f">>> Gesto: Ambos {'LARGO' if is_long else ''}")

    # Guiño Derecho
    if not (not der_open and not izq_open) and check_gesture("Derecho", izq_open and not der_open):
        events.append({"action": "guinho", "type": "derecho", "long_press": False})
        print(">>> Gesto: Guiño Derecho")

    # Guiño Izquierdo
    if not (not der_open and not izq_open) and check_gesture("Izquierdo", der_open and not izq_open):
        events.append({"action": "guinho", "type": "izquierdo", "long_press": False})
        print(">>> Gesto: Guiño Izquierdo")
        
    return events

def broadcast_gesture(open_eyes, events):
    if not CLIENTS or WS_LOOP is None: return
    # Payload de estado (siempre se envía)
    state_payload = json.dumps({"type": "eyes", "eyes": open_eyes})
    
    for ws in list(CLIENTS):
        try:
            asyncio.run_coroutine_threadsafe(ws.send(state_payload), WS_LOOP)
            for ev in events:
                asyncio.run_coroutine_threadsafe(ws.send(json.dumps(ev)), WS_LOOP)
        except: pass

# Ventana y controles
nombre_ventana = 'Tracking Ocular Avanzado (Ultra-Fluido)'
cv2.namedWindow(nombre_ventana)

def nada(x): pass
cv2.createTrackbar('Umbral IA %', nombre_ventana, 60, 100, nada)
cv2.createTrackbar('Expansion Ojo %', nombre_ventana, 50, 100, nada)

cap = cv2.VideoCapture(0) # Cambiar a 1 si usas cámara externa
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ====================================================================
#  PREPROCESADO DE IA (Sincronizado con entrenamiento)
# ====================================================================
def procesar_para_ia(roi):
    """Mismo preproceso que train_64_final.py"""
    img = cv2.resize(roi, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Mejora de contraste 1.2, brillo -30
    img = cv2.addWeighted(img, 1.2, np.zeros(img.shape, img.dtype), 0, -30)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0), img

# ====================================================================
#  PREDICTOR ASÍNCRONO (Hilos para evitar caídas de FPS)
# ====================================================================
class PredictorAsync:
    def __init__(self, modelo):
        self.model = modelo
        self.lock = threading.Lock()
        self.resultados = {0: (0.0, "ND", (128,128,128)), 1: (0.0, "ND", (128,128,128))}
        self.vis_imgs = {0: None, 1: None}
        self.cola = deque(maxlen=10) # Mayor cola para evitar colisiones entre ojos
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def enviar(self, idx, tensor, img_vis, umbral):
        self.cola.append((idx, tensor, img_vis, umbral))

    def obtener(self, idx):
        with self.lock:
            return self.resultados[idx], self.vis_imgs[idx]

    def _worker(self):
        while self.running:
            if self.cola:
                idx, tensor, img_vis, umbral = self.cola.popleft()
                try:
                    if self.model:
                        pred = float(self.model.predict(tensor, verbose=0)[0][0])
                        estado = "CERRADO" if pred > umbral else "ABIERTO"
                        color = (0, 0, 255) if estado == "CERRADO" else (0, 255, 0)
                        with self.lock:
                            self.resultados[idx] = (pred, estado, color)
                            self.vis_imgs[idx] = img_vis
                except Exception as e:
                    with self.lock:
                        self.resultados[idx] = (0.0, "ERROR", (0, 0, 255))
                    continue
            else:
                time.sleep(0.005)

predictor = PredictorAsync(model)

# ====================================================================
#  TRACKER INTELIGENTE CON CENTRADO EN PUPILA
# ====================================================================
class SmartTracker:
    def __init__(self):
        self.face = None
        self.ojos = [None, None]
        self.calibrado = False
        self.muestras = 0
        self.ALPHA = 0.25 # Aumentado para más fluidez
        self.frames_sin_cara = 0
        self.n_frames = 0
        self.franja = None

    def _ema(self, viejo, nuevo):
        if viejo is None: return nuevo
        return tuple(int(self.ALPHA * n + (1 - self.ALPHA) * v) for v, n in zip(viejo, nuevo))

    def _refinar_pupila(self, gray_roi):
        """Busca el punto más oscuro para centrar el ojo."""
        if gray_roi.size == 0: return None
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        _, _, min_loc, _ = cv2.minMaxLoc(blurred)
        return min_loc

    def actualizar(self, frame, gray):
        self.n_frames += 1
        h_f, w_f = gray.shape[:2]

        # 1. Rostro (cada 2 frames)
        if self.n_frames % 2 == 0 or self.face is None:
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))
            if len(faces) > 0:
                self.face = self._ema(self.face, max(faces, key=lambda r: r[2]*r[3]))
                self.frames_sin_cara = 0
            else:
                self.frames_sin_cara += 1

        if self.face is None or self.frames_sin_cara > 30:
            self.face = None
            self.calibrado = False
            self.muestras = 0
            return False

        fx, fy, fw, fh = self.face
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (50, 50, 50), 1)

        # 2. Nariz y franja de ojos (cada 5 frames)
        if self.n_frames % 5 == 0 or self.franja is None:
            roi_cara = gray[fy:fy+fh, fx:fx+fw]
            y_ns = int(fh * 0.35)
            roi_nariz = roi_cara[y_ns:fh, 0:fw]
            if roi_nariz.size > 0:
                noses = nose_cascade.detectMultiScale(roi_nariz, 1.1, 5, minSize=(25, 25))
                if len(noses) > 0:
                    _, ny, _, _ = max(noses, key=lambda n: n[2]*n[3])
                    self.franja = (fy + int(fh*0.12), fy + y_ns + ny)

        if self.franja is None: self.franja = (int(fy+fh*0.15), int(fy+fh*0.48))
        y_sup, y_inf = self.franja
        cv2.line(frame, (fx, y_inf), (fx+fw, y_inf), (255, 0, 0), 1)

        # 3. Ojos y refinamiento de pupila
        roi_franja = gray[y_sup:y_inf, fx:fx+fw]
        if roi_franja.size > 0:
            eyes = eye_cascade.detectMultiScale(roi_franja, 1.1, 4, minSize=(18, 18))
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[0])
                if (eyes[1][0] - eyes[0][0]) > fw * 0.15:
                    for i in range(2):
                        ex, ey, ew, eh = eyes[i]
                        roi_ojo = roi_franja[ey:ey+eh, ex:ex+ew]
                        pupila = self._refinar_pupila(roi_ojo)
                        
                        abs_x = fx + ex
                        abs_y = y_sup + ey
                        if pupila:
                            # Centrar rect en pupila
                            abs_x += pupila[0] - (ew // 2)
                            abs_y += pupila[1] - (eh // 2)
                        
                        self.ojos[i] = self._ema(self.ojos[i], (abs_x, abs_y, ew, eh))
                    
                    if not self.calibrado:
                        self.muestras += 1
                        if self.muestras > 10: self.calibrado = True
            else:
                # Feedback visual si solo detecta 1 ojo
                if not self.calibrado and len(eyes) == 1:
                    cv2.putText(frame, "Alinea ambos ojos...", (fx, y_sup-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,150,255), 1)

        return self.ojos[0] is not None and self.ojos[1] is not None

tracker = SmartTracker()

# ====================================================================
#  BUCLE PRINCIPAL
# ====================================================================
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    umbral = cv2.getTrackbarPos('Umbral IA %', nombre_ventana) / 100.0
    expansion = cv2.getTrackbarPos('Expansion Ojo %', nombre_ventana) / 100.0

    open_eyes = []
    if tracker.actualizar(frame, gray):
        for i, rect in enumerate(tracker.ojos):
            ex, ey, ew, eh = rect
            pw, ph = int(ew * expansion), int(eh * expansion)
            x1, y1 = max(0, ex-pw), max(0, ey-ph)
            x2, y2 = min(frame.shape[1], ex+ew+pw), min(frame.shape[0], ey+eh+ph)

            roi_ojo = frame[y1:y2, x1:x2].copy()
            if roi_ojo.size > 0:
                tensor, vis_ia = procesar_para_ia(roi_ojo)
                predictor.enviar(i, tensor, vis_ia, umbral)
                
                (pred, est, col), vis_pred = predictor.obtener(i)
                if est == "ABIERTO":
                    open_eyes.append("Izquierdo" if i == 0 else "Derecho")
                
                # Dibujar
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{['IZQ','DER'][i]}: {est}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                
                if vis_pred is not None:
                    # Conv para mostrar preproceso real
                    m = (vis_pred * 255).astype('uint8')
                    m = cv2.cvtColor(m, cv2.COLOR_RGB2BGR)
                    m = cv2.resize(m, (100, 100))
                    cv2.putText(m, f"{pred:.2f}", (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                    if i == 0: frame[10:110, 10:110] = m
                    else: frame[10:110, frame.shape[1]-110:frame.shape[1]-10] = m

    # --- Procesar Gestos y Transmitir ---
    # Se hace fuera del bloque 'actualizar' para enviar lista vacía si no hay cara
    events = detect_gestures(open_eyes)
    broadcast_gesture(open_eyes, events)

    # Barra Seguimiento/Calibración
    txt = "SISTEMA ACTIVO" if tracker.calibrado else f"CALIBRANDO ({tracker.muestras}/12)..."
    cv2.putText(frame, txt, (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0,255,0) if tracker.calibrado else (0,200,255), 1)

    cv2.imshow(nombre_ventana, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
predictor.running = False
import cv2
import numpy as np
import time
import threading
from collections import deque
import socket
import json
import asyncio
import sys
import os

# CONFIGURACION DE EJECUCION
HEADLESS = "--headless" in sys.argv
if HEADLESS:
    print(">>> MODO BACKGROUND ACTIVO (Sin Ventanas)")

# PATCH KERA 3 PARA CARGAR MODELOS KERAS 2 (.H5)
try:
    # import os # Already imported above
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import keras
    from keras import layers
    
    # Parche para error de 'batch_shape' y 'optional' al cargar .h5 antiguos en Keras 3
    _orig_input = layers.InputLayer
    class PatchedInputLayer(_orig_input):
        def __init__(self, *args, **kwargs):
            kwargs.pop('batch_shape', None)
            kwargs.pop('optional', None)
            super().__init__(*args, **kwargs)
    layers.InputLayer = PatchedInputLayer
    
    import tensorflow as tf
    TF_OK = True
    TF_PKG = keras
except Exception as e:
    TF_OK = False
    TF_ERR = str(e)

import mediapipe as mp

# DEBUG: Imprimir info del entorno
print("--- DEBUG INFO ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("------------------")

try:
    import websockets
except ImportError:
    print("ERROR: websockets no encontrado. Intentando instalar...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets
    print("websockets instalado dinámicamente.")

# Siempre habrá alguien en el mundo q piense en ti, en este mundo soy yo kar <3
# Intentar liberar el puerto 8765 si está bloqueado (común en Windows si no se cerró bien)
def kill_previous():
    if os.name == 'nt':
        import subprocess
        try:
            # Buscar el proceso usando netstat y matarlo
            output = subprocess.check_output('netstat -ano | findstr :8765', shell=True).decode()
            for line in output.strip().split('\n'):
                parts = [p for p in line.split() if p]
                if parts:
                    pid = parts[-1]
                    if pid != str(os.getpid()):
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                        print(f"✅ Puerto 8765 liberado (PID {pid} terminado).")
        except: pass

kill_previous()

# ====================================================================
#  UI INITIALIZATION (Show Loading Screen First)
# ====================================================================
nombre_ventana = 'GazeTracker'
if not HEADLESS:
    cv2.namedWindow(nombre_ventana)

def show_loading_msg(msg):
    if HEADLESS: 
        print(f"[INIT] {msg}")
        return
    temp = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(temp, msg, (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(nombre_ventana, temp)
    cv2.waitKey(1)

show_loading_msg("Iniciando y buscando modelos...")

# ====================================================================
#  MODELO REGISTRY — Detección dinámica de modelos (SOLO .H5)
# ====================================================================
MODELS = []
script_dir = os.path.dirname(os.path.realpath(__file__))
root_files = os.listdir(script_dir)
h5_files = [f for f in root_files if f.endswith('.h5')]

# --- Cargar modelos TensorFlow (.h5) ---
if h5_files:
    show_loading_msg(f"Cargando {len(h5_files)} modelos TensorFlow...")
    if not TF_OK:
        print(f"ERROR: Keras o TensorFlow no cargo correctamente: {TF_ERR}")
    
    for tf_name in h5_files:
        try:
            if not TF_OK: continue
            
            show_loading_msg(f"Cargando {tf_name}...")
            # Forzamos compilación False para evitar errores con optimizadores guardados
            m = TF_PKG.models.load_model(os.path.join(script_dir, tf_name), compile=False)
            
            # Forzar siempre 64x64 según requerimiento
            input_size = (64, 64)
            # Warm-up (algunas veces el primer predict es lento)
            dummy = np.zeros((1, input_size[0], input_size[1], 3), dtype='float32')
            m.predict(dummy, verbose=0)
            
            MODELS.append({
                'name': tf_name.replace('.h5', ''),
                'type': 'tf',
                'model': m,
                'input_size': input_size,
                'input_chw': False,
            })
            print(f"OK: TF modelo '{tf_name}' cargado.")
        except Exception as e:
            print(f"ERROR: No se pudo cargar TF '{tf_name}': {e}")
else:
    print("AVISO: No se encontraron archivos .h5 en el directorio.")

# --- Modelos ONNX eliminados por petición del usuario ---

if not MODELS:
    print("FATAL: No se encontró ningún modelo en la raíz. Saliendo.")
    sys.exit(1)

# Ordenar por nombre para consistencia
MODELS.sort(key=lambda x: x['name'])

print(f"\nModelos detectados ({len(MODELS)}):")
for i, m in enumerate(MODELS):
    print(f"  [{i}] {m['name']} ({m['type'].upper()})")

# Variable global para el modelo activo
current_model_idx = 0
def on_model_change(val):
    global current_model_idx
    current_model_idx = val
    m = MODELS[val]
    print(f"\n🔄 Switched to: [{val}] {m['name']} ({m['type'].upper()})")
    try: 
        if not HEADLESS:
            cv2.setTrackbarPos('Modelo', 'GazeTracker', val)
    except: pass
    # Notificar a los clientes del cambio
    if WS_LOOP and CLIENTS:
        sync_msg = json.dumps({"type": "model_change", "index": val})
        for ws in list(CLIENTS):
            try: asyncio.run_coroutine_threadsafe(ws.send(sync_msg), WS_LOOP)
            except: pass

def nada(x): pass
if not HEADLESS:
    cv2.createTrackbar('Umbral IA %', nombre_ventana, 60, 100, nada)
    cv2.createTrackbar('Expansion Ojo %', nombre_ventana, 50, 100, nada)
    cv2.createTrackbar('Modelo', nombre_ventana, 0, max(0, len(MODELS) - 1), on_model_change)

# --- MediaPipe Tasks Initialization ---
show_loading_msg("Iniciando MediaPipe Face Detection...")
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = os.path.join(script_dir, 'face_landmarker.task')
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
    mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
)

# Landmarks para los ojos
LEFT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# --- Comunicación y Gestos ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
DEST = ('127.0.0.1', 5005)

CLIENTS = set()
WS_LOOP = None
def run_ws_server():
    global WS_LOOP
    WS_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(WS_LOOP)
    
    async def h(ws):
        CLIENTS.add(ws)
        # Enviar lista de modelos al conectar
        try:
            model_info_list = [{"name": m['name'], "type": m['type']} for m in MODELS]
            initial_state = json.dumps({
                "type": "model_sync",
                "models": model_info_list,
                "current": current_model_idx
            })
            await ws.send(initial_state)
        except: pass
        
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    if data.get("action") == "switch_model":
                        new_idx = int(data.get("index", 0))
                        if 0 <= new_idx < len(MODELS):
                            try: 
                                if not HEADLESS:
                                    cv2.setTrackbarPos('Modelo', 'GazeTracker', new_idx)
                            except: pass
                except: pass
        finally:
            if ws in CLIENTS: CLIENTS.remove(ws)

    async def start():
        async with websockets.serve(h, "localhost", 8765):
            await asyncio.Future()
    WS_LOOP.run_until_complete(start())

threading.Thread(target=run_ws_server, daemon=True).start()

# --- Gesture State Tracking ---
g_state = {'p_counter': 0, 'l_counter': 0, 'r_counter': 0}

def detect_gestures(l_open, r_open, umbral_ia):
    events = []
    # Usamos el score normalizado (1.0=Abierto, 0.0=Cerrado)
    l_closed = l_open <= umbral_ia
    r_closed = r_open <= umbral_ia
    
    # --- Parpadeo (Ambos ojos) ---
    if l_closed and r_closed:
        g_state['l_counter'] = 0 # Reset wink counters if both closed
        g_state['r_counter'] = 0
        g_state['p_counter'] += 1
    elif g_state['p_counter'] > 1: # Mínimo 2 frames para parpadeo
        dur = g_state['p_counter']
        tipo = "LARGO" if dur > 25 else "CORTO"
        print(f">>> Gesto: Ambos {dur} {tipo}")
        events.append({"action": "parpadeo", "long_press": dur > 25, "duration": dur})
        g_state['p_counter'] = 0
    else:
        g_state['p_counter'] = 0

    # --- Guiño Izquierdo ---
    if l_closed and not r_closed:
        g_state['l_counter'] += 1
    elif g_state['l_counter'] >= 5: # Mínimo 5 frames para confirmar
        dur = g_state['l_counter']
        tipo = "LARGO" if dur > 25 else "CORTO"
        print(f">>> Gesto: IZQ {dur} {tipo}")
        events.append({"action": "guinho", "type": "izquierdo", "long_press": dur > 25, "duration": dur})
        g_state['l_counter'] = 0
    else:
        g_state['l_counter'] = 0

    # --- Guiño Derecho ---
    if r_closed and not l_closed:
        g_state['r_counter'] += 1
    elif g_state['r_counter'] >= 5:
        dur = g_state['r_counter']
        tipo = "LARGO" if dur > 25 else "CORTO"
        print(f">>> Gesto: DER {dur} {tipo}")
        events.append({"action": "guinho", "type": "derecho", "long_press": dur > 25, "duration": dur})
        g_state['r_counter'] = 0
    else:
        g_state['r_counter'] = 0
        
    return events

def broadcast(open_eyes, events, l_conf=0.0, r_conf=0.0):
    if not CLIENTS or WS_LOOP is None: return
    p = json.dumps({
        "type": "eyes", 
        "eyes": open_eyes,
        "l_conf": float(l_conf),
        "r_conf": float(r_conf)
    })
    for ws in list(CLIENTS):
        try:
            asyncio.run_coroutine_threadsafe(ws.send(p), WS_LOOP)
            for ev in events: asyncio.run_coroutine_threadsafe(ws.send(json.dumps(ev)), WS_LOOP)
        except: pass

# Landmarks de las esquinas de los ojos (estables para anclaje)
R_CORNERS = [33, 133]  # Ojo derecho
L_CORNERS = [362, 263] # Ojo izquierdo

# --- Preproceso y Predictor ---
def p_roi(roi, m_info):
    w, h = m_info['input_size']
    
    # Arreglar pérdida de resolución: si el ojo real es más grande que la entrada (ej: por estar cerca de cámara), usar interpolación de área densa
    interp = cv2.INTER_AREA if roi.shape[1] > w else cv2.INTER_LINEAR
    img = cv2.resize(roi, (w, h), interpolation=interp)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    
    # Mejorar contraste sincronizado con el entrenamiento (1.2 y -30/255)
    img = cv2.addWeighted(img, 1.2, np.zeros(img.shape, img.dtype), 0, -0.12)
    img = np.clip(img, 0, 1)
    
    if m_info['input_chw']: 
        img = np.transpose(img, (2, 0, 1))
    
    vis = cv2.resize(roi, (64, 64))
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    return np.expand_dims(img, axis=0), vis

class Predictor:
    def __init__(self):
        self.lock = threading.Lock()
        self.res = {0: (0.0, "ND", (128,128,128)), 1: (0.0, "ND", (128,128,128))}
        self.vis = {0: None, 1: None}
        self.pending, self.running = None, True
        threading.Thread(target=self._w, daemon=True).start()
    def send(self, t, v, u, m):
        with self.lock: self.pending = (t, v, u, m)
    def get(self, i):
        with self.lock: return self.res[i], self.vis[i]
    def _w(self):
        while self.running:
            with self.lock: j, self.pending = self.pending, None
            if j:
                t, v, u, m = j
                try:
                    b = np.concatenate(t, axis=0)
                    if m['type'] == 'onnx':
                        p = m['model'].run(None, {m['input_name']: b})[0]
                        with self.lock:
                            for i in range(2):
                                # Lógica ONNX: raw 1.0 = ABIERTO (según feedback "al revés")
                                raw_v = float(p[i])
                                open_val = raw_v
                                est = "ABIERTO" if open_val > u else "CERRADO"
                                color = (0,255,0) if est=="ABIERTO" else (0,0,255)
                                self.res[i], self.vis[i] = (open_val, est, color), v[i]
                                if i == 0: print(f"DEBUG {m['name']} | RAW: {raw_v:.3f} | OPEN: {open_val:.3f} | {est}   ", end='\r')
                    else:
                        p = m['model'].predict(b, verbose=0)
                        with self.lock:
                            for i in range(2):
                                # Lógica H5: raw 1.0 = CERRADO
                                raw_v = float(p[i][0])
                                open_val = 1.0 - raw_v
                                est = "CERRADO" if raw_v > u else "ABIERTO"
                                color = (0,0,255) if est=="CERRADO" else (0,255,0)
                                self.res[i], self.vis[i] = (open_val, est, color), v[i]
                                if i == 0: print(f"DEBUG {m['name']} | RAW: {raw_v:.3f} | OPEN: {open_val:.3f} | {est}   ", end='\r')
                except: pass
            else: time.sleep(0.005)

predictor = Predictor()

# --- Tracker ---
class Tracker:
    def __init__(self): self.ojos, self.muestras, self.calibrado = [None, None], 0, False
    def up(self, fr, fb, active_m_info=None):
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=fr), int(time.time()*1000))
        h, w = fb.shape[:2]
        if not res.face_landmarks: return False
        fl = res.face_landmarks[0]
        def get(idx, corner_idx):
            c_pts = np.array([(int(fl[i].x*w), int(fl[i].y*h)) for i in corner_idx])
            cx = int(np.mean(c_pts[:, 0]))
            cy = int(np.mean(c_pts[:, 1])) - 2
            
            dist = np.linalg.norm(c_pts[0] - c_pts[1])
            base_factor = 1.4
            if dist > 35:
                # Evitar que haga demasiado zoom-out si estamos muy cerca y pierda detalle del recuadro
                factor = min(base_factor + (dist - 35) * 0.015, 2.0)
            else:
                factor = base_factor
                
            ew = int(dist * factor) 
            
            # Respetar la relación de aspecto del modelo activo ONNX o H5 (ej: 40x24)
            if active_m_info:
                mw, mh = active_m_info['input_size']
                eh = int(ew * (mh / mw))
            else:
                eh = ew
            
            return (max(0, cx - ew//2), max(0, cy - eh//2), ew, eh)

        self.ojos = [get(LEFT_EYE_LANDMARKS, L_CORNERS), get(RIGHT_EYE_LANDMARKS, R_CORNERS)]
        if not self.calibrado:
            self.muestras += 1
            if self.muestras > 15: self.calibrado = True
        return True

tk = Tracker()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    fb = cv2.flip(frame, 1) # Flip visual
    l_conf, r_conf = 0.0, 0.0
    try:
        if not HEADLESS:
            umbral_ia = cv2.getTrackbarPos('Umbral IA %', 'GazeTracker') / 100.0
            exp = cv2.getTrackbarPos('Expansion Ojo %', 'GazeTracker') / 100.0
            m_idx = cv2.getTrackbarPos('Modelo', 'GazeTracker')
        else:
            umbral_ia, exp, m_idx = 0.6, 0.5, current_model_idx # Default values for headless
    except:
        umbral_ia, exp, m_idx = 0.6, 0.5, current_model_idx
    
    if m_idx < 0 or m_idx >= len(MODELS): m_idx = 0
    m_info = MODELS[m_idx]
    open_eyes = []
    
    if tk.up(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), fb, active_m_info=m_info):
        ts, vs, rs = [], [], []
        for i, r in enumerate(tk.ojos):
            ex, ey, ew, eh = r
            pw, ph = int(ew*exp), int(eh*exp)
            x1, y1, x2, y2 = max(0,ex-pw), max(0,ey-ph), min(frame.shape[1],ex+ew+pw), min(frame.shape[0],ey+eh+ph)
            rs.append((x1,y1,x2,y2))
            roi = frame[y1:y2, x1:x2].copy()
            if roi.size > 0:
                t, v = p_roi(roi, m_info)
                ts.append(t); vs.append(v)
            else:
                wi, hi = m_info['input_size']
                ts.append(np.zeros((1,3,hi,wi) if m_info['input_chw'] else (1,hi,wi,3), dtype='float32'))
                vs.append(None)
        if len(ts) == 2: predictor.send(ts, vs, umbral_ia, m_info)
        for i in range(2):
            if not rs[i]: continue
            (val, est, col), vis = predictor.get(i)
            if i == 0: l_conf = val
            else: r_conf = val
            if est == "ABIERTO": open_eyes.append(["Izquierdo", "Derecho"][i])
            x1, y1, x2, y2 = rs[i]
            if not HEADLESS:
                cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                cv2.putText(frame, f"{['IZQ','DER'][i]}: {est}", (x1, y1-5), 0, 0.45, col, 1)
                if vis is not None:
                    m = cv2.resize((vis*255).astype('uint8'), (100,100))
                    m = cv2.cvtColor(m, cv2.COLOR_RGB2BGR)
                    cv2.putText(m, f"{val:.2f}", (5,90), 0, 0.4, (255,255,255), 1)
                    if i == 0: frame[10:110, 10:110] = m
                    else: frame[10:110, frame.shape[1]-110:frame.shape[1]-10] = m
    broadcast(open_eyes, detect_gestures(l_conf, r_conf, umbral_ia), l_conf, r_conf)
    txt = f"ACTIVE [{m_info['name']}]" if tk.calibrado else f"CALIBRATING ({tk.muestras}/15)..."
    if not HEADLESS:
        cv2.putText(frame, txt, (10, frame.shape[0]-15), 0, 0.5, (0,255,0) if tk.calibrado else (0,200,255), 1)
        cv2.imshow(nombre_ventana, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    else:
        # En modo headless dormimos un poco para no saturar el CPU (aprox 30fps)
        time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
landmarker.close()
predictor.running = False

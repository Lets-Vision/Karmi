# sistema_completo_64.py
import cv2
import numpy as np
import os
import time
from tensorflow import keras

print("🎯 SISTEMA COMPLETO DETECCIÓN DE OJOS - 64x64")
print("=" * 50)

class DetectorOjos64:
    def __init__(self, modelo_path='modelo_ojos_simple.h5'):
        """
        Sistema que detecta caras y clasifica ojos usando modelo 64x64
        """
        # Cargar el modelo entrenado
        try:
            self.modelo = keras.models.load_model(modelo_path)
            print(f"✅ Modelo cargado: {modelo_path}")
        except:
            print(f"❌ No se pudo cargar el modelo: {modelo_path}")
            print("   Primero entrena el modelo con train_sin_sklearn.py")
            self.modelo = None
        
        # Tamaño del modelo (64x64)
        self.IMG_SIZE = (64, 64)
        
        # Cargar clasificador de caras de OpenCV (HAAR Cascade)
        # Viene incluido con OpenCV, no necesita descargar nada
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if os.path.exists(cascade_path):
            self.detector_caras = cv2.CascadeClassifier(cascade_path)
            print("✅ Detector de caras HAAR cargado")
        else:
            print("❌ No se encontró el detector HAAR")
            self.detector_caras = None
    
    def detectar_caras(self, imagen):
        """
        Detecta caras usando HAAR Cascade
        """
        if self.detector_caras is None:
            return []
        
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras
        caras = self.detector_caras.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return caras
    
    def detectar_ojos_en_cara(self, imagen, x, y, w, h):
        """
        Detecta ojos dentro de una región de cara
        """
        # Extraer región de la cara
        region_cara = imagen[y:y+h, x:x+w]
        
        if region_cara.size == 0:
            return []
        
        # Convertir a escala de grises para detección de ojos
        gris_cara = cv2.cvtColor(region_cara, cv2.COLOR_BGR2GRAY)
        
        # Cargar detector de ojos HAAR
        ojos_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        if ojos_cascade.empty():
            # Si no encuentra el detector de ojos, usar zonas fijas
            return self._estimar_posicion_ojos(x, y, w, h)
        
        # Detectar ojos
        ojos = ojos_cascade.detectMultiScale(
            gris_cara,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # Convertir coordenadas relativas a absolutas
        ojos_absolutos = []
        for (ex, ey, ew, eh) in ojos:
            ojos_absolutos.append({
                'x': x + ex,
                'y': y + ey,
                'w': ew,
                'h': eh,
                'imagen': region_cara[ey:ey+eh, ex:ex+ew]
            })
        
        return ojos_absolutos
    
    def _estimar_posicion_ojos(self, x, y, w, h):
        """
        Estima posición de ojos si no se detectan automáticamente
        """
        ojos = []
        
        # Ojo izquierdo (aproximadamente 25% desde el borde izquierdo)
        ojo_izq = {
            'x': x + int(w * 0.25),
            'y': y + int(h * 0.25),
            'w': int(w * 0.2),
            'h': int(h * 0.15),
            'nombre': 'izquierdo'
        }
        
        # Ojo derecho (aproximadamente 25% desde el borde derecho)
        ojo_der = {
            'x': x + int(w * 0.55),
            'y': y + int(h * 0.25),
            'w': int(w * 0.2),
            'h': int(h * 0.15),
            'nombre': 'derecho'
        }
        
        return [ojo_izq, ojo_der]
    
    def preprocesar_ojo(self, imagen_ojo):
        """
        Preprocesa imagen de ojo para el modelo 64x64
        """
        # Si la imagen está vacía o es muy pequeña
        if imagen_ojo.size == 0:
            # Crear imagen negra como fallback
            imagen_ojo = np.zeros((self.IMG_SIZE[0], self.IMG_SIZE[1], 3), dtype=np.uint8)
        
        # Redimensionar a 64x64
        img_resized = cv2.resize(imagen_ojo, self.IMG_SIZE)
        
        # Convertir a RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalizar
        img_normalized = img_rgb / 255.0
        
        return img_normalized
    
    def clasificar_ojo(self, imagen_ojo):
        """
        Clasifica si un ojo está abierto o cerrado
        """
        if self.modelo is None:
            return "ERROR", 0
        
        # Preprocesar
        img_procesada = self.preprocesar_ojo(imagen_ojo)
        img_input = np.expand_dims(img_procesada, axis=0)
        
        # Predecir
        try:
            prediccion = self.modelo.predict(img_input, verbose=0)[0][0]
            
            # Interpretar resultado
            if prediccion < 0.5:
                return "ABIERTO", (1 - prediccion) * 100
            else:
                return "CERRADO", prediccion * 100
        except:
            return "ERROR", 0
    
    def procesar_imagen(self, ruta_imagen):
        """
        Procesa una imagen completa
        """
        print(f"\n📂 Procesando: {ruta_imagen}")
        
        # Cargar imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print("❌ No se pudo cargar la imagen")
            return
        
        # Hacer copia para dibujar
        imagen_resultado = imagen.copy()
        
        # Detectar caras
        caras = self.detectar_caras(imagen)
        
        if len(caras) == 0:
            print("⚠️ No se detectaron caras")
            return
        
        print(f"👤 Caras detectadas: {len(caras)}")
        
        resultados_totales = []
        
        for i, (x, y, w, h) in enumerate(caras):
            print(f"\n  Cara #{i+1}:")
            
            # Dibujar rectángulo de la cara
            cv2.rectangle(imagen_resultado, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(imagen_resultado, f"Cara {i+1}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Detectar ojos en esta cara
            ojos = self.detectar_ojos_en_cara(imagen, x, y, w, h)
            
            if not ojos:
                print("    ⚠️ No se detectaron ojos específicos")
                continue
            
            print(f"    👁️ Ojos detectados: {len(ojos)}")
            
            for j, ojo in enumerate(ojos):
                # Extraer región del ojo
                if 'imagen' in ojo:
                    region_ojo = ojo['imagen']
                else:
                    # Si no hay imagen, extraer de las coordenadas estimadas
                    x1 = ojo['x']
                    y1 = ojo['y']
                    w1 = ojo['w']
                    h1 = ojo['h']
                    region_ojo = imagen[y1:y1+h1, x1:x1+w1]
                
                # Clasificar ojo
                estado, confianza = self.clasificar_ojo(region_ojo)
                
                # Nombre del ojo
                nombre_ojo = ojo.get('nombre', f"ojo_{j+1}")
                
                # Guardar resultado
                resultados_totales.append({
                    'cara': i+1,
                    'ojo': nombre_ojo,
                    'estado': estado,
                    'confianza': confianza,
                    'coords': (ojo['x'], ojo['y'], ojo['w'], ojo['h'])
                })
                
                # Dibujar resultado
                x1, y1, w1, h1 = ojo['x'], ojo['y'], ojo['w'], ojo['h']
                
                # Color según estado
                if estado == "ABIERTO":
                    color = (0, 255, 0)  # Verde
                elif estado == "CERRADO":
                    color = (0, 0, 255)  # Rojo
                else:
                    color = (255, 255, 0)  # Amarillo
                
                # Dibujar rectángulo del ojo
                cv2.rectangle(imagen_resultado, (x1, y1), (x1+w1, y1+h1), color, 2)
                
                # Texto con resultado
                texto = f"{nombre_ojo}: {estado} ({confianza:.0f}%)"
                cv2.putText(imagen_resultado, texto, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                print(f"      {nombre_ojo.upper()}: {estado} ({confianza:.1f}%)")
        
        # Mostrar estadísticas
        if resultados_totales:
            abiertos = sum(1 for r in resultados_totales if r['estado'] == "ABIERTO")
            cerrados = sum(1 for r in resultados_totales if r['estado'] == "CERRADO")
            
            print(f"\n📊 RESUMEN:")
            print(f"   Ojos abiertos: {abiertos}")
            print(f"   Ojos cerrados: {cerrados}")
            
            if cerrados > abiertos:
                print("   ⚠️ ALERTA: Mayoría de ojos cerrados")
        
        # Mostrar imagen
        cv2.imshow('Detección de Ojos 64x64', imagen_resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Guardar resultado
        cv2.imwrite('resultado_deteccion.jpg', imagen_resultado)
        print("💾 Imagen guardada como 'resultado_deteccion.jpg'")
        
        return resultados_totales

def modo_camara():
    """
    Modo de cámara en tiempo real
    """
    print("\n📷 INICIANDO MODO CÁMARA")
    print("=" * 50)
    print("Presiona 'q' para salir")
    print("Presiona 's' para guardar foto")
    
    # Crear detector
    detector = DetectorOjos64()
    
    if detector.modelo is None or detector.detector_caras is None:
        print("❌ No se pudo inicializar el detector")
        return
    
    # Iniciar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return
    
    # Configurar tamaño de cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n🎥 Cámara iniciada. Mirando a la cámara...")
    
    contador_fotos = 0
    
    while True:
        # Capturar frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Voltear horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Detectar caras
        caras = detector.detectar_caras(frame)
        
        # Procesar cada cara
        for (x, y, w, h) in caras:
            # Dibujar cara
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Detectar y clasificar ojos
            ojos = detector.detectar_ojos_en_cara(frame, x, y, w, h)
            
            ojos_cerrados = 0
            total_ojos = 0
            
            for ojo in ojos:
                if 'imagen' in ojo:
                    region_ojo = ojo['imagen']
                else:
                    x1, y1, w1, h1 = ojo['x'], ojo['y'], ojo['w'], ojo['h']
                    region_ojo = frame[y1:y1+h1, x1:x1+w1]
                
                # Clasificar
                estado, confianza = detector.clasificar_ojo(region_ojo)
                
                # Dibujar resultado
                x1, y1, w1, h1 = ojo['x'], ojo['y'], ojo['w'], ojo['h']
                color = (0, 255, 0) if estado == "ABIERTO" else (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), color, 2)
                
                # Texto pequeño para no saturar
                texto = f"{estado[:3]} {confianza:.0f}%"
                cv2.putText(frame, texto, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                total_ojos += 1
                if estado == "CERRADO":
                    ojos_cerrados += 1
            
            # Mostrar alerta si ojos cerrados
            if total_ojos > 0:
                if ojos_cerrados / total_ojos > 0.5:  # Más del 50% cerrados
                    cv2.putText(frame, "¡ALERTA! OJOS CERRADOS", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar FPS
        cv2.putText(frame, "Presiona 'q' para salir", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostrar frame
        cv2.imshow('Deteccion de Ojos en Tiempo Real', frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Guardar foto
            contador_fotos += 1
            nombre = f"foto_{contador_fotos}.jpg"
            cv2.imwrite(nombre, frame)
            print(f"📸 Foto guardada: {nombre}")
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Modo cámara finalizado")

def main():
    """
    Función principal
    """
    print("\n" + "=" * 50)
    print("SISTEMA DE DETECCIÓN DE OJOS - MENÚ")
    print("=" * 50)
    
    # Verificar si existe el modelo
    if not os.path.exists('modelo_ojos_simple.h5'):
        print("❌ Modelo no encontrado: 'modelo_ojos_simple.h5'")
        print("\n📝 Primero debes entrenar el modelo:")
        print("   1. Asegúrate de tener la carpeta 'dataset/' con:")
        print("      - dataset/abiertos/ (imágenes de ojos abiertos)")
        print("      - dataset/cerrados/ (imágenes de ojos cerrados)")
        print("   2. Ejecuta: python train_sin_sklearn.py")
        print("   3. Luego vuelve a ejecutar este script")
        return
    
    # Crear detector
    detector = DetectorOjos64()
    
    if detector.modelo is None or detector.detector_caras is None:
        return
    
    while True:
        print("\n🔍 ¿Qué quieres hacer?")
        print("   1. Probar con una imagen")
        print("   2. Usar cámara web en tiempo real")
        print("   3. Salir")
        
        opcion = input("\n📋 Selecciona una opción (1-3): ").strip()
        
        if opcion == '1':
            # Pedir ruta de imagen
            ruta = input("📂 Ruta de la imagen (o presiona Enter para usar ejemplo.jpg): ").strip()
            
            if not ruta:
                ruta = 'ejemplo.jpg'
            
            if os.path.exists(ruta):
                detector.procesar_imagen(ruta)
            else:
                print(f"❌ No se encuentra: {ruta}")
                print("   Crea una imagen llamada 'ejemplo.jpg' o especifica otra ruta")
        
        elif opcion == '2':
            modo_camara()
        
        elif opcion == '3':
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción no válida")

if __name__ == "__main__":
    main()
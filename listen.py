import socket

# Configuración del host y puerto donde try3.py enviará los mensajes
LOCAL_HOST = '0.0.0.0'  # Escucha en todas las interfaces
PORT = 5005             # Debe coincidir con el DEST_PORT de try3.py

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((LOCAL_HOST, PORT))

print(f"Escuchando eventos de ojos en {LOCAL_HOST}:{PORT} (Ctrl+C para salir)")

try:
    while True:
        data, addr = sock.recvfrom(1024)
        try:
            mensaje = data.decode('utf-8')
        except Exception:
            mensaje = str(data)
        print(f"Recibido de {addr[0]}:{addr[1]} --> {mensaje}")
except KeyboardInterrupt:
    print("\nEscucha detenida por el usuario.")
finally:
    sock.close()

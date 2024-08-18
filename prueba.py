import cv2
import numpy as np

# Inicializamos la captura de video desde la cámara del dispositivo
video_capture = cv2.VideoCapture(0)  # El valor 0 indica la cámara predeterminada del dispositivo

# Definimos el umbral de diferencia para la detección de movimiento
umbral_diferencia = 30

# Definimos el tamaño del kernel para la operación de dilatación
tamano_kernel = 5

while True:
    # Capturamos un frame de la cámara
    ret, frame = video_capture.read()

    if not ret:
        break

    # Convertimos el frame a escala de grises
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicamos un suavizado para reducir el ruido
    frame_gris = cv2.GaussianBlur(frame_gris, (21, 21), 0)

    # Si es la primera iteración, almacenamos el fondo
    if 'fondo' not in locals():
        fondo = frame_gris
        continue

    # Calculamos la diferencia entre el fondo y el frame actual
    diferencia = cv2.absdiff(fondo, frame_gris)

    # Aplicamos un umbral para obtener una imagen binaria
    _, umbral = cv2.threshold(diferencia, umbral_diferencia, 255, cv2.THRESH_BINARY)

    # Aplicamos una operación de dilatación para llenar agujeros
    umbral = cv2.dilate(umbral, None, iterations=tamano_kernel)

    # Encontramos contornos en la imagen umbral
    contornos, _ = cv2.findContours(umbral.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujamos los contornos en el frame original
    for contorno in contornos:
        if cv2.contourArea(contorno) > 500:  # Filtramos contornos pequeños
            (x, y, w, h) = cv2.boundingRect(contorno)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostramos el frame con los contornos dibujados
    cv2.imshow('Video con Detección de Movimiento', frame)

    # Si presionamos la tecla 'q', salimos del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos los recursos y cerramos las ventanas
video_capture.release()
cv2.destroyAllWindows()


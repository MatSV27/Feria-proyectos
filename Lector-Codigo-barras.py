import cv2
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
Codigo_estudiante = ""

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    detected_barcodes = decode(frame)
    if not detected_barcodes:
        print("No se detectó ningún código de barras")
    else:
        for barcode in detected_barcodes:
            if barcode.data != "":
                print(barcode.data)
                Codigo_estudiante = barcode.data.decode('utf-8')  # Almacena los datos del código de barras en la variable
                # Dibuja un rectángulo alrededor del código de barras
                (x, y, w, h) = barcode.rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break  # Rompe el bucle for una vez que se detecta un código de barras

        if Codigo_estudiante != "":
            break  # Rompe el bucle while si se ha almacenado algún dato en Codigo_estudiante

    cv2.imshow('scanner', frame)
    if cv2.waitKey(1) == ord('q'):
        break
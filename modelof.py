import cv2
import dlib
import os
import face_recognition
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
import time
import mediapipe as mp

# Rutas de las carpetas
categories = {
    "Alumno Matriculado": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Alumno Matriculado",
    "Alumno no Matriculado": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Alumno no Matriculado",
    "Profesor": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Profesor",
    "Trabajador": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Trabajador"
}

# Inicializar mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
ruta_destino = r'C:\Users\USER\Desktop\Modelo\FOTOMANO'

# Diccionario para almacenar las codificaciones faciales y los nombres de las imágenes de cada categoría
category_encodings = {}
category_names = {}

# Cargar las codificaciones faciales y los nombres de las imágenes de cada categoría
for category, folder_path in categories.items():
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    category_encodings[category] = []

    for img in image_files:
        face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(os.path.join(folder_path, img)))
        
        # Verificar si hay al menos una codificación facial
        if face_encodings:
            category_encodings[category].append(face_encodings[0])

    category_names[category] = [os.path.splitext(img)[0] for img in image_files]

# Crear la ventana de Tkinter
window = tk.Tk()

# Obtener el tamaño de la pantalla
ancho_pantalla = window.winfo_screenwidth()
alto_pantalla = window.winfo_screenheight()

# Cargar la imagen de fondo y escalarla al tamaño de la pantalla
ruta_imagen = r"C:\Users\USER\Desktop\Modelo\Interfaz\Interfaz.jpg"
imagen_fondo = Image.open(ruta_imagen)
imagen_fondo = imagen_fondo.resize((ancho_pantalla, alto_pantalla), Image.ANTIALIAS if hasattr(Image, 'ANTIALIAS') else Image.LANCZOS)
imagen_fondo = ImageTk.PhotoImage(imagen_fondo)

# Establecer las dimensiones de la ventana al tamaño de la pantalla
window.geometry(f"{ancho_pantalla}x{alto_pantalla}")

# Configurar el tamaño del cuadro de video y colocarlo en el centro
ancho_cuadro_video = 320  # Ajusta según el ancho deseado
alto_cuadro_video = 360  # Ajusta según el alto deseado
video_label = tk.Label(window, width=ancho_cuadro_video, height=alto_cuadro_video)
video_label.place(relx=1, rely=1, anchor=tk.CENTER)

# Crear el objeto VideoCapture
cap = cv2.VideoCapture(0)

# Crear un widget Label para la imagen de fondo
fondo_label = tk.Label(window, image=imagen_fondo)
fondo_label.place(x=0, y=0, relwidth=1, relheight=1)

# Crear el widget de la etiqueta para mostrar el video
label = tk.Label(window)
label.place(relx=0.515, rely=0.43, anchor=tk.CENTER)

dni_var = tk.StringVar()


# Configurar la entrada para el DNI
dni_entry = tk.Entry(window, textvariable=dni_var, font=("Arial", 14))
dni_entry.place(relx=0.92, rely=0.6450, anchor=tk.E)

def procesar_video():
    ret, frame = cap.read()

    # Encontrar todas las ubicaciones de rostros en el cuadro actual
    face_locations = face_recognition.face_locations(frame, model="hog")  # Puedes ajustar el modelo aquí

    # Verificar si se detectaron rostros
    if face_locations:
        for face_location in face_locations:
            top, right, bottom, left = face_location

            # Codificar el rostro actual
            face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location], model="cnn")[0]  # Puedes ajustar el modelo aquí

            text = "Externo"
            color = (0, 0, 255)  # Rojo

            for category, encodings in category_encodings.items():
                results = face_recognition.compare_faces(encodings, face_frame_encodings, tolerance=0.5)
                if True in results:
                    text = category + " - " + category_names[category][results.index(True)]
                    if category == "Alumno Matriculado":
                        color = (0, 255, 0)  # Verde 
                    elif category == "Alumno no Matriculado":
                        color = (0, 255, 255)  # AMARILLO claro
                    elif category == "Profesor":
                        color = (255, 0, 0)  # Azul
                    elif category == "Trabajador":
                        color = (255, 255, 0)  # Celeste
                    break 

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Pasar el marco a RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el marco
    result = hands.process(rgb)

    # Dibujar las manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Verificar si la mano está cerrada
    if result.multi_hand_landmarks:
        # Obtener las coordenadas de los puntos de interés
        landmarks = result.multi_hand_landmarks[0].landmark
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].x
        index_finger_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x

        # Si el pulgar está a la izquierda del nudillo del dedo índice, la mano está cerrada
        if thumb_tip < index_finger_pip and len(dni_var.get()) == 8:
            ruta_destino = r'C:\Users\USER\Desktop\Modelo\FOTOMANO'
            # Obtener la fecha y hora actual
            ahora = datetime.now()
            formato_fecha_hora = ahora.strftime("%y-%m-%d_%I-%M-%p")
            # Construir el nombre del archivo con la fecha y hora
            nombre_archivo = f"{dni_var.get()}_{formato_fecha_hora}.jpg"

            # Guardar la imagen en la ruta de destino
            cv2.imwrite(os.path.join(ruta_destino, nombre_archivo), frame)
  
            # Mostrar un mensaje indicando que la foto ha sido tomada
            print(f"Foto tomada y guardada como {nombre_archivo}")

    # Convertir el cuadro a formato PIL y luego a formato ImageTk
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image)

    # Mostrar el cuadro en la etiqueta
    label.config(image=photo)
    label.image = photo

    window.update_idletasks()

    # Llamar a esta función nuevamente después de 15 milisegundos
    window.after(15, procesar_video)

# Iniciar el bucle principal de Tkinter
window.after(15, procesar_video)
window.mainloop()

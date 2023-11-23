import cv2
import dlib
import os
import face_recognition
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime

# Rutas de las carpetas
categories = {
    "Alumno Matriculado": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Alumno Matriculado",
    "Alumno no Matriculado": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Alumno no Matriculado",
    "Profesor": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Profesor",
    "Trabajador": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Trabajador"
}
predictor_path = r"C:\Users\USER\Desktop\Modelo\shape_predictor_68_face_landmarks (1).dat"
predictor = dlib.shape_predictor(predictor_path)

# Inicializar contadores para contar el número de parpadeos
contador_parpadeos = 0
umbral_parpadeo = 5  # Número de parpadeos requeridos para capturar una foto

# Diccionario para almacenar las codificaciones faciales y los nombres de las imágenes de cada categoría
category_encodings = {}
category_names = {}

# Cargar las codificaciones faciales y los nombres de las imágenes de cada categoría
for category, folder_path in categories.items():
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    category_encodings[category] = [face_recognition.face_encodings(face_recognition.load_image_file(os.path.join(folder_path, img)))[0] for img in image_files]
    category_names[category] = [os.path.splitext(img)[0] for img in image_files]

# Inicializar el detector de caras de dlib (utilizando el modelo preentrenado)
detector = dlib.get_frontal_face_detector()

# Directorio para almacenar las capturas de fotos
capturas_directorio = "capturas"
os.makedirs(capturas_directorio, exist_ok=True)

# Función para calcular la razón de cierre de ojos
def calcular_razon_cierre_ojos(puntos_faciales):
    # Convierte los puntos faciales a un array NumPy
    puntos_faciales_np = np.array(puntos_faciales)

    # Calcular la distancia euclidiana entre los puntos verticales de los ojos
    d_vert_izquierdo = np.linalg.norm(puntos_faciales_np[1] - puntos_faciales_np[5])
    d_vert_derecho = np.linalg.norm(puntos_faciales_np[2] - puntos_faciales_np[4])

    # Calcular la distancia euclidiana entre los puntos horizontales de los ojos
    d_horiz = np.linalg.norm(puntos_faciales_np[0] - puntos_faciales_np[3])

    # Calcular la razón de cierre de ojos
    razon_cierre_ojos = (d_vert_izquierdo + d_vert_derecho) / (2.0 * d_horiz)

    return razon_cierre_ojos

# Función para capturar una foto
def capturar_foto(frame, persona):
    global capturas_directorio

    # Obtener la fecha y hora actual
    ahora = datetime.now()

    # Formatear la fecha y hora en el formato deseado
    fecha_hora_str = ahora.strftime("%Y%m%d-%H%M%S")

    # Generar un nombre único para la captura
    nombre_captura = f"{persona}-{fecha_hora_str}.jpg"

    # Guardar la captura en el directorio de capturas
    ruta_captura = os.path.join(capturas_directorio, nombre_captura)
    cv2.imwrite(ruta_captura, frame)

    print(f"¡Foto capturada! ({nombre_captura})")

# Función para obtener la categoría por el código
def obtener_categoria_por_codigo(codigo):
    for category, codes in category_names.items():
        if codigo in codes:
            return category
    return "Desconocido"

def verificar_persona(codigo):
    if len(codigo) == 8:
        for category in ["Profesor", "Trabajador"]:
            if codigo in category_names[category]:
                return category
    return "Externo"
# Crear la ventana de Tkinter
window = tk.Tk()

# Obtener el tamaño de la pantalla
ancho_pantalla = window.winfo_screenwidth()
alto_pantalla = window.winfo_screenheight()

# Cargar la imagen de fondo y escalarla al tamaño de la pantalla
ruta_imagen = r"C:\Users\USER\Desktop\Modelo\Interfaz\FONDO.jpg"
imagen_fondo = Image.open(ruta_imagen)
imagen_fondo = imagen_fondo.resize((ancho_pantalla, alto_pantalla), Image.ANTIALIAS if hasattr(Image, 'ANTIALIAS') else Image.LANCZOS)
imagen_fondo = ImageTk.PhotoImage(imagen_fondo)

# Establecer las dimensiones de la ventana al tamaño de la pantalla
window.geometry(f"{ancho_pantalla}x{alto_pantalla}")

# Configurar el tamaño del cuadro de video y colocarlo en el centro
ancho_cuadro_video = 460  # Ajusta según el ancho deseado
alto_cuadro_video = 480  # Ajusta según el alto deseado
video_label = tk.Label(window, width=ancho_cuadro_video, height=alto_cuadro_video)
video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Crear el objeto VideoCapture
cap = cv2.VideoCapture(0)

# Crear un widget Label para la imagen de fondo
fondo_label = tk.Label(window, image=imagen_fondo)
fondo_label.place(x=0, y=0, relwidth=1, relheight=1)

# Crear el widget de la etiqueta para mostrar el video
label = tk.Label(window)
label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Crear variables para almacenar el código y el DNI
codigo_var = tk.StringVar()
dni_var = tk.StringVar()

# Configurar la entrada para el código
codigo_entry = tk.Entry(window, textvariable=codigo_var, font=("Arial", 14))
codigo_entry.place(relx=0.1, rely=0.9, anchor=tk.W)

# Configurar la entrada para el DNI
dni_entry = tk.Entry(window, textvariable=dni_var, font=("Arial", 14))
dni_entry.place(relx=0.9, rely=0.9, anchor=tk.E)

def actualizar_codigo_var(*args):
    codigo = codigo_var.get()
    if len(codigo) == 9:
        print(f"Código actualizado: {codigo}")
        verificar_alumno()
    elif len(codigo) == 8:
        print(f"Código actualizado: {codigo}")
        verificar_persona(codigo)

codigo_var.trace_add("write", actualizar_codigo_var)

# Función para verificar si es alumno
def verificar_alumno():
    codigo = codigo_var.get()
    categoria = verificar_persona(codigo)
    if categoria == "Alumno":
        print("¡Usted es alumno! No debe colocar su DNI.")

# Función para el bucle de video
def video_loop():
    global contador_parpadeos

    # Leer el cuadro actual del objeto VideoCapture
    ret, frame = cap.read()

    # Obtener los códigos
    codigo = codigo_var.get()
    dni = dni_var.get()

    # Verificar si es alumno
    if len(codigo) == 9:
        categoria = obtener_categoria_por_codigo(codigo)
        if categoria != "Desconocido":
            print(f"¡Bienvenido, {categoria}!")
        else:
            print("Código no encontrado en la base de datos. ¡No es un alumno!")

    # Verificar si es profesor o trabajador
    elif len(dni) == 8:
        categoria = verificar_persona(dni)
        if categoria:
            # Realizar el reconocimiento facial y otras acciones según la categoría
            dlib_faces = detector(frame, 0)
            landmarks = predictor(frame, dlib_faces[0])
            landmarks = [(p.x, p.y) for p in landmarks.parts()]
            razon_cierre_ojos = calcular_razon_cierre_ojos(landmarks[36:48])

            if categoria == "Externo":
               # Agregar aquí la lógica adicional para el reconocimiento facial de profesor o trabajador
               if razon_cierre_ojos < 0.2:
                 contador_parpadeos += 1
               else:
                contador_parpadeos = 0

                # Verificar si se alcanzó el umbral de parpadeo
               if contador_parpadeos == umbral_parpadeo:
                 capturar_foto(frame, f"{categoria}_{dni}")
                 print(f"¡Foto capturada! ({categoria}_{dni})")

            # Resto del código para dibujar el cuadro delimitador y el texto
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            text = category + " - " + category_names[category][results.index(True)]
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        else:
            # Mostrar alerta de que no está en la base de datos y es externo
            print("Código no válido para Profesor o Trabajador. Es un externo.")

            # Capturar foto si el externo parpadea 5 veces
            if contador_parpadeos == umbral_parpadeo:
                capturar_foto(frame, f"DNI-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
                print("¡Foto capturada para externo!")

            contador_parpadeos = 0

    # Verificar si es DNI y no reconocido
    elif len(dni) == 8:
        print("Usted es alumno, no debe colocar su DNI.")
    face_locations = face_recognition.face_locations(frame, model="hog")
    # Resto del código para dibujar el cuadro delimitador y el texto
    for face_location in face_locations:
        # Resto del código para dibujar el cuadro delimitador y el texto
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Codificar el rostro actual
        face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location], model="small")[0]

        # Inicializar la categoría como "Externo"
        text = "Externo"

        # Comparar las codificaciones faciales
        for category, encodings in category_encodings.items():
            results = face_recognition.compare_faces(encodings, face_frame_encodings, tolerance=0.5)
            if True in results:
                # Si se encuentra una coincidencia, actualizar la categoría
                text = category + " - " + category_names[category][results.index(True)]
                break

        # Dibujar el texto en el cuadro
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Convertir el cuadro a formato PIL y luego a formato ImageTk
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image)

    # Mostrar el cuadro en la etiqueta
    label.config(image=photo)
    label.image = photo

    # Llamar a esta función nuevamente después de 15 milisegundos
    window.after(15, video_loop)

# Iniciar el bucle de video
video_loop()

# Iniciar el bucle principal de Tkinter
window.mainloop()
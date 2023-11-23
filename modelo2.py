import cv2
import dlib
import os
import face_recognition
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
import Augmentor

# Constantes y configuraciones
UMBRAL_PARPADEO = 5
NUEVO_ANCHO = 400
NUEVO_ALTO = 400

detector = dlib.get_frontal_face_detector()
predictor_path = r"C:\Users\USER\Desktop\Modelo\shape_predictor_68_face_landmarks (1).dat"
predictor = dlib.shape_predictor(predictor_path)

# Rutas de las carpetas
CATEGORIES = {
    "Alumno Matriculado": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Alumno Matriculado",
    "Alumno no Matriculado": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Alumno no Matriculado",
    "Profesor": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Profesor",
    "Trabajador": "C:\\Users\\USER\\Desktop\\Modelo\\Images\\Trabajador"
}

PREDICTOR_PATH = r"C:\Users\USER\Desktop\Modelo\shape_predictor_68_face_landmarks (1).dat"

# Inicializar contadores para contar el número de parpadeos
CONTADOR_PARPADEOS = 0

# Diccionario para almacenar las codificaciones faciales y los nombres de las imágenes de cada categoría
CATEGORY_ENCODINGS = {}
CATEGORY_NAMES = {}

# Directorio temporal para almacenar las imágenes aumentadas
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

def augment_data(categories, temp_dir):
    for category, folder_path in categories.items():
        pipeline = Augmentor.Pipeline(folder_path, output_directory=temp_dir)
        
        # Añadir más transformaciones
        pipeline.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        pipeline.flip_left_right(probability=0.5)
        pipeline.zoom_random(probability=0.5, percentage_area=0.8)
        pipeline.flip_top_bottom(probability=0.5)
        pipeline.skew(probability=0.5, magnitude=0.7)
        pipeline.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
        pipeline.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)

        num_augmented_images = 100
        try:
            pipeline.sample(num_augmented_images)
        except Exception as e:
            print(f"Error augmenting images for category {category}: {e}")
            continue

        augmented_images = os.listdir(temp_dir)
        for img in augmented_images:
            img_path = os.path.join(temp_dir, img)
            new_path = os.path.join(folder_path, img)
            try:
                os.rename(img_path, new_path)
            except Exception as e:
                print(f"Error moving augmented image {img}: {e}")

def load_face_encodings(categories):
    category_encodings = {}
    category_names = {}
    for category, folder_path in categories.items():
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"No hay archivos de imagen en la carpeta: {folder_path}")
            continue
        category_encodings[category] = []
        for img in image_files:
            image_path = os.path.join(folder_path, img)
            try:
                original_image = cv2.imread(image_path)
                resized_image = cv2.resize(original_image, (NUEVO_ANCHO, NUEVO_ALTO))
                face_encodings = face_recognition.face_encodings(resized_image)
                if face_encodings:
                    category_encodings[category].extend(face_encodings)
                else:
                    print(f"No se detectaron rostros en la imagen: {img}")
            except Exception as e:
                print(f"Error processing image {img}: {e}")
        category_names[category] = [os.path.splitext(img)[0] for img in image_files]
    return category_encodings, category_names

def calcular_razon_cierre_ojos(puntos_faciales):
    # Validación de entrada
    if len(puntos_faciales) <= 6:
        raise ValueError("Se esperaban 6 puntos faciales, pero se recibieron {}".format(len(puntos_faciales)))
    
    puntos_faciales_np = np.array(puntos_faciales)
    
    try:
        d_vert_izquierdo = np.linalg.norm(puntos_faciales_np[1] - puntos_faciales_np[5])
        d_vert_derecho = np.linalg.norm(puntos_faciales_np[2] - puntos_faciales_np[4])
        d_horiz = np.linalg.norm(puntos_faciales_np[0] - puntos_faciales_np[3])
        razon_cierre_ojos = (d_vert_izquierdo + d_vert_derecho) / (2.0 * d_horiz)
    except Exception as e:
        print("Error al calcular la razón de cierre de ojos: ", e)
        return None

    return razon_cierre_ojos

def capturar_foto(frame, capturas_directorio):
    global CONTADOR_PARPADEOS
    ahora = datetime.now()
    fecha_hora_str = ahora.strftime("%Y%m%d-%H%M%S")
    nombre_captura = f"captura_parpadeo_{CONTADOR_PARPADEOS + 1}-{fecha_hora_str}.jpg"
    ruta_captura = os.path.join(capturas_directorio, nombre_captura)
    cv2.imwrite(ruta_captura, frame)
    print(f"¡Foto capturada! ({nombre_captura})")

# Inicializar la ventana de Tkinter
window = tk.Tk()

# Obtener el tamaño de la pantalla
ANCHO_PANTALLA = 300
ALTO_PANTALLA = 400

# Cargar la imagen de fondo y escalarla al tamaño de la pantalla
RUTA_IMAGEN = r"C:\Users\USER\Desktop\Modelo\Interfaz\FONDO.jpg"
IMAGEN_FONDO = Image.open(RUTA_IMAGEN)
IMAGEN_FONDO = IMAGEN_FONDO.resize((ANCHO_PANTALLA, ALTO_PANTALLA), Image.ANTIALIAS if hasattr(Image, 'ANTIALIAS') else Image.LANCZOS)
IMAGEN_FONDO = ImageTk.PhotoImage(IMAGEN_FONDO)

# Establecer las dimensiones de la ventana al tamaño de la pantalla
window.geometry(f"{ANCHO_PANTALLA}x{ALTO_PANTALLA}")

# Configurar el tamaño del cuadro de video y colocarlo en el centro
ANCHO_CUADRO_VIDEO = 240
ALTO_CUADRO_VIDEO = 300
video_label = tk.Label(window, width=ANCHO_CUADRO_VIDEO, height=ALTO_CUADRO_VIDEO)
video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Crear el objeto VideoCapture
cap = cv2.VideoCapture(0)
    
# Crear un widget Label para la imagen de fondo
fondo_label = tk.Label(window, image=IMAGEN_FONDO)
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
# Agregar una variable para determinar si el sujeto es externo
ES_EXTERNO = False

def video_loop():
    global CONTADOR_PARPADEOS, ES_EXTERNO

    ret, frame = cap.read()
    try:
        face_locations = face_recognition.face_locations(frame, model="hog")
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return

    for face_location in face_locations:
        try:
            face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location], model="cnn")[0]
        except Exception as e:
            print(f"Error encoding face: {e}")
            continue

        dlib_faces = detector(frame, 0)

        if dlib_faces:
            landmarks = predictor(frame, dlib_faces[0])
            landmarks = [(p.x, p.y) for p in landmarks.parts()]
            razon_cierre_ojos = calcular_razon_cierre_ojos(landmarks[36:48])

            if razon_cierre_ojos < 0.2:
                CONTADOR_PARPADEOS += 1
                # Solo contar parpadeos si el sujeto es externo
                if ES_EXTERNO:
                    if CONTADOR_PARPADEOS == UMBRAL_PARPADEO:
                        capturar_foto(frame, "capturas")
                        CONTADOR_PARPADEOS = 0
            else:
                CONTADOR_PARPADEOS = 0

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            text = "Externo"
            for category, encodings in CATEGORY_ENCODINGS.items():
                face_matches = face_recognition.compare_faces(encodings, face_frame_encodings, tolerance=0.5)
                if True in face_matches:
                    text = category + " - " + CATEGORY_NAMES[category][face_matches.index(True)]
                    # Actualizar el estado del sujeto (interno o externo)
                    ES_EXTERNO = category == "Externo"
                    break
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image)
    label.config(image=photo)
    label.image = photo
    window.after(15, video_loop)

# Inicializar el aumento de datos
augment_data(CATEGORIES, TEMP_DIR)

# Cargar codificaciones faciales
CATEGORY_ENCODINGS, CATEGORY_NAMES = load_face_encodings(CATEGORIES)    

# Iniciar el bucle principal de Tkinter
window.after(100, video_loop)
window.mainloop()

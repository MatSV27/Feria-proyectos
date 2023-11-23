import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
from datetime import datetime

# Ruta de la carpeta para guardar las fotos
carpeta_destino = r"C:\Users\USER\Desktop\FOTOS EXTERNOS"

# Verificar si la carpeta de destino existe, si no, crearla
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Creamos la ventana principal
root = tk.Tk()
root.title("Interfaz de usuario")
root.attributes("-fullscreen", True)  # Hacer que la ventana ocupe toda la pantalla

# Creamos el objeto de la cámara web
captura = cv2.VideoCapture(0)

# Obtenemos las dimensiones originales del fotograma de la cámara
_, frame = captura.read()
altura_original, ancho_original, _ = frame.shape

# Calculamos las nuevas dimensiones para que el video ocupe el centro y tenga proporciones correctas
nueva_altura = root.winfo_screenheight()
factor_escala = nueva_altura / altura_original
nuevo_ancho = int(ancho_original * factor_escala)

# Creamos el widget de la imagen
imagen = tk.Label(root)
imagen.pack(expand=True)  # Hacer que la imagen ocupe todo el espacio disponible

frame = None  # Definimos frame como una variable global

def tomar_foto():
    global frame
    # Obtener la fecha y hora actual
    fecha_hora = datetime.now().strftime("%d-%m-%Y--H%H-M%M-S%S-%p")

    # Guardamos la imagen en la carpeta de destino con nombre que incluye fecha y hora
    nombre_archivo = f"{fecha_hora}.jpg"
    ruta_guardado = os.path.join(carpeta_destino, nombre_archivo)
    cv2.imwrite(ruta_guardado, frame)

# Creamos el botón de guardar
boton_guardar = tk.Button(root, text="Guardar", command=tomar_foto)
boton_guardar.place(relx=0.5, rely=0.95, anchor="s")  # Posicionar el botón en la parte inferior central

def actualizar_imagen():
    global frame
    # Capturamos el frame actual de la cámara web
    ret, frame = captura.read()

    # Redimensionamos el frame para que ocupe el centro con proporciones correctas
    frame = cv2.resize(frame, (nuevo_ancho, nueva_altura))

    # Convertimos el frame a un formato compatible con Tkinter
    frame_tk = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tk = Image.fromarray(frame_tk)
    frame_tk = ImageTk.PhotoImage(frame_tk)

    # Configuramos el nivel (z-order) de la imagen para que esté detrás del botón
    imagen.configure(image=frame_tk)
    imagen.image = frame_tk  # Mantenemos una referencia al objeto ImageTk
    imagen.lower(boton_guardar)  # Configuramos el nivel de la imagen para que esté detrás del botón

    # Llamamos a la función después de 10 milisegundos
    root.after(10, actualizar_imagen)

# Iniciamos el bucle principal
root.after(10, actualizar_imagen)
root.mainloop()

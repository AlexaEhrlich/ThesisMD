'''
##### FUNCIONAL #####
USO: Solo para grabación de video!
Requiere de buena entrada de datos, se ha trabajado con 4 cámaras y sin problema
'''

import cv2
import numpy as np
from loaders import LoadStreams

# La entrada de las cámaras es un archivo de texto, indicando las ubicaciones de las cámaras.
list_cameras = "YoloV8Loader/camaras.txt"

# Carga las cámaras usando la función LoadStreams
cameras = LoadStreams(list_cameras)

# Configuración del codec y VideoWriter para MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' es un códec compatible con MP4
output_files = []

# Configuración de FPS y duración en segundos
fps = 30  # Frames por segundo
duration_in_seconds = 30  # Duración en segundos
max_frames = fps * duration_in_seconds  # Total de frames a grabar

# Variables para comprobar si todas las cámaras están listas
camera_ready = False
frame_count = 0

# Verificar si todas las cámaras están listas
while not camera_ready:
    try:
        for batch in cameras:
            _, images, _ = batch
            
            # Verifica que todas las imágenes sean válidas
            if all(image is not None for image in images):
                camera_ready = True
                break
    except Exception as e:
        print(f"Error al verificar las cámaras: {e}")
        # Puedes añadir una pequeña espera para evitar un bucle excesivo
        cv2.waitKey(1000)

# Inicializar VideoWriter solo después de que todas las cámaras estén listas
for i, image in enumerate(images):
    if image is not None:
        height, width = image.shape[:2]
        video_writer = cv2.VideoWriter(f'down_{i}.mp4', fourcc, fps, (width, height))
        output_files.append(video_writer)

# Procesa cada batch de frames de las cámaras
while frame_count < max_frames:
    for batch in cameras:
        _, images, _ = batch
        
        for i, image in enumerate(images):
            if image is not None:
                # Graba el frame
                output_files[i].write(image)
        
        frame_count += 1
        if frame_count >= max_frames:
            break

# Libera todos los VideoWriters
for writer in output_files:
    writer.release()

# Cierra las ventanas de OpenCV (aunque no las usas)
cv2.destroyAllWindows()

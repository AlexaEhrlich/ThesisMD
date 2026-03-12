import cv2
import numpy as np

# Rutas de los 8 videos
video_paths = ['output_camera_0.mp4', 'output_camera_1.mp4', 'output_camera_2.mp4', 'output_camera_3.mp4',
               'output_camera_4.mp4', 'output_camera_5.mp4', 'output_camera_6.mp4', 'output_camera_7.mp4']

# Abrir los 8 videos
caps = [cv2.VideoCapture(path) for path in video_paths]

# Comprobar si todos los videos se abrieron correctamente
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error al abrir el video {i+1}")
        exit()

# Obtener las dimensiones de los videos (asumiendo que todos tienen la misma resolución)
width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir el tamaño del video combinado (4x2)
output_width = width * 4  # 4 videos en la fila
output_height = height * 2  # 2 filas

# Crear el VideoWriter para guardar el video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (output_width, output_height))

while True:
    # Leer un frame de cada video
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Si alguno de los videos terminó, romper el bucle
    if len(frames) < 8:
        break

    # Redimensionar todos los frames al mismo tamaño
    frames_resized = [cv2.resize(frame, (width, height)) for frame in frames]

    # Crear una ventana de 4x2 para mostrar los 8 videos
    top_row = np.hstack((frames_resized[0], frames_resized[1], frames_resized[2], frames_resized[3]))
    bottom_row = np.hstack((frames_resized[4], frames_resized[5], frames_resized[6], frames_resized[7]))
    combined_frame = np.vstack((top_row, bottom_row))

    # Mostrar los 8 videos en la misma ventana
    cv2.imshow('Videos', combined_frame)

    # Guardar el frame combinado en el archivo de salida
    out.write(combined_frame)

    # Ajustar a 33 para 30 fps
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Liberar los recursos
for cap in caps:
    cap.release()
out.release()
cv2.destroyAllWindows()

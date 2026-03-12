""""
USO: ARCHIVO PARA REALIZAR TRIANGULACION ITERATIVA Y GUARDAR LOS PUNTOS 3D
"""
import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Inicializar MediaPipe Pose (para detección de puntos clave del cuerpo)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def obtener_puntos_clave(frame):
    """
    Detecta puntos clave 2D del cuerpo en un frame usando MediaPipe Pose.
    Entrada: frame (numpy.ndarray): Imagen en formato BGR.
    Salida: np.ndarray con coordenadas (x, y) de los landmarks detectados, o None si no se detectan puntos clave.
    """
    resultados = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if resultados.pose_landmarks:
        return np.array([(lm.x, lm.y) for lm in resultados.pose_landmarks.landmark])
    return None

def triangular_iterativa(matrices_proyeccion, puntos):
    """
    Realiza triangulación lineal de un punto observado desde varias cámaras.
    Entrada: matrices_proyeccion (list): Matrices de proyección de cada cámara.
            puntos (list): Lista de coordenadas 2D observadas en cada cámara.
    Salida: np.ndarray con las coordenadas 3D estimadas del punto.
    """
    # Matriz 2xN con coordenadas
    puntos = np.array(puntos, dtype=np.float32).T
     # Triangulación con OpenCV (solo usa las dos primeras cámaras)
    puntos_h = cv2.triangulatePoints(matrices_proyeccion[0], matrices_proyeccion[1], puntos[:2, 0], puntos[:2, 1])
    # Convertir a coordenadas euclidianas
    puntos_3d = puntos_h[:3] / puntos_h[3]  
    return puntos_3d.flatten()

def reproyectar_puntos(puntos_3d, matrices_proyeccion):
    """
    Reproyecta los puntos 3D al plano 2D de cada cámara.
    Entrada: puntos_3d (np.ndarray): Puntos en coordenadas 3D.
             matrices_proyeccion (list): Matrices de proyección de las cámaras.
    Salida: list con arrays de coordenadas 2D reproyectadas en cada cámara.
    """
    puntos_2d_reproyectados = []
    for P in matrices_proyeccion:
        puntos_h = np.hstack((puntos_3d, np.ones((puntos_3d.shape[0], 1))))
        reproyectados_h = (P @ puntos_h.T).T
        reproyectados_2d = reproyectados_h[:, :2] / reproyectados_h[:, 2:]
        puntos_2d_reproyectados.append(reproyectados_2d)
    return puntos_2d_reproyectados

def calcular_error_reproyeccion(puntos_2d_reales, puntos_2d_reproyectados):
    """
    Calcula el error promedio entre los puntos 2D reales y los reproyectados.
    Entrada: puntos_2d_reales (list): Coordenadas reales detectadas en cada cámara.
             puntos_2d_reproyectados (list): Coordenadas reproyectadas desde 3D.
    Salida: float con el error medio de reproyección en píxeles.
    """
    errores = []
    for reales, reproyectados in zip(puntos_2d_reales, puntos_2d_reproyectados):
        error = np.linalg.norm(reales - reproyectados, axis=1)
        errores.append(np.mean(error))
    return np.mean(errores)

def guardar_puntos_2d_por_video(video_path, output_path, nombre_video):
    """
    Extrae puntos 2D de un video con MediaPipe y los guarda en archivo.
    Entrada: video_path (str): Ruta al archivo de video.
             output_path (str): Ruta del archivo de salida (.txt).
             nombre_video (str): Nombre identificador del video.
    Salida: Archivo de texto con coordenadas (x, y) de los landmarks.
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    puntos_guardados = 0

    with open(output_path, "w") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 10 == 0:
                puntos = obtener_puntos_clave(frame)
                if puntos is not None:
                    linea = ", ".join([f"{x:.6f}, {y:.6f}" for x, y in puntos])
                    f.write(linea + "\n")
                    puntos_guardados += 1

            frame_idx += 1

    cap.release()

def guardar_puntos(puntos_3d):
    """
    Guarda puntos 3D en un archivo de texto.
    Entrada: puntos_3d (np.ndarray): Lista de puntos 3D en formato (N, 3).
    Salida: Se añade una línea en el archivo coordenadas_3D.txt.
    """
    #Guarda los puntos 3D en un archivo de texto sin el número de frame
    with open(output_file, "a") as f:
        line = ", ".join([f"{x:.6f}, {y:.6f}, {z:.6f}" for x, y, z in puntos_3d]) + "\n"
        f.write(line)

# Definir ruta de salida del archivo de coordenadas
output_file = "coordenadas_3D.txt"

# Lista de videos desde distintas cámaras
video_paths = [
    "Dataset/Samantha/6_1.mp4",
    "Dataset/Samantha/6_3.mp4",
    "Dataset/Samantha/6_5.mp4",
    "Dataset/Samantha/6_7.mp4"
]
videos = [cv2.VideoCapture(path) for path in video_paths]

#Cargar matrices
try:
    matrices_proyeccion = [
        np.loadtxt("AutoCalibration/calibracion3/cam1_projection.txt"),
        np.loadtxt("AutoCalibration/calibracion3/cam3_projection.txt"),
        np.loadtxt("AutoCalibration/calibracion3/cam5_projection.txt"),
        np.loadtxt("AutoCalibration/calibracion3/cam7_projection.txt")
    ]
except Exception as e:
    print(f"Error al cargar las matrices de proyección: {e}")

# Determinar el número mínimo de frames disponibles en todos los videos
total_frames = min(int(v.get(cv2.CAP_PROP_FRAME_COUNT)) for v in videos)
frame_idx = 0 

# Guardar puntos 2D de cada cámara en archivos separados
for i, path in enumerate(video_paths):
    nombre_archivo = f"puntos_cam{(i*2)+1}.txt"
    guardar_puntos_2d_por_video(path, nombre_archivo, nombre_archivo)

# Lista para almacenar los errores de reproyección
errores_reproyeccion = []

# Procesamiento frame a frame (salto de 10 en 10)
while frame_idx < total_frames:
    puntos_2d = []
    # Extraer puntos 2D de cada cámara en el frame actual
    for video in videos:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if not ret:
            continue
        puntos_cam = obtener_puntos_clave(frame)
        if puntos_cam is None:
            continue
        puntos_2d.append(puntos_cam)
    # Si no se tienen todos los puntos de todas las cámaras, saltar fram
    if len(puntos_2d) < len(videos):
        frame_idx += 10
        continue
    # Triangulación de cada punto clave
    puntos_3d = []
    #Iterar sobre cada landmark
    for i in range(len(puntos_2d[0])):
        punto_3d = triangular_iterativa(matrices_proyeccion, [p[i] for p in puntos_2d])
        if punto_3d is not None:
            puntos_3d.append(punto_3d)

    if len(puntos_3d) > 0:
        puntos_3d = np.array(puntos_3d)
        #Normalizar puntos para la visualización
        puntos_3d -= puntos_3d.mean(axis=0)
        puntos_3d /= np.max(np.abs(puntos_3d))
        
        # Visualización del esqueleto en 3D (solo para debug o inspección)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.grid(False)
        ax.set_axis_off()

        # Dibujar los puntos 3D
        ax.scatter(puntos_3d[:, 0], puntos_3d[:, 1], puntos_3d[:, 2], c='blue', marker='o')

        # Dibujar las conexiones del esqueleto
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(puntos_3d) and end_idx < len(puntos_3d):
                x_vals = [puntos_3d[start_idx, 0], puntos_3d[end_idx, 0]]
                y_vals = [puntos_3d[start_idx, 1], puntos_3d[end_idx, 1]]
                z_vals = [puntos_3d[start_idx, 2], puntos_3d[end_idx, 2]]
                ax.plot(x_vals, y_vals, z_vals, 'r-')

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.title(f"Frame {frame_idx} - Esqueleto 3D")
        plt.show()

        # Guardar puntos 3D en archivo
        guardar_puntos(puntos_3d)

        # Calcular error de reproyección y guardarlo
        puntos_2d_reproyectados = reproyectar_puntos(puntos_3d, matrices_proyeccion)
        error_reproyeccion = calcular_error_reproyeccion(puntos_2d, puntos_2d_reproyectados)
        errores_reproyeccion.append(error_reproyeccion)

    print(f"Procesado frame {frame_idx}/{total_frames}") 
    frame_idx += 10  # Avanzar 10 frames

cv2.destroyAllWindows()

# Calcular y mostrar el error de reproyección promedio al final
if errores_reproyeccion:
    error_promedio = np.mean(errores_reproyeccion)
    print(f"\nError de reproyección promedio en todos los frames: {error_promedio:.4f} píxeles")
else:
    print("\nNo se pudo calcular el error de reproyección.")

print("Proceso completado. Coordenadas guardadas en 'coordenadas_3D.txt'.")

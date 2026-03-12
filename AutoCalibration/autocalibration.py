import cv2
import numpy as np
import glob
import os

# Parámetros del tablero de ajedrez
chessboard_size = (15, 12)
square_size = 4.0  

# Criterios de refinamiento para la detección de esquinas
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Puntos del mundo real 
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Listas para almacenar los puntos 3D y 2D detectados
objpoints = []  # Puntos 3D en el mundo real
imgpoints = []  # Puntos 2D en la imagen

# Cargar imágenes de calibración AutoCalibration\ImagenesCalibracion\2_cam1
folder_path = os.path.join( "AutoCalibration", "ImagenesCalibracion", "3_cam7")
images = glob.glob(os.path.join(folder_path, "*.jpg"))

if not images:
    print(f"No se encontraron imágenes en {folder_path}. Verifica la ruta.")
    exit()

# Variable para almacenar el tamaño de la imagen
gray = None  

for image_name in images:
    print(f"Procesando imagen: {image_name}")
    img = cv2.imread(image_name)

    if img is None:
        print(f"No se pudo cargar la imagen: {image_name}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔹 Buscar esquinas del tablero de ajedrez
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(refined_corners)

        # Dibujar y mostrar las esquinas detectadas
        cv2.drawChessboardCorners(img, chessboard_size, refined_corners, ret)
        cv2.imshow(f'Detected Corners - {os.path.basename(image_name)}', img)
        cv2.waitKey(500)
    else:
        print(f"No se detectaron esquinas en {image_name}")

cv2.destroyAllWindows()

#  Verificar si se encontraron esquinas en al menos una imagen
if gray is not None and objpoints and imgpoints:
    print(f"📸 Se detectaron esquinas en {len(objpoints)} imágenes.")
    
    # Calibración de cámara
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret and rvecs and tvecs:
        # Calcular el error medio de reproyección
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(objpoints)
        print(f"Error medio de reproyección: {mean_error:.6f} píxeles")
        # Convertir el vector de rotación a matriz de rotación 3x3
        rvec_matrix, _ = cv2.Rodrigues(rvecs[0])

        #  Asegurar que la traslación (tvec) es un vector columna de 3x1
        tvec_matrix = tvecs[0].reshape(3, 1)

        # Construir la matriz extrínseca 3x4 (rotación + traslación)
        extrinsic_matrix = np.hstack((rvec_matrix, tvec_matrix))

        # Calcular la matriz de proyección 3x4
        projection_matrix = camera_matrix @ extrinsic_matrix  # K @ [R|t]

        # Verificar dimensiones antes de guardar
        print("Dimensiones de la matriz de proyección:", projection_matrix.shape)  # Debe ser (3,4)

        # Directorio para guardar los archivos
        output_dir = "Thesis"
        os.makedirs(output_dir, exist_ok=True)

        # Guardar la matriz intrínseca 3x3
        intrinsic_path = os.path.join(output_dir, "cam1_intrinsic.txt")
        np.savetxt(intrinsic_path, camera_matrix, fmt='%0.6f')

        # Guardar la matriz de proyección 3x4
        projection_path = os.path.join(output_dir, "cam1_projection.txt")
        np.savetxt(projection_path, projection_matrix, fmt='%0.6f')

        print(f"Matriz intrínseca (3x3) guardada en '{intrinsic_path}'")
        print(f"Matriz de proyección (3x4) guardada en '{projection_path}'")

    else:
        print("Error en la calibración: No se pudo calcular la matriz de proyección.")

else:
    print("No se detectaron suficientes esquinas para calibrar la cámara.")



"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image_path = "ThesisAle/AutoCalibration/ImagenesCalibracion/cam1/WIN_20250211_13_56_53_Pro.jpg"
img = cv2.imread(image_path)

if img is None:
    print("❌ Error: No se pudo cargar la imagen. Revisa la ruta.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Rango de búsqueda de tamaños de patrón (intersecciones)
min_size = 5
max_size = 20

best_size = None
best_corners = 0

# Probar diferentes combinaciones de intersecciones (filas x columnas)
for rows in range(min_size, max_size):
    for cols in range(min_size, max_size):
        chessboard_size = (cols, rows)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret and corners.shape[0] > best_corners:
            best_size = chessboard_size
            best_corners = corners.shape[0]
            best_corners_positions = corners

# Dibujar el mejor patrón encontrado
if best_size:
    ret, corners = cv2.findChessboardCorners(gray, best_size, None)
    cv2.drawChessboardCorners(img, best_size, corners, ret)
    message = f"✅ Patrón detectado: {best_size[0]}x{best_size[1]} intersecciones"
else:
    message = "❌ No se detectó el patrón"

# Convertir imagen a RGB para visualizar correctamente
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Mostrar la imagen con la detección
plt.figure(figsize=(10, 6))
plt.imshow(img_rgb)
plt.title(message)
plt.axis("off")
plt.show()

# Devolver el mejor tamaño detectado
print(f"Tamaño detectado del tablero: {best_size}" if best_size else "No se detectó el patrón.")
"""
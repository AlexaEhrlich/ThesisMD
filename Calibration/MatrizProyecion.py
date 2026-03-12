'''
USO: 
Usar unicamente la primera vez para obtener las matrices de proyección, se guardan en archivos TXT
Este código permite hacer la calibración por medio de relación de puntos 3D y los mismos puntos en la imagen capturada
'''

import numpy as np
from scipy.linalg import svd

def ensure_homogeneous(points, dim):
    """
    Asegura que los puntos estén en coordenadas homogéneas.
    """
    if points.shape[1] != dim:
        ones = np.ones((points.shape[0], 1))
        points = np.hstack([points, ones])
    return points

def calibrate_camera(points_2d, points_3d):
    """
    Realiza la calibración de la cámara utilizando un sistema de 12 incógnitas.
    """
    if len(points_2d) != len(points_3d):
        raise ValueError("El número de puntos 2D y 3D debe ser el mismo.")

    num_points = len(points_2d)
    A = []

    for i in range(num_points):
        X, Y, Z, W = points_3d[i]
        x, y, w = points_2d[i]

        A.append([X, Y, Z, W, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x*W])
        A.append([0, 0, 0, 0, X, Y, Z, W, -y*X, -y*Y, -y*Z, -y*W])

    A = np.array(A)
    U, S, Vt = svd(A)
    P = Vt[-1].reshape(3, 4)
    P = P / P[-1, -1]


    return P

def calculate_reprojection_error(P, points_2d, points_3d):
    """
    Calcula el error de reproyección promedio.
    """
    total_error = 0
    num_points = len(points_2d)

    for i in range(num_points):
        point_3d = points_3d[i]
        projected_2d = P @ point_3d
        projected_2d /= projected_2d[2]  # Normalización

        error = np.linalg.norm(points_2d[i][:2] - projected_2d[:2])
        total_error += error

    mean_error = total_error / num_points
    return mean_error

def save_projection_matrix_txt(P, filename="matriz_proyeccion.txt"):
    """
    Guarda la matriz de proyección en un archivo de texto.
    """
    np.savetxt(filename, P, fmt='%.6f')
    print(f"Matriz de proyección guardada en {filename}")

# Datos
points_2d = np.array([
    [241, 1269],
    [233, 1180],
    [310, 1199],
    [340, 1112],
    [376, 1184],
    [418, 1114],
    [437, 1188],
    [505, 1192]
])

points_3d = np.array([
    [-36, 0, 4],
    [-34, 0, 30],
    [-12, 0, 6],
    [-2, 0, 22],
    [0, -6, 2],
    [0, -18, 30],
    [0, -22, 10],
    [0, -38, 18]
])

# Asegurar que los puntos estén en coordenadas homogéneas
points_2d = ensure_homogeneous(points_2d, 3)
points_3d = ensure_homogeneous(points_3d, 4)

# Calibrar la cámara
P = calibrate_camera(points_2d, points_3d)
print("Matriz de Proyección P:")
print(P)

# Calcular el error de reproyección
error = calculate_reprojection_error(P, points_2d, points_3d)
print(f"Error de Reproyección Promedio: {error:.4f} píxeles")


#ROTACION DE P
rotada_90 = np.rot90(P)
print(rotada_90)
# Guardar la matriz de proyección en un archivo .txt
save_projection_matrix_txt(rotada_90, "matriz_proyeccion_cam7_v2.txt")

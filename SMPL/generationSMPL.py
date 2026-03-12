"""
USO: GENERACIÓN DEL MODELO SMPL
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import torch
from smplx import SMPL
import trimesh
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def leer_todas_las_articulaciones(archivo):
    """
    Lee un archivo con coordenadas 3D de articulaciones.
    Entrada: archivo de texto con líneas separadas por coma en formato x,y,z.
    Salida: lista de frames, donde cada frame contiene una lista de articulaciones [x,y,z].
    """
    with open(archivo, 'r') as f:
        # Lista para almacenar todos los frames
        frames = []  

        for linea in f:
            # Convertir a flotantes
            valores = list(map(float, linea.strip().split(", ")))
            # Agrupar los valores en cada articulación
            articulaciones = [valores[i:i+3] for i in range(0, len(valores), 3)]
            frames.append(articulaciones)
    # Regresar lista de frames
    return frames 

def generar_rotaciones_smpl(rotaciones_axis_angle):
    """
    Convierte las rotaciones calculadas de articulaciones a formato SMPL.
    Entrada: diccionario con rotaciones en axis-angle por articulación.
    Salida: lista de 24 rotaciones axis-angle ordenadas según el modelo SMPL.
    """
    # Inicializar todas las rotaciones en cero
    smpl_rotaciones = [np.array([0, 0, 0])] * 24

    # Solo activamos las articulaciones de la parte inferior + torso + brazos
    mapeo_smpl = {
    0: "torso",
    1: "muslo_izquierdo",
    2: "muslo_derecho",
    4: "pantorrilla_izquierda",
    5: "pantorrilla_derecha",
    7: "pie_izquierdo",
    8: "pie_derecho",
    16: "brazo_izquierdo",
    17: "brazo_derecho",
    18: "antebrazo_izquierdo",
    19: "antebrazo_derecho",
    20: "mano_izquierda",
    21: "mano_derecha"
}

    #Asignar las rotaciones calculadas a a los índices correspondientes del modelo SMPL
    for indice_smpl, nombre_vector in mapeo_smpl.items():
        if nombre_vector in rotaciones_axis_angle:
            smpl_rotaciones[indice_smpl] = np.array(rotaciones_axis_angle[nombre_vector])
    #Regresar rotaciones
    return smpl_rotaciones


def generar_malla_smpl(smpl_model, rotaciones_smpl, betas=None):
    """
    Genera una malla 3D a partir del modelo SMPL y las rotaciones dadas.
    Entrada: modelo SMPL cargado, rotaciones en formato SMPL, betas opcionales (forma del cuerpo).
    Salida: vértices y caras de la malla generada.
    """
    # Usar GPU si está disponible, de lo contrario CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Asignamos la rotación global (orientación del torso) como la primera articulación
    global_orient = torch.tensor(np.array(rotaciones_smpl[0]), dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3)
    # Resto de articulaciones (cuerpo)
    body_pose_array = np.array(rotaciones_smpl[1:24])  # (23, 3)
    body_pose = torch.tensor(body_pose_array, dtype=torch.float32, device=device).reshape(1, -1)  # (1, 69)

    # Si no hay betas personalizados (parámetros de forma), usar vector nulo
    if betas is None:
        betas = torch.zeros(1, 10, dtype=torch.float32, device=device)

    smpl_model = smpl_model.to(device)

    #Generar salida del modelo SMPL (véritces y caras de las mallas)
    output = smpl_model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas
    )

    return output.vertices[0].cpu().detach().numpy(), smpl_model.faces

def visualizar_malla(vertices, faces):
    """
    Visualiza una malla 3D utilizando la librería trimesh.
    Entrada: vértices y caras de la malla.
    Salida: ventana interactiva de visualización.
    """
    #Visualiza una malla 3D con trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.show()

def visualizar_malla_simple(vertices):
    """
    Visualiza la malla como una nube de puntos con matplotlib.
    Entrada: lista de vértices 3D.
    Salida: gráfico en 3D mostrado en pantalla.
    """
    #Visualización con nubes de puntos
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.1)
    ax.set_title("SMPL Frame")
    plt.pause(0.10)
    plt.clf()

def alinear_vector(v):
    """
    Reordena y cambia el signo de los ejes para que coincidan con el sistema de SMPL.
    Entrada: v = [x, y, z] del sistema original
    Salida:  vector alineado al sistema de SMPL
    """
    x, y, z = v
    return [x, z, -y]

def calcular_vectores_alineados(frame):
    """
    Calcula los vectores de segmentos (huesos) a partir de un frame.
    Entrada: frame con coordenadas 3D de articulaciones.
    Salida: diccionario con vectores de segmentos clave (muslos, brazos, torso, etc.).
    """
    # Alinear todos los puntos del frame al sistema SMPL
    frame_alineado = [alinear_vector(p) for p in frame]

    #Calcular vectores de segmentos principales restando las articulaciones
    vectores = {
        "muslo_izquierdo": [frame_alineado[25][i] - frame_alineado[23][i] for i in range(3)],
        "muslo_derecho": [frame_alineado[26][i] - frame_alineado[24][i] for i in range(3)],
        "torso": [((frame_alineado[23][i] + frame_alineado[24][i]) / 2) - ((frame_alineado[11][i] + frame_alineado[12][i]) / 2) for i in range(3)],
        "pantorrilla_izquierda": [frame_alineado[27][i] - frame_alineado[25][i] for i in range(3)],
        "pantorrilla_derecha": [frame_alineado[28][i] - frame_alineado[26][i] for i in range(3)],
        "pie_izquierdo": [frame_alineado[31][i] - frame_alineado[27][i] for i in range(3)],
        "pie_derecho": [frame_alineado[32][i] - frame_alineado[28][i] for i in range(3)],
        "brazo_izquierdo": [frame_alineado[13][i] - frame_alineado[11][i] for i in range(3)],
        "brazo_derecho": [frame_alineado[14][i] - frame_alineado[12][i] for i in range(3)],
        "antebrazo_izquierdo": [frame_alineado[15][i] - frame_alineado[13][i] for i in range(3)],
        "antebrazo_derecho": [frame_alineado[16][i] - frame_alineado[14][i] for i in range(3)],
        "mano_izquierda": [frame_alineado[17][i] - frame_alineado[15][i] for i in range(3)],
        "mano_derecha": [frame_alineado[18][i] - frame_alineado[16][i] for i in range(3)],
    }
    #Regresar los vectores calculadoos
    return vectores

def rotacion_entre_vectores(v_from, v_to, nombre=""):
    """
    Calcula la rotación necesaria para alinear un vector con otro.
    Entrada: v_from (vector inicial), v_to (vector destino).
    Salida: objeto Rotation que representa la rotación.
    """
    v1 = np.array(v_from, dtype=float)
    v2 = np.array(v_to, dtype=float)
    #Normalización
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    ## Si algún vector es nulo, no hay rotación
    if v1_norm == 0 or v2_norm == 0:
        return None

    v1 /= v1_norm
    v2 /= v2_norm

    # Calcular ángulo entre vectores
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)

    #Si el ángulo es muy pequeñoo omitirlo
    if np.degrees(angle_rad) < 1.0:
        return None

    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        # Vectores opuestos — eje indefinido, generamos uno ortogonal arbitrario
        axis = np.cross(v1, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v1, [0, 1, 0])
        axis /= np.linalg.norm(axis)
    else:
        axis /= axis_norm

    # Construir rotación con vector rotacional
    rot = R.from_rotvec(axis * angle_rad)
    return rot

def obtener_betas(path_archivo):
    """
    Genera parámetros de forma (betas) para SMPL a partir de medidas antropométricas.
    Entrada: archivo con medidas estimadas.
    Salida: tensor de betas con valores normalizados entre [-3, 3].
    """
    # Leer medidas
    with open(path_archivo, "r") as f:
        # Saltar encabezado
        f.readline()
        linea = f.readline().strip()
        medidas = np.array([float(valor) for valor in linea.split(",")])

    # Normalizar al rango [-3, 3]
    min_vals = np.array([50, 60, 30, 25, 80, 100, 10])
    max_vals = np.array([80, 100, 70, 50, 130, 200, 25])
    medidas_norm = 6 * ((medidas - min_vals) / (max_vals - min_vals)) - 3
    medidas_norm = np.clip(medidas_norm, -3, 3)

    # Construir betas personalizados (forma del cuerpo)
    betas = torch.tensor([[
        0.0, 0.0, medidas_norm[2], medidas_norm[0], medidas_norm[3],
        medidas_norm[5], medidas_norm[6], medidas_norm[1], 0.0, 0.0
    ]], dtype=torch.float32)
    #Devolver el vector de betas
    return betas

def error_angular(v1, v2):
    """
    Calcula el error angular entre dos vectores.
    Entrada: dos vectores 3D.
    Salida: ángulo en grados entre los vectores.
    """
    # Calcula error angular entre dos vectores (en grados)
    v1 = np.array(v1) / np.linalg.norm(v1)
    v2 = np.array(v2) / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def longitud(v):
    """
    Calcula la magnitud de un vector.
    Entrada: vector 3D.
    Salida: valor escalar de la norma.
    """
    #Calcular magnitud de un vector
    return np.linalg.norm(v)

#### BLOQUE PRINCIPAL ######

# Crear carpeta de salida si no existe
Path("salida_mallas").mkdir(exist_ok=True)

# Cargar los frames y medidas
frames = leer_todas_las_articulaciones("coordenadas_3D.txt")
betas_personalizadas = obtener_betas("MedidasEstimadas.txt")

# Cargar el modelo SMPL desde su ubicación
MODEL_PATH = "C:/Users/AlexandraEhrlich/Desktop/models/neutral.pkl"
smpl_model = SMPL(model_path=MODEL_PATH, gender="neutral", batch_size=1)

# Calcular los vectores de referencia de la T-Pose (en el primer frame)
vectores_neutros = calcular_vectores_alineados(frames[0])

# Inicializar estructura para guardar las métricas de error
errores = []

# Recorrer todos los frames
for idx, frame in enumerate(frames):
    vectores_frame = calcular_vectores_alineados(frame)
    rotaciones_axis_angle = {}

    # ---------- LADO IZQUIERDO ----------
    rot_muslo_i = rotacion_entre_vectores(vectores_neutros["muslo_izquierdo"], vectores_frame["muslo_izquierdo"])
    if rot_muslo_i:
        rotaciones_axis_angle["muslo_izquierdo"] = rot_muslo_i.as_rotvec()
        pantorrilla_i_neutra_rotada = rot_muslo_i.apply(vectores_neutros["pantorrilla_izquierda"])
    else:
        pantorrilla_i_neutra_rotada = vectores_neutros["pantorrilla_izquierda"]

    rot_pantorrilla_i = rotacion_entre_vectores(pantorrilla_i_neutra_rotada, vectores_frame["pantorrilla_izquierda"])
    if rot_pantorrilla_i:
        rotaciones_axis_angle["pantorrilla_izquierda"] = rot_pantorrilla_i.as_rotvec()
        pie_i_neutro_rotado = rot_pantorrilla_i.apply(vectores_neutros["pie_izquierdo"])
    else:
        pie_i_neutro_rotado = vectores_neutros["pie_izquierdo"]

    rot_pie_i = rotacion_entre_vectores(pie_i_neutro_rotado, vectores_frame["pie_izquierdo"])
    if rot_pie_i:
        rotaciones_axis_angle["pie_izquierdo"] = rot_pie_i.as_rotvec()

    rot_brazo_i = rotacion_entre_vectores(vectores_neutros["brazo_izquierdo"], vectores_frame["brazo_izquierdo"])
    if rot_brazo_i:
        rotaciones_axis_angle["brazo_izquierdo"] = rot_brazo_i.as_rotvec()
        antebrazo_i_neutro_rotado = rot_brazo_i.apply(vectores_neutros["antebrazo_izquierdo"])
    else:
        antebrazo_i_neutro_rotado = vectores_neutros["antebrazo_izquierdo"]

    rot_antebrazo_i = rotacion_entre_vectores(antebrazo_i_neutro_rotado, vectores_frame["antebrazo_izquierdo"])
    if rot_antebrazo_i:
        rotaciones_axis_angle["antebrazo_izquierdo"] = rot_antebrazo_i.as_rotvec()
        mano_i_neutra_rotada = rot_antebrazo_i.apply(vectores_neutros["mano_izquierda"])
    else:
        mano_i_neutra_rotada = vectores_neutros["mano_izquierda"]

    rot_mano_i = rotacion_entre_vectores(mano_i_neutra_rotada, vectores_frame["mano_izquierda"])
    if rot_mano_i:
        rotaciones_axis_angle["mano_izquierda"] = rot_mano_i.as_rotvec()

    # ---------- LADO DERECHO ----------
    rot_muslo_d = rotacion_entre_vectores(vectores_neutros["muslo_derecho"], vectores_frame["muslo_derecho"])
    if rot_muslo_d:
        rotaciones_axis_angle["muslo_derecho"] = rot_muslo_d.as_rotvec()
        pantorrilla_d_neutra_rotada = rot_muslo_d.apply(vectores_neutros["pantorrilla_derecha"])
    else:
        pantorrilla_d_neutra_rotada = vectores_neutros["pantorrilla_derecha"]

    rot_pantorrilla_d = rotacion_entre_vectores(pantorrilla_d_neutra_rotada, vectores_frame["pantorrilla_derecha"])
    if rot_pantorrilla_d:
        rotaciones_axis_angle["pantorrilla_derecha"] = rot_pantorrilla_d.as_rotvec()
        pie_d_neutro_rotado = rot_pantorrilla_d.apply(vectores_neutros["pie_derecho"])
    else:
        pie_d_neutro_rotado = vectores_neutros["pie_derecho"]

    rot_pie_d = rotacion_entre_vectores(pie_d_neutro_rotado, vectores_frame["pie_derecho"])
    if rot_pie_d:
        rotaciones_axis_angle["pie_derecho"] = rot_pie_d.as_rotvec()

    rot_brazo_d = rotacion_entre_vectores(vectores_neutros["brazo_derecho"], vectores_frame["brazo_derecho"])
    if rot_brazo_d:
        rotaciones_axis_angle["brazo_derecho"] = rot_brazo_d.as_rotvec()
        antebrazo_d_neutro_rotado = rot_brazo_d.apply(vectores_neutros["antebrazo_derecho"])
    else:
        antebrazo_d_neutro_rotado = vectores_neutros["antebrazo_derecho"]

    rot_antebrazo_d = rotacion_entre_vectores(antebrazo_d_neutro_rotado, vectores_frame["antebrazo_derecho"])
    if rot_antebrazo_d:
        rotaciones_axis_angle["antebrazo_derecho"] = rot_antebrazo_d.as_rotvec()
        mano_d_neutra_rotada = rot_antebrazo_d.apply(vectores_neutros["mano_derecha"])
    else:
        mano_d_neutra_rotada = vectores_neutros["mano_derecha"]

    rot_mano_d = rotacion_entre_vectores(mano_d_neutra_rotada, vectores_frame["mano_derecha"])
    if rot_mano_d:
        rotaciones_axis_angle["mano_derecha"] = rot_mano_d.as_rotvec()

    # ---------- TORSO ----------
    rot_torso = rotacion_entre_vectores(vectores_neutros["torso"], vectores_frame["torso"])
    if rot_torso:
        rotaciones_axis_angle["torso"] = rot_torso.as_rotvec()

    # Generar las rotaciones en el formato de SMPL
    smpl_rotaciones_final = generar_rotaciones_smpl(rotaciones_axis_angle)

    # Generar la malla SMPL coon las rotaciones y los betas
    vertices, faces = generar_malla_smpl(smpl_model, smpl_rotaciones_final, betas=betas_personalizadas)

    # Exportar el frame como .obj
    malla = trimesh.Trimesh(vertices=vertices, faces=faces)
    malla.export(f"salida_mallas/frame_{idx:04d}.obj")

    print(f"Frame {idx} exportado como frame_{idx:04d}.obj")

    # Cálculo de errores por segmento
    for nombre_segmento, vector_neutro in vectores_neutros.items():
        vector_frame = vectores_frame[nombre_segmento]
        if nombre_segmento in rotaciones_axis_angle:
            rot = R.from_rotvec(rotaciones_axis_angle[nombre_segmento])
            vector_rotado = rot.apply(vector_neutro)

            ang_error = error_angular(vector_rotado, vector_frame)
            len_error = abs(longitud(vector_frame) - longitud(vector_neutro))

            errores.append({
                "Frame": idx,
                "Segmento": nombre_segmento,
                "Error Angular (°)": round(ang_error, 2),
                "Error Longitud (cm)": round(len_error, 2)
            })
    
# Guardar resultados como CSV
df_errores = pd.DataFrame(errores)
df_errores.to_csv("ErroresSegmentos.csv", index=False)
print("Errores guardados en errores_segmentos.csv")
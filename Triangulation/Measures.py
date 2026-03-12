import numpy as np

# Rutas de los archivos
archivo_entrada = "coordenadas_3D.txt"
archivo_salida = "MedidasEstimadas.txt"

# Índices de las articulaciones en MediaPipe
indices = {
    "hombro_izq": 11, "hombro_der": 12,
    "cadera_izq": 23, "cadera_der": 24,
    "muñeca_izq": 15, "muñeca_der": 16,
    "rodilla_izq": 25, "rodilla_der": 26,
    "talon_izq": 29, "talon_der": 30
}

def calcular_distancia(p1, p2):
    """
    Calcula la distancia euclidiana entre dos puntos 3D.

    Entrada:
        p1 (list/array): Coordenadas [x, y, z] del primer punto.
        p2 (list/array): Coordenadas [x, y, z] del segundo punto.

    Salida:
        float: Distancia euclidiana entre ambos puntos.
    """
    #Calcula la distancia euclidiana entre dos puntos en 3D.
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Leer archivo de coordenadas 3D (cada linea es un frame)
with open(archivo_entrada, "r") as f:
    lineas = f.readlines()

# Inicializar factor de escala
factor_escala = None

# Procesar cada línea (frame)
medidas = []
for linea in lineas:
    valores = list(map(float, linea.strip().split(",")))
    
    # Obtener coordenadas de cada articulación
    puntos_3D = [valores[i:i+3] for i in range(0, len(valores), 3)]

    # Calcular el factor de escala solo en el primer frame
    if factor_escala is None:
        distancia_hombros = calcular_distancia(puntos_3D[indices["hombro_izq"]], puntos_3D[indices["hombro_der"]])
        distancia_real_hombros_cm = 36  # Ajusta según datos conocidos
        factor_escala = distancia_real_hombros_cm / distancia_hombros

    # Convertir puntos 3D a cm
    puntos_3D_cm = np.array(puntos_3D) * factor_escala

    # Cálculo de medidas en cm tomando el máximo entre lado izquierdo y derecho
    largo_pierna = max(
        calcular_distancia(puntos_3D_cm[indices["cadera_izq"]], puntos_3D_cm[indices["talon_izq"]]),
        calcular_distancia(puntos_3D_cm[indices["cadera_der"]], puntos_3D_cm[indices["talon_der"]])
    )
    largo_brazo = max(
        calcular_distancia(puntos_3D_cm[indices["hombro_izq"]], puntos_3D_cm[indices["muñeca_izq"]]),
        calcular_distancia(puntos_3D_cm[indices["hombro_der"]], puntos_3D_cm[indices["muñeca_der"]])
    )
    ancho_hombros = calcular_distancia(puntos_3D_cm[indices["hombro_izq"]], puntos_3D_cm[indices["hombro_der"]])
    ancho_caderas = calcular_distancia(puntos_3D_cm[indices["cadera_izq"]], puntos_3D_cm[indices["cadera_der"]])
    altura_talon_hombro = max(
        calcular_distancia(puntos_3D_cm[indices["talon_izq"]], puntos_3D_cm[indices["hombro_izq"]]),
        calcular_distancia(puntos_3D_cm[indices["talon_der"]], puntos_3D_cm[indices["hombro_der"]])
    )

    # Calcular altura total usando la proporción de 7.5 cabezas
    altura_total = (altura_talon_hombro * 7.5) / 6.5

    tamaño_cabeza = altura_total-altura_talon_hombro

    # Guardar resultados del frame
    medidas.append([largo_pierna, largo_brazo, ancho_hombros, ancho_caderas, altura_talon_hombro, altura_total, tamaño_cabeza])

# Convertir medidas a NumPy array para promedios
medidas = np.array(medidas)
medias_finales = np.mean(medidas, axis=0)

# Guardar resultados en archivo de salida
with open(archivo_salida, "w") as f:
    f.write("Largo pierna, Largo brazo, Ancho hombros, Ancho caderas, Altura talón-hombro, Altura total, Tamaño cabeza\n")
    f.write(", ".join(map(str, medias_finales)) + "\n")

# Mostrar resultados en consola
print("\nMedidas promedio del cuerpo en cm:")
print(f"Largo pierna: {medias_finales[0]:.2f} cm")
print(f"Largo brazo: {medias_finales[1]:.2f} cm")
print(f"Ancho hombros: {medias_finales[2]:.2f} cm")
print(f"Ancho caderas: {medias_finales[3]:.2f} cm")
print(f"Altura talón-hombro: {medias_finales[4]:.2f} cm")
print(f"Altura total estimada: {medias_finales[5]:.2f} cm")
print(f"Tamaño cabeza: {medias_finales[6]:.2f} cm")

print(f"\nResultados guardados en {archivo_salida}")


import pandas as pd

# Cargar el archivo CSV generado previamente
df = pd.read_csv("ErroresSegmentos.csv")

# Promedio general de errores
promedios_generales = df[["Error Angular (°)", "Error Longitud (cm)"]].mean()
print("\n--- Promedios Generales ---")
print(promedios_generales)

# Desviaciones estándar
desviaciones = df[["Error Angular (°)", "Error Longitud (cm)"]].std()
print("\n--- Desviaciones Estándar ---")
print(desviaciones)

# Promedio por segmento
promedios_por_segmento = df.groupby("Segmento")[["Error Angular (°)", "Error Longitud (cm)"]].mean()
print("\n--- Promedio por Segmento ---")
print(promedios_por_segmento)

# Promedio por frame
promedios_por_frame = df.groupby("Frame")[["Error Angular (°)", "Error Longitud (cm)"]].mean()
print("\n--- Promedio por Frame ---")
print(promedios_por_frame)
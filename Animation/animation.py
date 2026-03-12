from vedo import Mesh, Plotter
import imageio.v2 as imageio
import glob
import time
import os


# Cargar y colorear mallas
archivos = sorted(glob.glob("salida_mallas/frame_*.obj"))
mallas = [Mesh(a).c("pink") for a in archivos]

# Crear carpeta de imágenes
os.makedirs("frames_png", exist_ok=True)

# Configurar visualizador
plotter = Plotter(size=(800, 600), interactive=False, offscreen=True)

# Mostrar el primer frame para configurar la cámara correctamente
plotter.show(mallas[0], zoom=1.0, resetcam=True, interactive=False)
plotter.clear()

# Generar imágenes
for i, malla in enumerate(mallas):
    plotter.show(malla, resetcam=False)
    filename = f"frames_png/frame_{i:04d}.png"
    plotter.screenshot(filename)
    plotter.clear()
    print(f"Guardada imagen {filename}")


imagenes = sorted(glob.glob("frames_png/*.png"))
frames = [imageio.imread(img) for img in imagenes]

# O guardar como MP4
imageio.mimsave("animacion.mp4", frames, fps=3) 
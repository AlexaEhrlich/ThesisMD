
GUÍA DE USO:
En este documento te doy una guía del uso del código que realicé para la generación de modelos SMPL con datos propios y del contenido de esta memoria.

Dataset:
Contiene un total de 12 carpetas pertenecientes a cada una de las personas grabadas para la generación del mismo.
Cada una de estas carpetas contiene en promedio un total de 136 videos nombrados de la forma: "1_1", donde el primer número se corresponde con el ejercicio y el segundo con el número de cámara que fue grabado. (Para obtener información de la numeración revisar el archivo de tesis).

Dataset Output:
Contiene un total de 12 carpetas que se corresponden con las personas grabadas y los nombres de los mismos.
Cada una de estas carpetas contiene:
	-Un total de 17 carpetas, cada una correspondiente a un ejercicio, en cada una de estas carpetas se tienen los siguientes archivos:
		-Una carpeta "frames_png" que contiene una imagen PNG de cada frame evaluado.
		-Una carpeta "salida_mallas" que contiene los modelos OBJ de los modelos SMPL generados.
		-Un video de la animación con las imágenes PNG.
		-Un archivo de coordenadas 3D calculadas.
		-Un archivo de las coordenadas 2D calculadas para cada una de las cámaras.
	-Un archivo de medidas reales tomadas a los participantes (altura y distancia entre hombros).
	-Un archivo de las medidas calculadas en el sistema.

Calibration:
Contiene una carpeta de imágenes usadas para un proceso de calibración manual, un archivo CSV que contiene una relación de puntos en la imagen (píxeles) y en el mundo real (cm) y los archivos de matriz de proyección generados con el código "MatrizProyeccion.py" en la misma carpeta.

El uso del resto del contenido se describe a continuación junto con el uso de los mismos códigos:

1. Captura de videos:
	Se realizó por medio de la implementación de loaders de YoloV8 para captura sincronizada de datos en el archivo "loaders.py". La implementación de captura se realizó en el código "loader.py".
	En este caso se utilizaron dos computadoras de forma simultánea, los archivos con número de cámara par se capturaron con una computadora y los de número de cámara impar se capturaron con otra computadora.

2. Autocalibración:
	Este proceso se realizó en tres experimentos. El archivo "ImagenesCalibracion" contiene las distintas carpetas de imágenes usadas para estos experimentos. En las carpetas de forma "calibracion1" se guardan las matrices de proyección y parámetros intrínsecos de cada cámara para cada experimento.
	El código "autocalibration.py" fue utilizado para realizar los procesos de autocalibración.

3. Triangulación:
	Posteriormente se realizó el proceso de autocalibración mediante el método de calibración iterativa (código "IterativeTriangulation.py"), guardando con este código los archivos de "coordenadas3D" y "coordenadas2D" generadas para cada persona, ejercicio y cámara según fuese el caso.
	Además, en este proceso se realizó la estimación de medidas mediante el código "Measures.py" y con el archivo "MedidasReales.txt", generando el archivo "MedidasEstimadas.txt" para cada una de las personas consideradas en el experimento.

4. Generación del modelo SMPL (archivo "generationSMPL.py"):
	En este bloque se realiza principalmente la generación del modelo SMPL de la siguiente forma:
	
	a ) Se realiza una correspondencia de articulaciones de MediaPipe y SMPL:
		MediaPipe tiene varios puntos clave (33 en total), que incluyen tanto las articulaciones principales como otros puntos adicionales (como orejas y ojos). En SMPL, las articulaciones principales incluyen hombros, caderas, codos, rodillas, tobillos, entre otros.
		La correspondencia de puntos clave está dada por:
			Hombro izquierdo (MP: 11) ↔ Hombro izquierdo (SMPL: 16)
			Hombro derecho (MP: 12) ↔ Hombro derecho (SMPL: 17)
			Codo izquierdo (MP: 13) ↔ Codo izquierdo (SMPL: 18)
			Codo derecho (MP: 14) ↔ Codo derecho (SMPL: 19)
			Muñeca izquierda (MP: 15) ↔ Muñeca izquierda (SMPL: 20)
			Muñeca derecha (MP: 16) ↔ Muñeca derecha (SMPL: 21)
			Cadera izquierda (MP: 23) ↔ Cadera izquierda (SMPL: 1)
			Cadera derecha (MP: 24) ↔ Cadera derecha (SMPL: 2)
			Rodilla izquierda (MP: 25) ↔ Rodilla izquierda (SMPL: 4)
			Rodilla derecha (MP: 26) ↔ Rodilla derecha (SMPL: 5)
			Tobillo izquierdo (MP: 27) ↔ Tobillo izquierdo (SMPL: 7)
			Tobillo derecho (MP: 28) ↔ Tobillo derecho (SMPL: 8)
	b ) Se obtienen los vectores de rotación:
		- Obtener los vectores neutros usando la pose T.
		- Obtener el vector de dirección para cada uno de los segmentos en la pose de un frame arbitrario.
		- Obtener el vector de rotación.
	c ) Generar los parámetros de forma:
		Esto se realiza una sola vez en el primer frame, ya que esta información no cambia a lo largo de la ejecución.
		Estos parámetros se generan mediante las medidas calculadas con las siguientes correspondencias con el vector de medidas guardadas anteriormente:
			𝛽_0 Altura general [5]
			𝛽_1 Grosor del torso -
			𝛽_2 Ancho de hombros [2]
			𝛽_3 Logitud de piernas [0]
			𝛽_4 Ancho cadera [3]
			𝛽_5 Tamaño pecho -
			𝛽_6 Tamaño de la cabeza [6]
			𝛽_7 Logitud de brazos [1]
			𝛽_8 Forma de las piernas -
			𝛽_9 Distribución de la grasa - 
		Nota: Si se usan más de 10 valores (hasta 300), se obtienen más detalles corporales, pero los primeros 10 ya explican el 95% de la variabilidad.

	d ) Generar la animación con SMPL: 
		Una vez obtenidos los parámetros de pose y forma, se genera la malla correspondiente con SMPL pasando los parámetros al modelo SMPL.
		Sin realizar optimización, pues genera mejores resultados.
		La malla se guarda como archivo OBJ y como imagen PNG.

	e ) Cálculo de error:
		En este código se calculan también las métricas de error, las cuales son almacenadas y guardadas en los archivos "ErroresSegmentos.csv".

5. Generación de la animación:
	Mediante el código "animation.py" se leen las imágenes PNG guardadas y se genera la animación de movimiento.

6. Análisis de métricas:
	El análisis de métricas se realiza mediante la lectura de los archivos "ErroresSegmentos.csv" y el cálculo de métricas como promedio y desviación estándar.


NOTAS AL LECTOR:

Para mayor cantidad de detalles leer individualmente cada uno de los códigos con sus respectivos comentarios.
Se recomienda trabajar la sincronización entre cámaras pares e impares, ya que fueron capturadas con distintos equipos de cómputo.
Actualmente no se recomienda el uso de los modelos SMPL almacenados como OBJ en este archivo; sin embargo, se agregan como un punto de comparativa para trabajos posteriores.
Para cualquier detalle de implementación o lógica se recomienda la lectura del archivo de tesis adjunto en esta USB.
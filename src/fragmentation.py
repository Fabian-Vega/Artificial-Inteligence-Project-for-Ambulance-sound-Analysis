from pydub import AudioSegment
import os

def cortar_mp3(ruta_mp3, duracion_segmento, ruta_salida, prefijo_salida):
    # Cargar el archivo MP3
    audio = AudioSegment.from_mp3(ruta_mp3)

    # Duración del archivo MP3 en milisegundos
    duracion_total = len(audio)

    # Duración del segmento en milisegundos
    duracion_segmento_ms = duracion_segmento * 1000

    # Inicializar variables
    inicio = 0
    fin = duracion_segmento_ms
    segmentos = []

    # Crear la carpeta de salida si no existe
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)

    # Cortar el archivo en segmentos de duración_segmento_ms
    while inicio < duracion_total:
        # Cortar el segmento
        segmento = audio[inicio:fin]

        # Agregar el segmento a la lista de segmentos
        segmentos.append(segmento)

        # Guardar el segmento en un archivo individual
        nombre_salida = f'{prefijo_salida}_{len(segmentos)}.mp3'
        segmento.export(os.path.join(ruta_salida, nombre_salida), format='mp3')

        # Actualizar los puntos de inicio y fin
        inicio += duracion_segmento_ms
        fin += duracion_segmento_ms

    return segmentos

# Ejemplo de uso
ruta_mp3 = '../data/untitled.mp3'
ruta_salida = '../unheard'
duracion_segmento = 3  # Duración del segmento en segundos
prefijo_salida = 'segmento'
segmentos = cortar_mp3(ruta_mp3, duracion_segmento, ruta_salida, prefijo_salida)

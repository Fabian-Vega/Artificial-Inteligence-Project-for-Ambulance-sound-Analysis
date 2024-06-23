from pydub import AudioSegment
import os

# Carpeta de entrada y salida
carpeta_entrada = '../data/unheard'
carpeta_salida = '../data/unheard_wav'

# Crear la carpeta de salida si no existe
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# Recorrer todos los archivos en la carpeta de entrada
for archivo in os.listdir(carpeta_entrada):
    if archivo.endswith('.mp3'):
        # Obtener la ruta completa del archivo de entrada y salida
        ruta_entrada = os.path.join(carpeta_entrada, archivo)
        nombre_salida = archivo.replace('.mp3', '.wav')
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)
        
        # Convertir el archivo .mp3 a .wav
        cancion = AudioSegment.from_mp3(ruta_entrada)
        cancion.export(ruta_salida, format='wav')
        

print('Todos los archivos .mp3 han sido convertidos a .wav.')

from moviepy.editor import *

def convertir_mp4_a_mp3(ruta_mp4, ruta_mp3):
    video = VideoFileClip(ruta_mp4)
    audio = video.audio
    audio.write_audiofile(ruta_mp3)

# Ejemplo de uso
ruta_mp4 = '../data/sonidoAmbiente.mp4'
ruta_mp3 = '../data/sonidoAmbiente.mp3'
convertir_mp4_a_mp3(ruta_mp4, ruta_mp3)
import os
from pydub import AudioSegment

def convert_mp3_to_wav(directory):
    # Verificar si la ruta del directorio existe
    if not os.path.exists(directory):
        print(f"La carpeta {directory} no existe.")
        return

    # Obtener lista de archivos .mp3 en la carpeta
    files = [f for f in os.listdir(directory) if f.endswith('.mp3')]

    for file in files:
        mp3_path = os.path.join(directory, file)
        wav_path = os.path.join(directory, os.path.splitext(file)[0] + '.wav')
        
        # Convertir el archivo .mp3 a .wav
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format='wav')
        
        # Eliminar el archivo .mp3 original
        os.remove(mp3_path)
        print(f"Convertido y eliminado: {file}")

if __name__ == "__main__":
    # Definir la ruta del directorio que contiene los archivos .mp3
    directory = './data/sirens'

    convert_mp3_to_wav(directory)
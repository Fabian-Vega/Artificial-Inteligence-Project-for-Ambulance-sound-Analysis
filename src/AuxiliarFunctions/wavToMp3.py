import os
from pydub import AudioSegment

def convert_wav_to_mp3(directory):
    # Verificar si la ruta del directorio existe
    if not os.path.exists(directory):
        print(f"La carpeta {directory} no existe.")
        return

    # Obtener lista de archivos .wav en la carpeta
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]

    for file in files:
        wav_path = os.path.join(directory, file)
        mp3_path = os.path.join(directory, os.path.splitext(file)[0] + '.mp3')
        
        # Convertir el archivo .wav a .mp3
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format='mp3')
        
        # Eliminar el archivo .wav original
        os.remove(wav_path)
        print(f"Convertido y eliminado: {file}")

if __name__ == "__main__":
    # Definir la ruta del directorio que contiene los archivos .wav
    directory = './data/TestAudiosWithAmbulance'

    convert_wav_to_mp3(directory)

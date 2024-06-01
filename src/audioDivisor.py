import wave
import os

def split_wav(input_file, output_folder, segment_length=3):
    # Asegurarse de que el directorio de salida existe
    os.makedirs(output_folder, exist_ok=True)
    
    with wave.open(input_file, 'rb') as wf:
        # Obtener parámetros del archivo de audio
        params = wf.getparams()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        sample_width = wf.getsampwidth()
        n_channels = wf.getnchannels()
        
        # Duración total del archivo de audio en segundos
        duration = nframes / float(framerate)
        
        # Dividir el audio en segmentos de la longitud especificada
        for i in range(0, int(duration), segment_length):
            wf.setpos(i * framerate)
            frames = wf.readframes(segment_length * framerate)
            
            segment_name = f"{output_folder}/segment_{i // segment_length + 1}.wav"
            with wave.open(segment_name, 'wb') as segment_wf:
                segment_wf.setnchannels(n_channels)
                segment_wf.setsampwidth(sample_width)
                segment_wf.setframerate(framerate)
                segment_wf.writeframes(frames)
            
            print(f"Archivo guardado: {segment_name}")

# Uso de la función
input_audio_file = "./audio/ambulance_dataset.wav"
output_folder = "./data"

split_wav(input_audio_file, output_folder)

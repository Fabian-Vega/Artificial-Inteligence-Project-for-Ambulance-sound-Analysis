import os
import re

def rename_segments(folder):
    # Expresión regular para capturar el nombre del archivo y el número
    pattern = re.compile(r"segment_(\d+)\.mp3")
    
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            number = match.group(1)
            new_number = number.zfill(3)  # Agregar ceros a la izquierda para que tenga al menos 3 dígitos
            new_filename = f"segment_{new_number}.mp3"
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_filename)
            
            # Renombrar el archivo
            os.rename(old_path, new_path)
            print(f"Renombrado: {filename} -> {new_filename}")

# Uso de la función
folder = "./data"  # Cambia esto a la ruta de tu carpeta de archivos MP3

rename_segments(folder)

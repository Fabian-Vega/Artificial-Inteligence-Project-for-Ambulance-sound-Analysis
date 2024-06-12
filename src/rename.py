import os
import re

def rename_mp3_files(input_path):
    # Verifica si la ruta de entrada es un directorio válido
    if not os.path.isdir(input_path):
        print(f"La ruta {input_path} no es un directorio válido.")
        return
    
    # Obtén una lista de todos los archivos en el directorio especificado
    files = os.listdir(input_path)
    
    # Patrón regex para coincidir con archivos de la forma 'segmento_n.mp3' donde n es de 1 a 3 dígitos
    pattern = re.compile(r'^segmento_(\d{1,3})\.mp3$')
    
    for file in files:
        match = pattern.match(file)
        if match:
            # Extrae el número del nombre del archivo
            number = match.group(1)
            # Formatea el número con ceros a la izquierda para que tenga tres dígitos
            new_number = number.zfill(3)
            # Crea el nuevo nombre del archivo
            new_name = f"segment_{new_number}.mp3"
            # Construye las rutas completas de los archivos
            old_file_path = os.path.join(input_path, file)
            new_file_path = os.path.join(input_path, new_name)
            # Renombra el archivo
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{file}' to '{new_name}'")

if __name__ == "__main__":
    input_path = '../data/unheard'
    rename_mp3_files(input_path)

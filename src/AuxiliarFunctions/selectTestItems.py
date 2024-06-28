import os
import shutil
import random

def move_random_files(input_path, output_path, x):
    # Verificar si las rutas de entrada y salida existen
    if not os.path.exists(input_path):
        print(f"La carpeta de entrada {input_path} no existe.")
        return
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Se ha creado la carpeta de salida {output_path}.")

    # Obtener lista de archivos en la carpeta de entrada
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

    # Verificar si hay suficientes archivos para mover
    if len(files) < x:
        print(f"No hay suficientes archivos en la carpeta de entrada para mover {x} archivos.")
        return

    # Seleccionar aleatoriamente x archivos
    selected_files = random.sample(files, x)

    # Mover los archivos seleccionados a la carpeta de salida
    for file in selected_files:
        src = os.path.join(input_path, file)
        dst = os.path.join(output_path, file)
        shutil.move(src, dst)
        print(f"Movido: {file}")

if __name__ == "__main__":
    # Definir las rutas de entrada y salida y la cantidad de archivos a mover
    input_path = './data/unheard_wav'
    output_path = './data/TestAudiosWithoutAmbulance'
    x = 50

    move_random_files(input_path, output_path, x)
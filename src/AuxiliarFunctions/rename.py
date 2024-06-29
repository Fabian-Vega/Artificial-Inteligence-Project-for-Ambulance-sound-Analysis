import os

def rename_files_with_pattern(path, extension, pattern):
    # Verificar si la ruta del directorio existe
    if not os.path.exists(path):
        print(f"La carpeta {path} no existe.")
        return

    # Obtener lista de archivos con la extensión especificada en la carpeta
    files = [f for f in os.listdir(path) if f.endswith(extension)]

    # Ordenar los archivos para mantener un orden consistente
    files.sort()

    # Renombrar los archivos
    for i, file in enumerate(files):
        new_name = f"{pattern}{str(i + 1).zfill(2)}{extension}"
        old_path = os.path.join(path, file)
        new_path = os.path.join(path, new_name)
        os.rename(old_path, new_path)
        print(f"Renombrado: {file} -> {new_name}")

if __name__ == "__main__":
    # Definir los parámetros
    path = './data/sirens_validation'
    extension = '.wav'
    pattern = 'sirens_validation_'

    rename_files_with_pattern(path, extension, pattern)
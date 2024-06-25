import os

def rename_mp3_files_in_current_folder():
    # Get the current working directory
    folder_path = os.getcwd()
    
    # Get list of all files in the directory
    files = os.listdir(folder_path)
    
    # Filter only .mp3 files
    mp3_files = [file for file in files if file.endswith('.mp3')]
    
    # Sort files to ensure consistent numbering
    mp3_files.sort()
    
    # Rename each mp3 file
    for index, mp3_file in enumerate(mp3_files):
        new_name = f"testAudioNOAmbulance_{index}.mp3"
        old_file_path = os.path.join(folder_path, mp3_file)
        new_file_path = os.path.join(folder_path, new_name)
        
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_file_path} -> {new_file_path}")

# Example usage
rename_mp3_files_in_current_folder()

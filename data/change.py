import os

def rename_files_in_directory(directory):
    # List all files in the directory
    for filename in os.listdir(directory):
        # Ignore directories, focus on files only
        if os.path.isfile(os.path.join(directory, filename)):
            # Create the new filename by adding '1' at the beginning
            new_filename = f"2{filename}"
            
            # Construct full old and new file paths
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed '{filename}' to '{new_filename}'")

# Example usage:
# Specify the directory containing the files you want to rename
directory_path = os.path.join('data', 'SirensMove')

# Call the function to rename files
rename_files_in_directory(directory_path)

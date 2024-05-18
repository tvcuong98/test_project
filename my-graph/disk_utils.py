import os
import glob
def delete_all_files_in_folder(folder_path):
    # Get a list of all files in the folder
    files = glob.glob(os.path.join(folder_path, '*'))
    
    # Iterate over the list of files and remove each one
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
import os
import glob

def change_label_class(folder_path: str) -> None:
    """
    Read all .txt files under the specified folder and change the first letter
    of every line from '4' to '2'.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing .txt files to be modified.

    """
    # Construct the path pattern to match all .txt files within the folder
    path_pattern = os.path.join(folder_path, "*/*.txt")
    
    # Find all .txt files in the specified folder
    txt_files = glob.glob(path_pattern)
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            lines = file.readlines()
        
        # Modify the first character of each line from '4' to '2' if necessary
        modified_lines = [line.replace('2', '3', 1) if line.startswith('2') else line for line in lines]
        
        # Write the modified lines back to the file
        with open(txt_file, 'w') as file:
            file.writelines(modified_lines)

# Example usage
folder_path = 'jpgs'  # Adjust this path as necessary
change_label_class(folder_path)

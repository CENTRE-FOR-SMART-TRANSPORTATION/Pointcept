import os
import glob

def rename_files(base_dir):
    # Use glob to find all txt files in 'Annotations' subfolders
    for txt_file in glob.glob(os.path.join(base_dir, '**', 'Annotations', '*.txt'), recursive=True):
        # Check if 'highway-guardrails' is in the filename
        if 'highway-guardrail' in os.path.basename(txt_file):
            # Construct the new filename
            new_file_name = os.path.basename(txt_file).replace('highway-guardrail', 'highway-guardrails')
            new_file_path = os.path.join(os.path.dirname(txt_file), new_file_name)
            
            # Rename the file
            os.rename(txt_file, new_file_path)
            print(f"Renamed: {txt_file} to {new_file_path}")

# Replace 'Hesham_full_35' with the path to your base directory
base_directory = 'Hesham_full_35'
rename_files(base_directory)


import os
import glob

def remove_null_chars_in_txt_files(base_dir):
    # Use glob to find all txt files in 'Annotations' subfolders
    for txt_file in glob.glob(os.path.join(base_dir, '**', 'Annotations', '*.txt'), recursive=True):
        with open(txt_file, 'r') as file:
            content = file.read()
        cleaned_content = content.replace('\0', '')
        with open(txt_file, 'w') as file:
            file.write(cleaned_content)
        print(f"Null characters removed from {txt_file}")

# Replace 'Hesham_full_35' with the path to your base directory
base_directory = '../Hesham_files/Val/'
remove_null_chars_in_txt_files(base_directory)


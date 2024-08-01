import os

def process_folder(folder_path, target_columns):
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        process_single_file(file_path, target_columns)

def process_single_file(file_path, target_columns):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = ' '.join(lines[i].split()[-target_columns:]) + '\n'

    with open(file_path, 'w') as file:
        file.writelines(lines)

target_columns = 136

folder_path = ''

process_folder(folder_path, target_columns)

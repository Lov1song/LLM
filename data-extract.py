import os
import gzip
from tqdm import tqdm

def gz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".gz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

folder_path = "D:/Ay1/fcc-gpt-course/automated_evaluation/automated_evaluation_up/spinnerchief/corpus/paragraphs"
output_file_train = "tarin_split.txt"
output_file_val = "val_split.txt"
vocab_file = "vocab.txt"
# split_files = int(input("how many files would you like to split this into?"))

files = gz_files_in_dir(folder_path)
total_files = len(files)
split_index = int(total_files * 0.9)
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()

with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with gzip.open(file_path, "rt", encoding="utf-8",errors="ignore") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with gzip.open(file_path, "rt", encoding="utf-8",errors="ignore") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

with open(vocab_file,"w",encoding="utf-8",errors="ignore") as vfile:
    for char in vocab:
        vfile.write(char + '\n')
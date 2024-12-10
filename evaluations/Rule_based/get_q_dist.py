import os

if os.system("git --version") == 0:  
    os.system("git clone https://github.com/nschaetti/SFGram-dataset.git")
else:
    print("Git is not installed")
import json
import numpy as np
import pandas as pd
import torch
import requests
import random
from collections import Counter

def calculate_word_frequencies(text):
    words = text.split()  
    word_counts = Counter(words)  
    total_count = sum(word_counts.values())  
    word_probabilities = {word: count / total_count for word, count in word_counts.items()}  
    return word_probabilities

#os.chdir('/content/SFGram-dataset/book-contents')
base_url = 'https://raw.githubusercontent.com/nschaetti/SFGram-dataset/master/book-contents/'
file_prefix = 'book'
file_extension = '.txt'

file_numbers = range(1, 1004) 

all_paragraphs = []

for file_number in file_numbers:
    file_name = f"{file_prefix}{file_number:05}{file_extension}"
    file_url = os.path.join(base_url, file_name)

    try:
        response = requests.get(file_url)
        if response.status_code != 200:
            print(f"There is something wrong with the file{file_name}，skip.")
            continue

        paragraphs = []
        current_paragraph = []

        for line in response.text.split('\n'):
            if line.strip():  
                current_paragraph.append(line.strip())
            elif current_paragraph:  
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

    
        num_paragraphs_to_sample = min(5, len(paragraphs))
        random_paragraphs = random.sample(paragraphs, num_paragraphs_to_sample)

        all_paragraphs.extend(random_paragraphs)

    except Exception as e:
        print(f"File {file_name} has bug:{e}")

final_text = " ".join(all_paragraphs)

reference_sentence = final_text
q_dist = calculate_word_frequencies(reference_sentence)
file_name = "./Evaluation/Rule_based/q_dist.json"
with open(file_name, "w", encoding="utf-8") as file:
    json.dump(q_dist, file, indent=4, ensure_ascii=False)

print(f"Word frequencies have been saved as JSON to {file_name}")
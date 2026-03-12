import os
import urllib.request
import zipfile

### Stage 1: Preprocessing

# Downloads dataset on first run
# Just opens if already downloaded
def load_text8(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "text8")
    
    if not os.path.exists(filepath):
        zip_path = os.path.join(data_dir, "text8.zip")
        
        if not os.path.exists(zip_path):
            print("Downloading text8 dataset...")
            url = "http://mattmahoney.net/dc/text8.zip"
            urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)
    
    with open(filepath, "r") as f:
        return f.read().split()
    
import numpy as np
from collections import Counter

words = load_text8()

min_count = 5
word_counts = Counter(words)

# Build vocabulary only from frequent words
word_to_idx = {}
idx_to_word = {}
idx = 0
for word, count in word_counts.items():
    if count >= min_count:
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        idx += 1

corpus = [word_to_idx[w] for w in words if w in word_to_idx]
corpus = np.array(corpus, dtype=np.int32)

print(f"Vocabulary size: {len(word_to_idx)}")
print(f"Corpus length: {len(corpus)}")

### Stage 2: Initialize Embeddings
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

def build_vocab(words, min_count = 5):

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

    return word_to_idx, idx_to_word, word_counts

def build_corpus(words, word_to_idx):
    corpus = [word_to_idx[w] for w in words if w in word_to_idx]
    return np.array(corpus, dtype=np.int32)


### Stage 2: Subsample Vocabulary (According to Mikolov)

def subsample_corpus(corpus, word_counts, t=1e-5):
    total_words = sum(word_counts.values())
    freqs = {idx: count / total_words for idx, count in word_counts.items()}

    # P_keep(w) = sqrt(t / f(w))  (clamped to max 1.0)
    keep_probs = {idx: min(1.0, np.sqrt(t / freqs[idx])) for idx in freqs}
 
    # Generate uniform random numbers for the entire corpus at once
    rand_vals = np.random.random(len(corpus))
 
    # Keep word if random value < keep probability
    mask = np.array([rand_vals[i] < keep_probs[corpus[i]] for i in range(len(corpus))])
    return corpus[mask]


### Stage 3: Negative Sampling Distribution (For Skip-Gram)

def build_noise_distribuiton(word_counts, vocab_size):
    freqs = np.zeros(vocab_size)
    for idx, count in word_counts.items():
        freqs[idx] = count

    noise_dist = freqs ** 0.75
    noise_dist = noise_dist / noise_dist.sum()
    return noise_dist


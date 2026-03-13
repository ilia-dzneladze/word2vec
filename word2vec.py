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


### Stage 4: Initialize Embedding Vectors with Small Random Values

def initialize_embeddings(vocab_size, embedding_dim):
    U = np.random.randn(vocab_size, embedding_dim) * 0.01
    V = np.random.randn(vocab_size, embedding_dim) * 0.01
    return U, V


### Stage 5: Forward Pass and Loss

# Normalize neg-pos to 0-1
def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))

# J = -log(pos_score) - sum(log(1 - neg_scores))
def compute_loss(pos_score, neg_scores):
    loss = -np.log(pos_score + 1e-10) - np.sum(np.log(1 - neg_scores + 1e-10))
    return loss

## Stage 6: Training: Gradients and Parameter Updates (The real meat of the script)

def train_pair(w_c, w_o, neg_indices, U, V, lr):
    # Forward Pass
    u_c = U[w_c]
    v_pos = V[w_o]
    v_negs = V[neg_indices]

    # Compute Dot Products
    pos_dot = v_pos @ u_c
    neg_dots = v_negs @ u_c

    # Pass Through Sigmoid
    pos_score = sigmoid(pos_dot)
    neg_scores = sigmoid(neg_dots)

    # Compute Loss
    loss = compute_loss(pos_score, neg_scores)

    # Compute Gradients
    old_u_c = u_c.copy()
    grad_v_pos = (pos_score - 1) * old_u_c
    grad_v_negs = neg_scores.reshape(-1, 1) * old_u_c
    grad_u = (pos_score - 1) * v_pos + (neg_scores.reshape(-1, 1) * v_negs).sum(axis=0)

    # Apply updates (theta <- sum_i(sigma_i * v_wi))
    U[w_c]        -= lr * grad_u
    V[w_o]         -= lr * grad_v_pos
    V[neg_indices] -= lr * grad_v_negs

    return loss
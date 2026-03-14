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

def subsample_corpus(corpus, word_counts, word_to_idx, vocab_size, t=1e-5):
    total_words = sum(word_counts.values())

    keep_probs = np.ones(vocab_size)
    for word, count in word_counts.items():
        if word in word_to_idx:
            freq = count / total_words
            keep_probs[word_to_idx[word]] = min(1.0, np.sqrt(t / freq))

    rand_vals = np.random.random(len(corpus))
    mask = rand_vals < keep_probs[corpus]
    return corpus[mask]


### Stage 3: Negative Sampling Distribution (For Skip-Gram)

def build_noise_distribution(word_counts, word_to_idx, vocab_size):
    freqs = np.zeros(vocab_size)
    for word, count in word_counts.items():
        if word in word_to_idx:
            freqs[word_to_idx[word]] = count

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

def generate_pairs(corpus, start, end, window):
    centers = []
    contexts = []
    corpus_len = len(corpus)

    for t in range(start, min(end, corpus_len)):
        w_c = corpus[t]
        actual_window = np.random.randint(1, window + 1)

        j_start = max(0, t - actual_window)
        j_end = min(corpus_len, t + actual_window + 1)

        for j in range(j_start, j_end):
            if j == t:
                continue
            centers.append(w_c)
            contexts.append(corpus[j])

    return np.array(centers, dtype=np.int32), np.array(contexts, dtype=np.int32)

def train_batch(center_ids, context_ids, neg_ids, U, V, lr):
    B = len(center_ids)

    # Forward pass
    u_centers = U[center_ids]
    v_contexts = V[context_ids]
    v_negs = V[neg_ids]

    pos_dots = np.sum(u_centers * v_contexts, axis=1)
    neg_dots = np.sum(u_centers[:, np.newaxis, :] * v_negs, axis=2)

    pos_scores = sigmoid(pos_dots)
    neg_scores = sigmoid(neg_dots)

    # Loss
    loss = -np.sum(np.log(pos_scores + 1e-10)) \
           - np.sum(np.log(1 - neg_scores + 1e-10))

    # Gradients (same formulas, just batched)
    old_u_centers = u_centers.copy()
    pos_coeffs = (pos_scores - 1).reshape(-1, 1)
    neg_coeffs = neg_scores[:, :, np.newaxis]

    grad_v_pos = pos_coeffs * old_u_centers
    grad_v_negs = neg_coeffs * old_u_centers[:, np.newaxis, :]
    grad_u = pos_coeffs * v_contexts + np.sum(neg_coeffs * v_negs, axis=1)

    # Updates
    np.add.at(U, center_ids, -lr * grad_u)
    np.add.at(V, context_ids, -lr * grad_v_pos)

    neg_flat = neg_ids.reshape(-1)
    grad_v_negs_flat = grad_v_negs.reshape(-1, grad_v_negs.shape[2])
    np.add.at(V, neg_flat, -lr * grad_v_negs_flat)

    return loss

### Stage 7: Training Loop

import time

def train(corpus, vocab_size, noise_dist, embedding_dim=100, window=5,
          num_neg=5, initial_lr=0.025, min_lr=0.0001, epochs=1,
          batch_size=512, chunk_size=100000):

    lr = 0
    U, V = initialize_embeddings(vocab_size, embedding_dim)
    total_words = len(corpus) * epochs
    words_processed = 0
    running_loss = 0.0
    running_pairs = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_start = time.time()

        for chunk_start in range(0, len(corpus), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(corpus))

            centers, contexts = generate_pairs(corpus, chunk_start, chunk_end, window)
            if len(centers) == 0:
                continue

            all_negs = np.random.choice(
                vocab_size, size=(len(centers), num_neg), replace=True, p=noise_dist
            )

            shuffle_idx = np.random.permutation(len(centers))
            centers = centers[shuffle_idx]
            contexts = contexts[shuffle_idx]
            all_negs = all_negs[shuffle_idx]

            for b_start in range(0, len(centers), batch_size):
                b_end = min(b_start + batch_size, len(centers))
                lr = initial_lr - (initial_lr - min_lr) * (words_processed / total_words)

                loss = train_batch(
                    centers[b_start:b_end],
                    contexts[b_start:b_end],
                    all_negs[b_start:b_end],
                    U, V, lr
                )
                running_loss += loss
                running_pairs += (b_end - b_start)

            words_processed += (chunk_end - chunk_start)

            elapsed = time.time() - epoch_start
            wps = words_processed / elapsed if elapsed > 0 else 0
            avg_loss = running_loss / running_pairs if running_pairs > 0 else 0
            print(f"  [{chunk_end:>10}/{len(corpus)}] "
                  f"loss: {avg_loss:.4f}  "
                  f"lr: {lr:.6f}  "
                  f"words/sec: {wps:.0f}")
            running_loss = 0.0
            running_pairs = 0

        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch + 1} completed in {epoch_time:.1f}s")

    return U, V


### Stage 8: Evaulation techniques (Cosine Wimilarity and Analogy)

def most_similar(word, U, word_to_idx, idx_to_word, top_n=10):
    if word not in word_to_idx:
        print(f"'{word}' not in vocabulary")
        return
 
    idx = word_to_idx[word]
    word_vec = U[idx]
 
    # Cosine similarity against all words:
    dots = U @ word_vec                        # a^T b for all words
    norms = np.linalg.norm(U, axis=1)          # ||a|| for all words
    word_norm = np.linalg.norm(word_vec)        # ||b||
 
    similarities = dots / (norms * word_norm + 1e-10)
 
    # Get top indices (excluding the word itself)
    top_indices = np.argsort(similarities)[::-1][1:top_n + 1]
 
    print(f"\nMost similar to '{word}':")
    for i in top_indices:
        print(f"  {idx_to_word[i]:15s} {similarities[i]:.4f}")

def analogy(a, b, c, U, word_to_idx, idx_to_word, top_n=5):
    for word in [a, b, c]:
        if word not in word_to_idx:
            print(f"'{word}' not in vocabulary")
            return
 
    vec = U[word_to_idx[b]] - U[word_to_idx[a]] + U[word_to_idx[c]]
 
    # Cosine similarity of result vector against all words
    dots = U @ vec
    norms = np.linalg.norm(U, axis=1)
    vec_norm = np.linalg.norm(vec)
    similarities = dots / (norms * vec_norm + 1e-10)
 
    # Exclude input words
    exclude = {word_to_idx[a], word_to_idx[b], word_to_idx[c]}
    top_indices = np.argsort(similarities)[::-1]
    results = [i for i in top_indices if i not in exclude][:top_n]
 
    print(f"\n'{a}' is to '{b}' as '{c}' is to ...?")
    for i in results:
        print(f"  {idx_to_word[i]:15s} {similarities[i]:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Word2Vec Skip-Gram with Negative Sampling (Pure NumPy)")
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--num_neg", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--initial_lr", type=float, default=0.025)
    parser.add_argument("--min_lr", type=float, default=0.0001)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=100000)
    parser.add_argument("--subsample_t", type=float, default=1e-5)
    args = parser.parse_args()

    print("Hyperparameters:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # Stage 1: Load and preprocess
    print("\nLoading text8 dataset...")
    words = load_text8()
    print(f"Raw corpus: {len(words)} words")
 
    word_to_idx, idx_to_word, word_counts = build_vocab(words, min_count=args.min_count)
    vocab_size = len(word_to_idx)
    print(f"Vocabulary size: {vocab_size}")
 
    corpus = build_corpus(words, word_to_idx)
    print(f"Corpus length after filtering: {len(corpus)}")
 
    # Stage 2: Subsample frequent words
    corpus = subsample_corpus(corpus, word_counts, word_to_idx, vocab_size, t=args.subsample_t)
    print(f"Corpus length after subsampling: {len(corpus)}")
 
    # Stage 3: Build noise distribution 
    noise_dist = build_noise_distribution(word_counts, word_to_idx, vocab_size)
 
    # Stages 4-7: Train
    print("\nStarting training...")
    U, V = train(
        corpus=corpus,
        vocab_size=vocab_size,
        noise_dist=noise_dist,
        embedding_dim=args.embedding_dim,
        window=args.window,
        num_neg=args.num_neg,
        initial_lr=args.initial_lr,
        min_lr=args.min_lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    # Stage 8: Evaluation
    print("EVALUATION:")
 
    test_words = ["king", "queen", "computer", "france", "dog"]
    for word in test_words:
        most_similar(word, U, word_to_idx, idx_to_word)
 
    analogy("man", "king", "woman", U, word_to_idx, idx_to_word)
    analogy("france", "paris", "germany", U, word_to_idx, idx_to_word)

    print("Test it yourself (Type 'end' to stop):")
    while (word := input("\n> ")) not in ("end", ""):
        most_similar(word, U, word_to_idx, idx_to_word)
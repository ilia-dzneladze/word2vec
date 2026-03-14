# Word2Vec: Skip-Gram with Negative Sampling (Pure NumPy)

A from-scratch implementation of Word2Vec skip-gram with negative sampling, built entirely in NumPy with no ML frameworks. Trained on the text8 dataset (first 100MB of cleaned Wikipedia).

## Design Decisions

**Why Skip-Gram over CBOW?** CBOW averages context vectors into one prediction per window, so rare words get few, blurry updates. Skip-Gram generates a separate training pair for each context word, giving rare words multiple gradient signals per occurrence. Mikolov et al. showed it consistently outperforms CBOW on analogy and similarity benchmarks.

**Why Negative Sampling over full Softmax?** Full softmax requires dot products with every word in the vocabulary per training step: O(|V|) per pair. Negative sampling replaces this with a binary classification task (real pair vs. k random fake pairs), reducing it to O(k) where k is typically 5-15.

**Single file.** The entire implementation lives in `word2vec.py`. Every stage of the algorithm can be read linearly, top to bottom.

**Both single-pair and batched training.** `train_pair` implements straightforward one-pair-at-a-time logic for clarity. `train_batch` implements the same math vectorized across batches for performance (~50-100x faster). The training loop uses `train_batch`.

**np.add.at for updates.** When multiple pairs in a batch share the same word index, standard indexing (`U[ids] -= grads`) only applies the last update. `np.add.at` accumulates all gradient contributions correctly.

**No framework dependency.** Pure NumPy means every operation (forward pass, loss, gradients, updates) is explicit and inspectable.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/word2vec.git
cd word2vec
python word2vec.py
```

The text8 dataset (~31MB compressed) downloads automatically on first run.

**Dependencies:** Python 3.8+ and NumPy only.

## Configuration

Every hyperparameter is configurable via command line:

```bash
python word2vec.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--embedding_dim` | 100 | Dimensionality of word vectors |
| `--window` | 5 | Max context window size |
| `--num_neg` | 5 | Negative samples per training pair |
| `--epochs` | 1 | Passes over the corpus |
| `--initial_lr` | 0.025 | Starting learning rate |
| `--min_lr` | 0.0001 | Minimum learning rate (linear decay) |
| `--min_count` | 5 | Minimum word frequency for vocabulary |
| `--batch_size` | 512 | Pairs per batch |
| `--chunk_size` | 100000 | Corpus chunk size for pair generation |
| `--subsample_t` | 1e-5 | Subsampling threshold for frequent words |

### Example Configurations

```bash
# Default: ~13 min on CPU, decent results
python word2vec.py

# Higher quality
python word2vec.py --epochs 5

# Powerful machine
python word2vec.py --embedding_dim 300 --batch_size 2048 --epochs 3

# Quick test
python word2vec.py --embedding_dim 50 --epochs 1
```

## Pipeline

1. **Preprocessing.** Download text8, build vocabulary (drop words below `min_count`), convert to integer indices.
2. **Subsampling.** Randomly discard frequent words (keep probability: sqrt(t / freq)). Speeds up training and improves rare word embeddings.
3. **Noise Distribution.** Build negative sampling distribution: P(w) ~ count(w)^0.75, a smoothed unigram that balances between raw frequency and uniform.
4. **Initialization.** Two matrices U (center) and V (context), shape (vocab_size, embedding_dim), initialized with small random values.
5. **Forward Pass.** Compute sigmoid of dot products between center and context/negative vectors.
6. **Gradients.** Derived via chain rule: real context vectors are pulled toward center, negatives are pushed away.
7. **Training Loop.** Process corpus in chunks, generate pairs, train in batches with linear learning rate decay.
8. **Evaluation.** Cosine similarity for nearest neighbors, vector arithmetic for analogies.

## Sample Results (5 epochs, default settings, ~55 min on CPU)
```
Most similar to 'king':       conqueror, highness, pretender
Most similar to 'queen':      elizabeth, monarch, princess, consort
Most similar to 'computer':   computers, hardware, microcomputer, computing
Most similar to 'france':     belgium, netherlands, vichy
Most similar to 'dog':        hound, hounds, dogs, cat

'france' is to 'paris' as 'germany' is to ...?
  munich, berlin, leipzig, dresden
```

## Dataset

[text8](http://mattmahoney.net/dc/text8.zip): first 100M characters of cleaned English Wikipedia. ~17M tokens, ~71K vocabulary after filtering. Downloads automatically on first run.

## Project Structure

```
word2vec/
├── word2vec.py       # Complete implementation
├── README.md
├── .gitignore
└── data/             # Auto-created, gitignored
```
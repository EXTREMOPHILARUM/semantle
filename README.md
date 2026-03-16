# Semantle Solver

Solves [Semantle](https://semantle.com/) / [Pimantle](https://semantle.pimanrul.es/) in ~4 guesses using information-theoretic word elimination.

## How it works

1. Uses the exact Pimantle/Semantle vocabulary (106,981 words) with Google News word2vec 300d embeddings
2. Each guess eliminates candidates whose expected similarity to the guess doesn't match the observed score
3. Next guess is chosen to maximize expected candidate elimination (minimizes the largest remaining bucket)
4. Converges to the answer in 3-4 guesses on average

### Benchmark (30 random words)

| Metric | Value |
|--------|-------|
| Mean | 3.9 guesses |
| Median | 4 |
| ≤4 guesses | 97% |
| Min / Max | 3 / 5 |

## Setup

```bash
uv sync
```

First run downloads the word2vec model (~1.7GB) and builds a vector cache (~14s). Subsequent runs load from cache in ~0.3s.

## Usage

**Interactive mode** — solver suggests guesses, you enter scores from Semantle:

```bash
uv run python3 solver.py
```

**Simulate** against a specific word:

```bash
uv run python3 solver.py --sim --target crab
```

**Simulate** against a random word:

```bash
uv run python3 solver.py --sim
```

**Benchmark** over N random words:

```bash
uv run python3 solver.py --bench 50
```

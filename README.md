# Semantle Solver

Solves [Semantle](https://semantle.com/) in ~3-4 guesses using information-theoretic word elimination.

## How it works

1. Uses Google News word2vec embeddings (300d, 50k vocabulary)
2. Each guess is chosen to maximally split the remaining candidate set
3. After receiving a similarity score, eliminates all candidates whose expected similarity doesn't match
4. Converges to the answer in 3-4 guesses on average

## Setup

```bash
uv sync
```

First run downloads the word2vec model (~1.7GB, cached after that).

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

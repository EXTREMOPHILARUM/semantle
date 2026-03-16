"""
Semantle/Pimantle Solver — minimize guesses using information-theoretic word elimination.

Uses the exact same word list and word2vec embeddings as Pimantle (106,981 words).
Each guess + score eliminates candidates whose similarity to the guess doesn't match.
Next guess is chosen to maximize expected elimination.

Usage:
    uv run python3 solver.py                          # interactive mode
    uv run python3 solver.py --sim --target crab      # simulate against a word
    uv run python3 solver.py --sim                    # simulate against random word
    uv run python3 solver.py --bench N                # benchmark over N random words
"""

import json
import numpy as np
import os
import struct
import sys
import time
from pathlib import Path
from sklearn.preprocessing import normalize

DATA_DIR = Path(__file__).parent
WORD_LIST_PATH = DATA_DIR / "word_list.json"
VECTORS_CACHE = DATA_DIR / "vectors.npz"

# ---------------------------------------------------------------------------
# Load word list + vectors
# ---------------------------------------------------------------------------

_words = None
_vectors = None
_w2i = None


def load_word_list():
    """Load Pimantle's 106k word list."""
    global _words, _w2i
    if _words is not None:
        return _words, _w2i
    with open(WORD_LIST_PATH) as f:
        wl = json.load(f)
    _words = [entry[0] for entry in wl]
    _w2i = {w: i for i, w in enumerate(_words)}
    return _words, _w2i


def load_vectors():
    """Load word2vec vectors for the Pimantle word list. Caches to disk."""
    global _vectors, _words, _w2i
    words, w2i = load_word_list()

    if _vectors is not None:
        return words, _vectors, w2i

    if VECTORS_CACHE.exists():
        print("Loading cached vectors...")
        t0 = time.time()
        data = np.load(VECTORS_CACHE)
        _vectors = data["vectors"]
        print(f"Loaded {_vectors.shape[0]} vectors in {time.time()-t0:.1f}s")
        return words, _vectors, w2i

    print("Building vector cache from word2vec (one-time, ~30s)...")
    t0 = time.time()
    import gensim.downloader as api
    model = api.load("word2vec-google-news-300")

    dim = 300
    vecs = np.zeros((len(words), dim), dtype=np.float32)
    found = 0
    for i, w in enumerate(words):
        if w in model:
            vecs[i] = model[w]
            found += 1

    _vectors = normalize(vecs, axis=1).astype(np.float32)
    np.savez_compressed(VECTORS_CACHE, vectors=_vectors)
    print(f"Cached {found}/{len(words)} vectors in {time.time()-t0:.1f}s")
    return words, _vectors, w2i


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SemantleSolver:
    def __init__(self, tolerance=0.15):
        self.words, self.vectors, self.w2i = load_vectors()
        self.tolerance = tolerance
        self.candidates = np.arange(len(self.words))
        self.guesses = []

    def reset(self):
        self.candidates = np.arange(len(self.words))
        self.guesses = []

    def remaining(self):
        return len(self.candidates)

    def candidate_words(self):
        return [self.words[i] for i in self.candidates[:20]]

    def _pick_best_guess(self):
        n_cand = len(self.candidates)
        if n_cand <= 2:
            return self.candidates[0]

        # Sample candidates for evaluation
        max_eval = 400
        max_check = 600

        if n_cand > max_eval:
            eval_idx = np.random.choice(self.candidates, max_eval, replace=False)
        else:
            eval_idx = self.candidates

        if n_cand > max_check:
            check_idx = np.random.choice(self.candidates, max_check, replace=False)
        else:
            check_idx = self.candidates

        check_vecs = self.vectors[check_idx]
        n_check = len(check_idx)

        best_guess = eval_idx[0]
        best_expected_remaining = n_check  # worst case: no elimination

        tol = self.tolerance
        for gi in eval_idx:
            # Similarities from this guess to all check candidates
            sims = (check_vecs @ self.vectors[gi]) * 100.0
            # Bin by tolerance
            bins = np.round(sims / tol).astype(np.int32)
            _, counts = np.unique(bins, return_counts=True)
            # Expected remaining = E[|bucket|] = sum(count^2) / total
            expected_remaining = np.sum(counts * counts) / n_check

            if expected_remaining < best_expected_remaining:
                best_expected_remaining = expected_remaining
                best_guess = gi

        return best_guess

    def suggest(self):
        if len(self.candidates) == 0:
            return None

        if len(self.guesses) == 0:
            for starter in ["game", "place", "thing", "water", "light"]:
                if starter in self.w2i:
                    return starter

        idx = self._pick_best_guess()
        return self.words[idx]

    def update(self, word, score):
        self.guesses.append((word, score))

        if word not in self.w2i:
            print(f"  Warning: '{word}' not in vocabulary, can't filter")
            return

        gvec = self.vectors[self.w2i[word]]
        cand_vecs = self.vectors[self.candidates]
        expected_scores = (cand_vecs @ gvec) * 100.0

        mask = np.abs(expected_scores - score) <= self.tolerance
        old_count = len(self.candidates)
        self.candidates = self.candidates[mask]
        new_count = len(self.candidates)
        print(f"  '{word}' score={score} → eliminated {old_count - new_count}, remaining: {new_count}")

        if score == 100.0:
            print(f"\n  SOLVED in {len(self.guesses)} guesses!")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(target_word=None, quiet=False):
    solver = SemantleSolver()

    if target_word is None:
        target_word = np.random.choice(solver.words)

    if target_word not in solver.w2i:
        print(f"'{target_word}' not in vocabulary!")
        return None

    if not quiet:
        print(f"Simulating for target: '{target_word}'")

    target_vec = solver.vectors[solver.w2i[target_word]]

    for turn in range(1, 100):
        guess = solver.suggest()
        if guess is None:
            if not quiet:
                print("No candidates left!")
            break

        sim = float(target_vec @ solver.vectors[solver.w2i[guess]])
        score = round(sim * 100, 2)

        if not quiet:
            print(f"  Turn {turn}: guess='{guess}', score={score}, candidates={solver.remaining()}")

        solver.update(guess, score)

        if score == 100.0 or guess == target_word:
            if not quiet:
                print(f"\nSolved '{target_word}' in {turn} guesses!")
            return turn

        if not quiet and solver.remaining() <= 5:
            print(f"  Top candidates: {solver.candidate_words()}")

    if not quiet:
        print("Failed to solve in 100 guesses")
    return 100


def benchmark(n=50):
    words, _, _ = load_vectors()
    targets = np.random.choice(words, n, replace=False)
    results = []
    for i, t in enumerate(targets):
        r = simulate(t, quiet=True)
        if r is not None:
            results.append(r)
            print(f"  [{i+1}/{n}] '{t}' → {r} guesses")
    results = np.array(results)
    print(f"\nBenchmark ({len(results)} words):")
    print(f"  Mean: {results.mean():.1f} guesses")
    print(f"  Median: {np.median(results):.0f}")
    print(f"  Min: {results.min()}, Max: {results.max()}")
    print(f"  ≤3 guesses: {(results <= 3).sum()}/{len(results)} ({(results <= 3).mean()*100:.0f}%)")
    print(f"  ≤4 guesses: {(results <= 4).sum()}/{len(results)} ({(results <= 4).mean()*100:.0f}%)")


# ---------------------------------------------------------------------------
# Interactive
# ---------------------------------------------------------------------------

def interactive():
    solver = SemantleSolver()
    print("\nSemantle Solver — Interactive Mode")
    print("I'll suggest words. You enter the score Semantle gives back.")
    print("Type 'q' to quit, 'r' to reset.\n")

    turn = 0
    while True:
        turn += 1
        guess = solver.suggest()
        if guess is None:
            print("No candidates remaining! Try resetting with 'r'.")
            resp = input("(r/q): ").strip()
            if resp == 'q':
                break
            if resp == 'r':
                solver.reset()
                turn = 0
            continue

        print(f"\n  Turn {turn} | Candidates: {solver.remaining()}")
        print(f"  → Guess: {guess}")

        if solver.remaining() <= 10:
            print(f"  Possible words: {solver.candidate_words()}")

        response = input("  Score (or 'skip'/'q'/'r'): ").strip()

        if response == 'q':
            break
        if response == 'r':
            solver.reset()
            turn = 0
            continue
        if response == 'skip':
            continue

        try:
            score = float(response)
        except ValueError:
            print("  Invalid score, skipping.")
            continue

        solver.update(guess, score)

        if score == 100.0:
            print(f"\nSolved in {turn} guesses!")
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--bench" in args:
        idx = args.index("--bench")
        n = int(args[idx + 1]) if idx + 1 < len(args) else 50
        benchmark(n)
    elif "--sim" in args:
        target = None
        if "--target" in args:
            idx = args.index("--target")
            target = args[idx + 1]
        simulate(target)
    else:
        interactive()

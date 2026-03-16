"""
Semantle Solver — minimize guesses using information-theoretic word elimination.

Strategy:
1. Load word2vec embeddings (Google News 300d via gensim)
2. Precompute a similarity matrix for candidate words
3. Each guess returns a cosine similarity score (0-100 scale)
4. Eliminate all words whose similarity to the guessed word doesn't match
   the expected similarity (within tolerance)
5. Pick next guess that maximizes expected candidate elimination

Usage:
    uv run python3 solver.py          # interactive mode against live Semantle
    uv run python3 solver.py --sim    # simulate against a random target word
    uv run python3 solver.py --sim --target crab  # simulate against specific word
"""

import numpy as np
import sys
import time
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Word vectors
# ---------------------------------------------------------------------------

_vectors = None
_words = None
_word2idx = None


def load_vectors(limit=50000, model_name=None):
    """Load word2vec vectors via gensim downloader. Cached after first download."""
    global _vectors, _words, _word2idx
    if _vectors is not None:
        return _words, _vectors, _word2idx

    import gensim.downloader as api

    if model_name is None:
        # Try Google News first; fall back to smaller GloVe if not yet downloaded
        gn = "word2vec-google-news-300"
        try:
            info = api.info()
            model_dir = api.BASE_DIR
            import os
            if os.path.exists(os.path.join(model_dir, gn)):
                model_name = gn
            else:
                print("Google News vectors not yet downloaded.")
                print("Using glove-wiki-gigaword-300 (~376MB, faster download).")
                print("For best results, run: uv run python3 -c \"import gensim.downloader as api; api.load('word2vec-google-news-300')\"")
                model_name = "glove-wiki-gigaword-300"
        except Exception:
            model_name = "glove-wiki-gigaword-300"

    print(f"Loading '{model_name}' vectors...")
    t0 = time.time()
    model = api.load(model_name)

    # Take top `limit` words (most frequent), filter to clean alpha words
    words = []
    vecs = []
    for w in model.key_to_index:
        if len(words) >= limit:
            break
        # Semantle only uses lowercase single words without special chars
        if w.isalpha() and w.islower() and len(w) > 1:
            words.append(w)
            vecs.append(model[w])

    _words = words
    _vectors = normalize(np.array(vecs, dtype=np.float32))
    _word2idx = {w: i for i, w in enumerate(words)}
    print(f"Loaded {len(words)} words in {time.time()-t0:.1f}s")
    return _words, _vectors, _word2idx


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def cos_sim(vec_a, mat_b):
    """Cosine similarity between one vector and a matrix of vectors (all normalized)."""
    return (mat_b @ vec_a).astype(np.float64)


def semantle_score(sim):
    """Convert raw cosine similarity to Semantle's 0-100 scale."""
    return np.round(sim * 100, 2)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SemantleSolver:
    def __init__(self, tolerance=0.25):
        """
        tolerance: how close (in Semantle score units) a candidate's expected
                   similarity must be to the observed score to survive filtering.
                   Semantle rounds to 2 decimals, so 0.25 is tight.
        """
        self.words, self.vectors, self.w2i = load_vectors()
        self.tolerance = tolerance
        self.candidates = list(range(len(self.words)))  # indices
        self.guesses = []

    def reset(self):
        self.candidates = list(range(len(self.words)))
        self.guesses = []

    def remaining(self):
        return len(self.candidates)

    def candidate_words(self):
        return [self.words[i] for i in self.candidates]

    def _pick_best_guess(self):
        """
        Pick the word that, on average, eliminates the most candidates.

        For efficiency: sample candidates if too many, and evaluate a subset
        of potential guesses.
        """
        n_cand = len(self.candidates)

        if n_cand <= 2:
            return self.candidates[0]

        # Subsample candidates for evaluation if needed
        max_eval = 300  # max candidates to evaluate as potential guesses
        max_sim_check = 500  # max candidates to check similarity against

        eval_indices = self.candidates
        if n_cand > max_eval:
            eval_indices = list(np.random.choice(self.candidates, max_eval, replace=False))

        check_indices = self.candidates
        if n_cand > max_sim_check:
            check_indices = list(np.random.choice(self.candidates, max_sim_check, replace=False))

        check_vecs = self.vectors[check_indices]
        n_check = len(check_indices)

        best_guess = eval_indices[0]
        best_score = 0  # best expected elimination

        for gi in eval_indices:
            gvec = self.vectors[gi]
            sims = semantle_score(cos_sim(gvec, check_vecs))

            # For each possible observed score (bucket by tolerance),
            # count how many candidates would be eliminated
            # We approximate by binning similarities
            bins = np.round(sims / self.tolerance) * self.tolerance
            unique, counts = np.unique(bins, return_counts=True)

            # Expected remaining after this guess = sum(count^2) / total
            # (each bin has count candidates; if we land in that bin, count remain)
            expected_remaining = np.sum(counts ** 2) / n_check
            expected_eliminated = n_check - expected_remaining

            if expected_eliminated > best_score:
                best_score = expected_eliminated
                best_guess = gi

        return best_guess

    def suggest(self):
        """Suggest the next word to guess."""
        if len(self.candidates) == 0:
            return None

        # First guess: use a word near the center of the embedding space
        if len(self.guesses) == 0:
            # "the" or similar high-frequency word as anchor
            for starter in ["game", "place", "thing", "water", "light"]:
                if starter in self.w2i:
                    return starter
            return self.words[self.candidates[0]]

        idx = self._pick_best_guess()
        return self.words[idx]

    def update(self, word, score):
        """
        After guessing `word` and getting back `score` (Semantle's number),
        eliminate candidates that don't match.
        """
        self.guesses.append((word, score))

        if word not in self.w2i:
            print(f"  Warning: '{word}' not in vocabulary, can't filter")
            return

        gvec = self.vectors[self.w2i[word]]
        cand_vecs = self.vectors[self.candidates]
        expected_scores = semantle_score(cos_sim(gvec, cand_vecs))

        # Keep candidates whose expected similarity to the guessed word
        # matches the observed score within tolerance
        mask = np.abs(expected_scores - score) <= self.tolerance
        old_count = len(self.candidates)
        self.candidates = [self.candidates[i] for i in range(len(self.candidates)) if mask[i]]
        new_count = len(self.candidates)
        print(f"  '{word}' score={score} → eliminated {old_count - new_count}, remaining: {new_count}")

        if score == 100.0:
            print(f"\n  SOLVED in {len(self.guesses)} guesses!")


# ---------------------------------------------------------------------------
# Simulation mode
# ---------------------------------------------------------------------------

def simulate(target_word=None):
    solver = SemantleSolver()

    if target_word is None:
        target_word = np.random.choice(solver.words)

    if target_word not in solver.w2i:
        print(f"'{target_word}' not in vocabulary!")
        return

    print(f"Simulating Semantle for target: '{target_word}'")
    target_vec = solver.vectors[solver.w2i[target_word]]

    for turn in range(1, 100):
        guess = solver.suggest()
        if guess is None:
            print("No candidates left — something went wrong!")
            break

        # Compute the score the game would return
        if guess in solver.w2i:
            sim = float(target_vec @ solver.vectors[solver.w2i[guess]])
            score = round(sim * 100, 2)
        else:
            score = 0.0

        print(f"  Turn {turn}: guess='{guess}', score={score}, candidates={solver.remaining()}")

        solver.update(guess, score)

        if score == 100.0 or guess == target_word:
            print(f"\nSolved '{target_word}' in {turn} guesses!")
            return turn

        if solver.remaining() <= 5:
            print(f"  Top candidates: {solver.candidate_words()[:5]}")

    print(f"Failed to solve in 100 guesses")
    return 100


# ---------------------------------------------------------------------------
# Interactive mode (you play Semantle, solver tells you what to guess)
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
            guess = input("Manual guess (or 'r'/'q'): ").strip()
            if guess == 'q':
                break
            if guess == 'r':
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

    if "--sim" in args:
        target = None
        if "--target" in args:
            idx = args.index("--target")
            target = args[idx + 1]
        simulate(target)
    else:
        interactive()

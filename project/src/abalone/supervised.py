"""Learning the evaluation weights from examples (Chapter 19): logistic regression on game outcomes.

The supervised counterpart of TDLeaf(λ). Each position of a finished game becomes one example: the six
features of the side to move, labelled with how that game ended for the side to move (1 win, ½ draw, 0 loss).
Logistic regression then fits

    P(side to move wins | s) = σ(Σ βᵢ·fᵢ(s))

This is the method Buro (1995) used for Othello and chess engines use as "Texel tuning". There is no intercept:
the features are zero-sum, so a position seen from the other side has the opposite score, and the model must
give it the complementary probability.

Data: self-play games with the hand-set U at depth 2 and 4 random opening plies, the same budget as the
TDLeaf run (1,200 games), so the two learners see the same kind of games but learn differently: TDLeaf from
its own changing play, logistic regression from fixed games and their final results. Only quiet positions are
used (no push-off available), as in Texel tuning, because a position in the middle of an exchange is
mislabelled by its static features. Train and test are split by game, not by position, so no game is on both
sides.

Scale: TDLeaf reads a value v as tanh(v/K) = 2σ(2v/K) − 1, so the fitted β turn into evaluation weights
w = β·K/2. Both learned weight sets then live on the same scale as the hand-set U.
"""
import json
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

from . import features as F
from .engine import START, successors, apply, can_push_off, move_name, parse_move, WIN_COUNT
from .game import play, random_opening
from .search import Searcher
from .tdleaf import K


def _game(args):
    seed, depth, opening_plies, weights = args
    rng = random.Random(seed)
    start = rng.choice(list(START))
    op = random_opening(rng, start, opening_plies)
    a = Searcher(weights=weights, depth=depth, name="black")
    b = Searcher(weights=weights, depth=depth, name="white")
    r = play(a, b, start, op)
    return {"seed": seed, "start": start, "opening_plies": opening_plies, "moves": r["moves"],
            "winner": r["winner"], "reason": r["reason"], "plies": r["plies"], "off": r["off"]}


def generate(out, games=1200, depth=2, opening_plies=4, seed=19, weights=None, workers=None):
    """Play self-play games with the hand-set U and save them (moves + result) to out/games.jsonl."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(seed * 100_000 + i, depth, opening_plies, weights or F.HAND) for i in range(games)]
    t0 = time.time()
    with mp.Pool(workers or mp.cpu_count()) as pool:
        recs = pool.map(_game, jobs, chunksize=4)
    with open(out / "games.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("%d games in %.0fs" % (len(recs), time.time() - t0), flush=True)
    return recs


def quiet(P, O):
    """No push-off is legal for either side. (The cheap can_push_off prefilter is not enough here: it also
    flags every edge marble that merely touches an opponent, which would remove every position where the
    'exposed' feature is non-zero and leave the model unable to learn it.)"""
    for X, Y in ((P, O), (O, P)):
        if can_push_off(X, Y) and any(off for *_, off in successors(X, Y)):
            return False
    return True


def positions(rec):
    """Replay one game; yield (P, O, label for the side to move P) for its quiet positions."""
    P, O = START[rec["start"]]
    side = 0
    result = {"black": 1.0, "white": 0.0, "draw": 0.5}[rec["winner"]]      # from Black's point of view
    for ply, text in enumerate(rec["moves"]):
        if ply >= rec["opening_plies"] and quiet(P, O):
            yield P, O, result if side == 0 else 1.0 - result
        P2, O2, _ = apply(P, O, parse_move(text))
        P, O, side = O2, P2, side ^ 1


def examples(rec):
    """(features of the side to move, label) for each quiet position of one game."""
    for P, O, t in positions(rec):
        yield F.vector(P, O), t


def load(out):
    return [json.loads(l) for l in (Path(out) / "games.jsonl").read_text().splitlines()]


def test_games(n, seed=19, test_share=0.2):
    """The game ids held out for testing: split by game, the same split for every model."""
    rng = random.Random(seed)
    ids = list(range(n))
    rng.shuffle(ids)
    return set(ids[:int(n * test_share)])


def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z > -40 else 0.0


def fit(X, y, ridge=1e-3, iterations=25):
    """Logistic regression without intercept by Newton's method (IRLS). Returns β."""
    n, d = len(X), len(X[0])
    beta = [0.0] * d
    for _ in range(iterations):
        g = [0.0] * d
        H = [[ridge * n if i == j else 0.0 for j in range(d)] for i in range(d)]
        for x, t in zip(X, y):
            p = sigmoid(sum(b * xi for b, xi in zip(beta, x)))
            r, s = p - t, p * (1 - p)
            for i in range(d):
                g[i] += r * x[i]
                for j in range(i, d):
                    H[i][j] += s * x[i] * x[j]
        for i in range(d):
            g[i] += ridge * n * beta[i]
            for j in range(i):
                H[i][j] = H[j][i]
        step = solve(H, g)
        beta = [b - s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-9:
            break
    return beta


def solve(A, b):
    """Gaussian elimination with partial pivoting (6×6, so no library needed)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))) / M[r][r]
    return x


def metrics(X, y, beta):
    """Log-loss, accuracy on decisive positions, and mean predicted probability."""
    ll, right, decisive = 0.0, 0, 0
    for x, t in zip(X, y):
        p = min(max(sigmoid(sum(b * xi for b, xi in zip(beta, x))), 1e-12), 1 - 1e-12)
        ll -= t * math.log(p) + (1 - t) * math.log(1 - p)
        if t != 0.5:
            decisive += 1
            right += (p > 0.5) == (t == 1.0)
    return {"log_loss": round(ll / len(X), 4), "accuracy_decisive": round(right / decisive, 4) if decisive else None,
            "positions": len(X), "decisive_positions": decisive}


def scale_only(X, y, w):
    """Best single scale c for a fixed weight set (so hand-set and TDLeaf weights are judged fairly)."""
    wl = F.weights_list(w)
    Z = [[sum(a * b for a, b in zip(wl, x))] for x in X]
    return fit(Z, y)[0]


def run(out, games=1200, depth=2, seed=19, test_share=0.2, compare=None, data=None):
    """Generate (or reuse) the games, fit β on the training games, report all weight sets on the test games.

    data = another folder whose games.jsonl to learn from (e.g. the TDLeaf run's own games) instead of out/.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(data or out) / "games.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else generate(out, games, depth, seed=seed)
    test_ids = test_games(len(recs), seed, test_share)
    n_test = len(test_ids)
    Xtr, ytr, Xte, yte = [], [], [], []
    for i, rec in enumerate(recs):
        for x, t in examples(rec):
            (Xte if i in test_ids else Xtr).append(x)
            (yte if i in test_ids else ytr).append(t)
    t0 = time.time()
    beta = fit(Xtr, ytr)
    w = {k: b * K / 2 for k, b in zip(F.FEATURES, beta)}
    report = {
        "data": str(path), "games": len(recs), "train_games": len(recs) - n_test, "test_games": n_test,
        "train_positions": len(Xtr), "test_positions": len(Xte), "fit_seconds": round(time.time() - t0, 1),
        "results": {k: sum(r["reason"] == k for r in recs) for k in ("six off", "repetition", "ply cap")},
        "beta": dict(zip(F.FEATURES, beta)), "weights": w,
        "test": {"logistic": metrics(Xte, yte, beta)},
        "train": {"logistic": metrics(Xtr, ytr, beta)},
        "baseline_coin_flip_log_loss": round(math.log(2), 4),
    }
    for name, ws in (compare or {}).items():
        c = scale_only(Xtr, ytr, ws)
        bl = [c * v for v in F.weights_list(ws)]
        report["test"][name] = dict(metrics(Xte, yte, bl), fitted_scale=c)
    (out / "weights-final.json").write_text(json.dumps(w, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report

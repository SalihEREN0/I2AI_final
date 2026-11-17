"""TDLeaf(λ): learning the evaluation weights by self-play (Chapter 23).

Baxter, Tridgell & Weaver (2000), as used in KnightCap. TD(λ) as in TD-Gammon and Abalearn, but the value of
a position is the alpha-beta value, and the gradient is taken at the leaf of the principal variation, where
the static evaluation produced that value. The same alpha-beta then plays with the learned weights.

Per game (self-play, both sides use the current weights):
  vₜ  = search value at ply t, turned to Black's point of view
  rₜ  = tanh(vₜ / K)                           a value in (−1, 1), read as Black's expected result
  ∇rₜ = (1 − rₜ²) / K · sₜ · f(leafₜ)          sₜ = ±1 for Black/White to move; f = features at the PV leaf
  r_N = the game's reward R
  dₜ  = rₜ₊₁ − rₜ                              temporal difference
  Δw  = α · Σₜ ∇rₜ · Σⱼ≥ₜ λ^(j−t) dⱼ

Reward (experiment E5 in docs/a4-research.md):
  "result"   +1 / −1 for six off, 0 for any draw: the textbook reward, sparse
  "marbles"  ±1 for six off; otherwise tanh(marble difference / 2). Abalearn found that self-play stalls
             when nothing rewards progress; this rewards it without ever beating a real win.

Every game is saved to games.jsonl (moves, result, the iteration that played it), so the training data can be
replayed and reused, e.g. by the supervised models in supervised.py and valuenet.py.

Games of one iteration are played in parallel with the same weights and their updates averaged (a batched
variant of the per-game update; it keeps 8 cores busy). Openings are varied by random first plies.
"""
import json
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

from . import features as F
from .engine import START, successors, apply, move_name, WIN_COUNT
from .search import Searcher, WIN

K = 30.0           # eval units per tanh unit: 3 marbles' worth (marbles weight 10) ≈ r of 0.76


def reward(winner, reason, black_left, white_left, mode):
    if reason == "six off":
        return 1.0 if winner == "black" else -1.0
    if mode == "result":
        return 0.0
    return math.tanh((black_left - white_left) / 2.0)


def self_play(args):
    """One self-play game. Returns the weight update (list), and a summary."""
    weights, depth, lam, mode, opening_plies, seed, max_plies = args
    rng = random.Random(seed)
    start = rng.choice(list(START))
    B, W = START[start]
    s = Searcher(weights=weights, depth=depth, name="tdleaf")
    P, O, side = B, W, 0
    seen, rs, grads, moves = set(), [], [], []
    winner, reason = "draw", "ply cap"
    plies = 0
    for ply in range(max_plies):
        if ply < opening_plies:
            mv = rng.choice(list(successors(P, O)))[0]
        else:
            mv, tr = s.choose(P, O, ply, seen)
            v_side = s.last_value
            sign = 1.0 if side == 0 else -1.0
            if abs(v_side) >= WIN / 2:                     # a forced result: exact target, no gradient
                rs.append(math.copysign(1.0, v_side * sign))
                grads.append(None)
            else:
                r = math.tanh(sign * v_side / K)
                a, b = s.leaf(P, O, s.last_pv)
                f = F.vector(a, b)                         # features at the PV leaf, root mover's view
                g = (1 - r * r) / K * sign
                rs.append(r)
                grads.append([g * x for x in f])
        moves.append(move_name(mv))
        P2, O2, _ = apply(P, O, mv)
        plies += 1
        if O2.bit_count() <= WIN_COUNT:
            winner, reason = ("black" if side == 0 else "white"), "six off"
            P, O = O2, P2
            side ^= 1
            break
        if (O2, P2) in seen:
            # a repeated position: the game is heading for a draw by repetition; stop here
            winner, reason = "draw", "repetition"
            P, O, side = O2, P2, side ^ 1
            break
        seen.add((O2, P2))
        P, O, side = O2, P2, side ^ 1
    Bf, Wf = (P, O) if side == 0 else (O, P)
    if reason == "ply cap":
        nb, nw = Bf.bit_count(), Wf.bit_count()
        winner = "black" if nb > nw else "white" if nw > nb else "draw"
    R = reward(winner, reason, Bf.bit_count(), Wf.bit_count(), mode)

    # TD(λ) on the sequence r_0 .. r_{N-1}, R
    n = len(rs)
    seq = rs + [R]
    d = [seq[t + 1] - seq[t] for t in range(n)]
    update = [0.0] * len(F.FEATURES)
    acc = 0.0
    for t in range(n - 1, -1, -1):                         # acc_t = Σ_{j≥t} λ^{j−t} d_j
        acc = d[t] + lam * acc
        if grads[t] is not None:
            for i, gi in enumerate(grads[t]):
                update[i] += gi * acc
    return update, {"seed": seed, "start": start, "opening_plies": opening_plies, "moves": moves,
                    "winner": winner, "reason": reason, "plies": plies, "R": R,
                    "off": {"black": 14 - Bf.bit_count(), "white": 14 - Wf.bit_count()},
                    "mean_abs_td": sum(abs(x) for x in d) / n if n else 0.0}


def train(out, iterations=50, games_per_iter=8, depth=2, alpha=1.0, lam=0.7, mode="marbles",
          opening_plies=4, max_plies=200, seed=1, workers=None, init=None, alpha_decay=0):
    """Run TDLeaf(λ) and write checkpoints and a log to `out`. Returns the final weights.

    With `alpha_decay` T > 0 the step size at iteration t is alpha / (1 + (t − 1)/T), so the weights settle
    instead of jittering at the noise floor of a constant step."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    w = dict(init or F.HAND)
    cfg = {"iterations": iterations, "games_per_iter": games_per_iter, "depth": depth, "alpha": alpha,
           "alpha_decay": alpha_decay, "lambda": lam, "reward": mode, "opening_plies": opening_plies, "max_plies": max_plies, "seed": seed,
           "K": K, "init": dict(w), "features": F.FEATURES, "scale": F.SCALE}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(out / "log.jsonl", "a")
    games_out = open(out / "games.jsonl", "a")       # every self-play game: moves and result, replayable
    workers = workers or min(games_per_iter, mp.cpu_count())
    t0 = time.time()
    with mp.Pool(workers) as pool:
        for it in range(1, iterations + 1):
            jobs = [(w, depth, lam, mode, opening_plies, seed * 1_000_003 + it * 1000 + g, max_plies)
                    for g in range(games_per_iter)]
            res = pool.map(self_play, jobs)
            avg = [sum(u[i] for u, _ in res) / len(res) for i in range(len(F.FEATURES))]
            a_t = alpha / (1 + (it - 1) / alpha_decay) if alpha_decay else alpha
            for k, du in zip(F.FEATURES, avg):
                w[k] += a_t * du
            games = [g for _, g in res]
            for g in games:
                games_out.write(json.dumps(dict(g, iteration=it)) + "\n")
            games_out.flush()
            row = {"iteration": it, "games": it * games_per_iter, "seconds": round(time.time() - t0, 1),
                   "alpha": a_t, "weights": dict(w), "update": dict(zip(F.FEATURES, avg)),
                   "results": {k: sum(g["reason"] == k for g in games) for k in ("six off", "repetition", "ply cap")},
                   "mean_plies": sum(g["plies"] for g in games) / len(games),
                   "mean_marbles_off": sum(g["off"]["black"] + g["off"]["white"] for g in games) / len(games),
                   "mean_abs_td": sum(g["mean_abs_td"] for g in games) / len(games)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            (out / ("weights-%04d.json" % it)).write_text(json.dumps(w, indent=1))
            print("it %3d  games %4d  %6.0fs  six-off %d rep %d cap %d  plies %5.1f  off %.1f  w %s" % (
                it, row["games"], row["seconds"], row["results"]["six off"], row["results"]["repetition"],
                row["results"]["ply cap"], row["mean_plies"], row["mean_marbles_off"],
                " ".join("%s=%.2f" % (k[:4], v) for k, v in w.items())), flush=True)
    (out / "weights-final.json").write_text(json.dumps(w, indent=1))
    return w

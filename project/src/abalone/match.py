"""Matches under fixed conditions (docs/a4-research.md §6): same openings, both colours, same budget.

Agent specs:
  random                       uniformly random legal move
  greedy                       one ply, greedy on Assignment 1's U (Assignment 3's greedy choice)
  greedy-andor                 the Assignment 3 agent: AND–OR for a push-off it can force, otherwise greedy on U
  ab:depth=3                  alpha-beta; options: depth, time (s per move), tt, ordering, killers, quiescence
                               (0/1), weights=path/to/weights.json, name=label
  a4:time=0.5,...              the same, with the engine exactly as presented in Assignment 4 (src/abalone_a4)
  mm:time=0.5,...              minimax without alpha-beta (search.MinimaxSearcher); options: depth, time,
                               quiescence, weights, name
  mcts:time=0.5,...            Monte Carlo tree search (mcts.MCTS) with the same evaluation at its leaves;
                               options: time, weights, c (UCB1 constant), name
  net:time=0.5,net=path.json   the same alpha-beta with the Chapter 22 value network as its evaluation
"""
import json
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

from .game import play, RandomAgent, GreedyU, GreedyAndOr, random_opening
from .search import Searcher
from .engine import START


def make_agent(spec, seed=0):
    if spec == "random":
        return RandomAgent(seed)
    if spec == "greedy":
        return GreedyU()
    if spec == "greedy-andor":
        return GreedyAndOr()
    kind, _, rest = spec.partition(":")
    if kind not in ("ab", "net", "a4", "mm", "mcts"):
        raise ValueError("unknown agent " + spec)
    opts = dict(kv.split("=", 1) for kv in rest.split(",") if kv)
    kw = {"name": opts.pop("name", spec)}
    net = opts.pop("net", None) if kind == "net" else None
    if "weights" in opts:
        kw["weights"] = json.loads(Path(opts.pop("weights")).read_text())
    if kind == "mcts":
        from .mcts import MCTS
        if "c" in opts:
            kw["c"] = float(opts.pop("c"))
        kw["time_limit"] = float(opts.pop("time", 0.5))
        if opts:
            raise ValueError("unknown options " + ", ".join(opts))
        return MCTS(seed=seed, **kw)
    if "depth" in opts:
        kw["depth"] = int(opts.pop("depth"))
    if "time" in opts:
        kw["time_limit"] = float(opts.pop("time"))
        kw.setdefault("depth", 64)
    for k in ("tt", "ordering", "killers", "quiescence"):
        if k in opts:
            kw[k] = opts.pop(k) not in ("0", "false", "off")
    if opts:
        raise ValueError("unknown options " + ", ".join(opts))
    if kind == "net":
        from .valuenet import NetSearcher, load_net
        return NetSearcher(load_net(net or "experiments/ch22-valuenet/net.json"), **kw)
    if kind == "a4":
        from abalone_a4.search import Searcher as SearcherA4
        return SearcherA4(**kw)
    if kind == "mm":
        from .search import MinimaxSearcher
        return MinimaxSearcher(**kw)
    return Searcher(**kw)


def _one(job):
    a_spec, b_spec, a_black, start, opening, seed, keep_trace = job
    a, b = make_agent(a_spec, seed), make_agent(b_spec, seed + 1)
    trace = [] if keep_trace else None
    r = play(a, b, start, opening) if a_black else play(b, a, start, opening)
    r["a_colour"] = "black" if a_black else "white"
    r["a_result"] = "draw" if r["winner"] == "draw" else "win" if r["winner"] == r["a_colour"] else "loss"
    del r["final"]
    if keep_trace:
        r["trace"] = trace
    return r


def run(a_spec, b_spec, openings=10, opening_plies=2, seed=7, out=None, workers=None, keep_trace=False):
    """Each opening is played twice, A as Black and A as White, from both start positions in turn."""
    rng = random.Random(seed)
    starts = list(START)
    jobs = []
    for i in range(openings):
        start = starts[i % len(starts)]
        op = random_opening(rng, start, opening_plies)
        for a_black in (True, False):
            jobs.append((a_spec, b_spec, a_black, start, op, seed * 1000 + 2 * i + a_black, keep_trace))
    t0 = time.time()
    with mp.Pool(workers or mp.cpu_count()) as pool:
        games = pool.map(_one, jobs)
    s = summarise(a_spec, b_spec, games)
    s["seconds"] = round(time.time() - t0, 1)
    s["conditions"] = {"openings": openings, "opening_plies": opening_plies, "seed": seed, "starts": starts,
                       "max_plies": 200, "repetition": "third occurrence = draw"}
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(s, indent=1))
        with open(out / "games.jsonl", "w") as f:
            for g in games:
                f.write(json.dumps(g) + "\n")
    return s


def summarise(a_spec, b_spec, games):
    n = len(games)
    w = sum(g["a_result"] == "win" for g in games)
    d = sum(g["a_result"] == "draw" for g in games)
    l = n - w - d
    pts = [1.0 if g["a_result"] == "win" else 0.5 if g["a_result"] == "draw" else 0.0 for g in games]
    score = sum(pts) / n
    sd = math.sqrt(sum((p - score) ** 2 for p in pts) / (n - 1)) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n)
    marg = [(g["off"]["white"] - g["off"]["black"]) * (1 if g["a_colour"] == "black" else -1) for g in games]
    elo = None if score in (0.0, 1.0) else round(-400 * math.log10(1 / score - 1))
    return {"a": a_spec, "b": b_spec, "games": n, "a_wins": w, "draws": d, "a_losses": l,
            "a_score": round(score, 3), "a_score_ci95": [round(max(0, score - half), 3), round(min(1, score + half), 3)],
            "elo_diff": elo, "a_mean_marble_margin": round(sum(marg) / n, 2),
            "reasons": {k: sum(g["reason"] == k for g in games) for k in ("six off", "repetition", "ply cap", "no legal move")},
            "mean_plies": round(sum(g["plies"] for g in games) / n, 1)}

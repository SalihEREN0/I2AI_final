"""How far each searcher gets in 0.5 s: completed depth for alpha-beta (current and A4 engine) and minimax, and
iterations for Monte Carlo tree search. Explains the matches in experiments/final.

Run from the repository root, with nothing else running: python experiments/final/baselines-depth/run.py
Positions: the first 100 of the 386 search-cost positions; one process, one position at a time.
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from abalone import supervised as S
from abalone.engine import START, apply, parse_move
from abalone.match import make_agent

W = "weights=experiments/tdleaf-d2-marbles/weights-final.json"
recs = S.load("experiments/ch19-logistic")
positions = []
for gi in sorted(S.test_games(len(recs)))[:20]:
    rec = recs[gi]
    P, O = START[rec["start"]]
    for ply, text in enumerate(rec["moves"]):
        if ply >= rec["opening_plies"] and (ply - rec["opening_plies"]) % 8 == 0:
            positions.append((P, O))
        P2, O2, _ = apply(P, O, parse_move(text))
        P, O = O2, P2
positions = positions[:100]

from abalone.steady import steady  # noqa: E402

out = {"positions": len(positions), "time_per_move": 0.5, "conditions": steady()}
for name, spec in (("alpha-beta", "ab"), ("alpha-beta, A4 engine", "a4"), ("minimax", "mm")):
    depths = []
    for P, O in positions:
        _, tr = make_agent(f"{spec}:time=0.5,{W},name={spec}").choose(P, O)
        depths.append(tr["depth"])
    hist = Counter(depths)
    out[name] = {"mean_depth": round(statistics.mean(depths), 2), "depths": {str(d): hist[d] for d in sorted(hist)}}
    print(name, out[name], flush=True)
its = []
for P, O in positions:
    _, tr = make_agent(f"mcts:time=0.5,{W},name=mcts").choose(P, O)
    its.append(tr["stats"]["iterations"])
out["mcts"] = {"mean_iterations": round(statistics.mean(its)), "min": min(its), "max": max(its)}
print("mcts", out["mcts"], flush=True)
Path("experiments/final/baselines-depth/summary.json").write_text(json.dumps(out, indent=1))

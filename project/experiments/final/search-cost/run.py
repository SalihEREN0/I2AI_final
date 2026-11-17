"""What alpha-beta pruning and each search enhancement save, in nodes, at a fixed depth (the cost side of E2), on
the current engine and with minimax without pruning for reference.

Run from the repository root: python experiments/final/search-cost/run.py
Positions and agent as in experiments/search-cost/run.py (the Assignment 4 version): every 8th ply after the
opening of the first 20 Chapter 19 test games (386 positions), TDLeaf weights. Node counts are exact and
repeatable and equal those of the Assignment 4 engine (tests/test_speed.py); times depend on the machine.
Minimax is run to depth 3 only: at depth 4 it would visit some 60^4 positions per move.
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from abalone import supervised as S
from abalone.engine import START, apply, parse_move
from abalone.match import make_agent

W = "weights=experiments/tdleaf-d2-marbles/weights-final.json"
CONFIGS = {2: {"full": "ab:", "no-pruning": "mm:"},
           3: {"full": "ab:", "no-ordering": "ab:ordering=0,", "no-tt": "ab:tt=0,", "no-killers": "ab:killers=0,",
               "no-quiescence": "ab:quiescence=0,", "no-pruning": "mm:"},
           4: {"full": "ab:", "no-ordering": "ab:ordering=0,", "no-tt": "ab:tt=0,", "no-killers": "ab:killers=0,",
               "no-quiescence": "ab:quiescence=0,"}}

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

from abalone.steady import steady  # noqa: E402

out = {"positions": len(positions), "agent": "alpha-beta (current engine), TDLeaf weights",
       "conditions": steady(), "depths": {}}
for depth, configs in CONFIGS.items():
    row = {}
    for name, spec in configs.items():
        nodes, ms = [], []
        for P, O in positions:
            agent = make_agent("%sdepth=%d,%s,name=%s" % (spec, depth, W, name))
            t0 = time.perf_counter()
            _, tr = agent.choose(P, O)
            ms.append(1000 * (time.perf_counter() - t0))
            nodes.append(tr["stats"]["nodes"])
        row[name] = {"mean_nodes": round(statistics.mean(nodes)), "median_nodes": round(statistics.median(nodes)),
                     "mean_ms": round(statistics.mean(ms), 1)}
        print(depth, name, row[name], flush=True)
    out["depths"][str(depth)] = row
Path("experiments/final/search-cost/summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

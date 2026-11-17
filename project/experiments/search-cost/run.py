"""What each search enhancement saves, in nodes, at a fixed depth (the cost side of experiment E2).

Run from the repository root: python experiments/search-cost/run.py
Positions: every 8th ply after the opening of the first 20 Chapter 19 test games (held-out games of the
hand-set agent). Agent: alpha-beta with the TDLeaf weights at a fixed depth, so the node counts are exact and
repeatable; times depend on the machine and are reported only as a guide.
Pruning, ordering, killers and the TT never change the value at a fixed depth (tests/test_search.py), so the
node count is their whole effect here. Quiescence does change the value: it searches more on purpose.
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
CONFIGS = {"full": "", "no-ordering": ",ordering=0", "no-tt": ",tt=0", "no-killers": ",killers=0",
           "no-quiescence": ",quiescence=0"}

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

out = {"positions": len(positions), "agent": "alpha-beta, TDLeaf weights", "depths": {}}
for depth in (3, 4):
    row = {}
    for name, extra in CONFIGS.items():
        nodes, ms = [], []
        for P, O in positions:
            agent = make_agent("ab:depth=%d,%s%s,name=%s" % (depth, W, extra, name))
            t0 = time.perf_counter()
            _, tr = agent.choose(P, O)
            ms.append(1000 * (time.perf_counter() - t0))
            nodes.append(tr["stats"]["nodes"])
        row[name] = {"mean_nodes": round(statistics.mean(nodes)), "median_nodes": round(statistics.median(nodes)),
                     "mean_ms": round(statistics.mean(ms), 1)}
        print(depth, name, row[name], flush=True)
    out["depths"][str(depth)] = row
Path("experiments/search-cost/summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

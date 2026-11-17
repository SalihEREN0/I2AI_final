"""How much faster is the current engine than the agent presented in Assignment 4, and how much deeper does that
let it search in the same time?

Run from the repository root: python experiments/engine-speed/run.py
Positions: the 386 positions of experiments/search-cost (every 8th ply after the opening of the first 20
Chapter 19 test games). Both engines use the TDLeaf weights; one process, one position at a time, so the
times compare the engines and not the machine's load. At a fixed depth both must return the same move, value
and node count (tests/test_speed.py checks this on other positions; here it is checked on all 386).
"""
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from abalone import supervised as S
from abalone.engine import START, apply, parse_move, move_name
from abalone.search import Searcher
from abalone_a4.search import Searcher as SearcherA4

W = json.loads(Path("experiments/tdleaf-d2-marbles/weights-final.json").read_text())
ENGINES = {"a4": SearcherA4, "current": Searcher}

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

out = {"positions": len(positions), "python": sys.version.split()[0], "conditions": steady(),
       "fixed_depth": {}, "time_limit_0.5s": {}}
for depth in (3, 4):
    rows, same = {}, True
    fingerprints = {}
    for name, cls in ENGINES.items():
        ms, nodes, fp = [], [], []
        for P, O in positions:
            t0 = time.perf_counter()
            mv, tr = cls(weights=W, depth=depth).choose(P, O)
            ms.append(1000 * (time.perf_counter() - t0))
            nodes.append(tr["stats"]["nodes"])
            fp.append((move_name(mv), tr["value"], tr["stats"]["nodes"], tr["stats"]["qnodes"]))
        fingerprints[name] = fp
        rows[name] = {"mean_ms": round(statistics.mean(ms), 1), "median_ms": round(statistics.median(ms), 1),
                      "total_s": round(sum(ms) / 1000, 1), "mean_nodes": round(statistics.mean(nodes)),
                      "nodes_per_s": round(sum(nodes) / (sum(ms) / 1000))}
        print(depth, name, rows[name], flush=True)
    rows["identical_search"] = fingerprints["a4"] == fingerprints["current"]
    rows["speedup"] = round(rows["a4"]["total_s"] / rows["current"]["total_s"], 2)
    out["fixed_depth"][str(depth)] = rows
    print(depth, "identical:", rows["identical_search"], "speedup:", rows["speedup"], flush=True)

for name, cls in ENGINES.items():
    depths, nodes = [], []
    for P, O in positions:
        _, tr = cls(weights=W, time_limit=0.5, depth=64).choose(P, O)
        depths.append(tr["depth"])
        nodes.append(tr["stats"]["nodes"])
    hist = Counter(depths)
    out["time_limit_0.5s"][name] = {"mean_depth": round(statistics.mean(depths), 2),
                                    "depths": {str(d): hist[d] for d in sorted(hist)},
                                    "mean_nodes": round(statistics.mean(nodes))}
    print("0.5s", name, out["time_limit_0.5s"][name], flush=True)

Path("experiments/engine-speed/summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

"""Does a single-process time depend on the core it runs on? The same 20 depth-4 searches (the first 20 of the 386
search-cost positions, current engine, TDLeaf weights), with the process pinned to one logical CPU after another.

Run from the repository root: python experiments/engine-speed/cores.py [CPU ...]
Output on the November 2026 laptop, on battery and on AC power: cores.log. It is why every timing run pins itself to
one fast core (src/abalone/steady.py) and runs on AC power.
"""
import gc
import statistics
import sys
import time

sys.path.insert(0, "src")
from abalone import supervised as S  # noqa: E402
from abalone.engine import START, apply, parse_move  # noqa: E402
from abalone.match import make_agent  # noqa: E402
from abalone.steady import steady  # noqa: E402

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
positions = positions[:20]

for cpu in [int(c) for c in sys.argv[1:]] or [0, 2, 4, 6, 8, 12, 16, 20, 0]:
    steady(cpu)
    time.sleep(0.2)
    ms, nodes = [], 0
    for P, O in positions:
        agent = make_agent("ab:depth=4,%s,name=t" % W)
        gc.collect()
        t0 = time.perf_counter()
        _, tr = agent.choose(P, O)
        ms.append(1000 * (time.perf_counter() - t0))
        nodes += tr["stats"]["nodes"]
    print("cpu %2d: mean %.1f ms, median %.1f ms, %d nodes/s" % (
        cpu, statistics.mean(ms), statistics.median(ms), nodes / (sum(ms) / 1000)), flush=True)

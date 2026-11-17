"""Why do edge and exposed keep moving in the 400-iteration TDLeaf runs?

1. Correlation of the six features over every position of the seed-1 run's self-play, and the eigenvectors of
   their covariance: the direction with the smallest eigenvalue is the one the positions constrain least.
2. The three seeds' weights at iterations 150 and 400, projected on the two eigenvectors that edge and exposed
   load on: the well-constrained one should stay put, the weak one may still move.
Run from the repository root: PYTHONPATH=src python3 experiments/convergence/analyse.py"""
import json
import math
from pathlib import Path

from abalone import features as F
from abalone.engine import START, apply, parse_move


def jacobi(A, sweeps=50):
    """Eigenvalues and eigenvectors (columns) of a small symmetric matrix."""
    n = len(A)
    A = [row[:] for row in A]
    V = [[float(i == j) for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-20:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-15:
                    continue
                th = 0.5 * math.atan2(2 * A[p][q], A[q][q] - A[p][p])
                c, s = math.cos(th), math.sin(th)
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [A[i][i] for i in range(n)], V


RUNS = {"seed 1": "experiments/tdleaf-d2-marbles-400", "seed 2": "experiments/tdleaf-d2-marbles-400-seed2",
        "seed 3": "experiments/tdleaf-d2-marbles-400-seed3"}
paths = [RUNS["seed 1"] + "/games.jsonl"]
X = []
for path in paths:
    for line in open(path):
        rec = json.loads(line)
        P, O = START[rec["start"]]
        for ply, text in enumerate(rec["moves"]):
            if ply >= rec["opening_plies"]:
                X.append(F.vector(P, O))
            P2, O2, _ = apply(P, O, parse_move(text))
            P, O = O2, P2
n, d = len(X), len(F.FEATURES)
mean = [sum(x[i] for x in X) / n for i in range(d)]
cov = [[sum((x[i] - mean[i]) * (x[j] - mean[j]) for x in X) / (n - 1) for j in range(d)] for i in range(d)]
corr = [[cov[i][j] / math.sqrt(cov[i][i] * cov[j][j]) for j in range(d)] for i in range(d)]
vals, V = jacobi(cov)
order = sorted(range(d), key=lambda i: vals[i])
weakest = [V[k][order[0]] for k in range(d)]
sign = 1 if max(weakest, key=abs) > 0 else -1
strong = [V[k][order[-1]] for k in range(d)]                 # largest eigenvalue, for comparison
sign_s = 1 if max(strong, key=abs) > 0 else -1
seeds = {}
for name, run in RUNS.items():
    log = [json.loads(l) for l in open(run + "/log.jsonl")]
    seeds[name] = {}
    for it in (150, 400):
        w = F.weights_list(log[it - 1]["weights"])
        seeds[name][it] = {"weights": {k: round(v, 2) for k, v in zip(F.FEATURES, w)},
                           "along_weakest": round(sum(a * b for a, b in zip(w, weakest)) * sign, 2)}
out = {"games": paths, "positions": n,
       "corr": {a: {b: round(corr[i][j], 3) for j, b in enumerate(F.FEATURES)} for i, a in enumerate(F.FEATURES)},
       "eigenvalues": [round(vals[i], 5) for i in order],
       "weakest_direction": {k: round(sign * v, 3) for k, v in zip(F.FEATURES, weakest)},
       "strongest_direction": {k: round(sign_s * v, 3) for k, v in zip(F.FEATURES, strong)},
       "seeds": seeds}
Path("experiments/convergence/summary.json").write_text(json.dumps(out, indent=1))
print("positions", n)
print("corr row edge   ", out["corr"]["edge"])
print("corr row exposed", out["corr"]["exposed"])
print("eigenvalues", out["eigenvalues"])
print("weakest direction", out["weakest_direction"])
for name, r in seeds.items():
    print(name, "along weakest: it150 %.2f -> it400 %.2f" % (r[150]["along_weakest"], r[400]["along_weakest"]))

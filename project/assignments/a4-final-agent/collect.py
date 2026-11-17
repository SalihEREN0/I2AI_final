"""Collects every number the Assignment 4 deck shows from experiments/ into results/a4data.json, and writes it into
deck.html as `window.A4 = {...};`, so no number on a slide is typed by hand.

Run from the repository root: python assignments/a4-final-agent/collect.py
It only reads saved results; reproducing them is described in src/abalone/README.md.
"""
import ast
import datetime
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from abalone.engine import START, perft  # noqa: E402
HERE = pathlib.Path(__file__).resolve().parent
EX = ROOT / "experiments"


def load(p):
    return json.loads((EX / p).read_text())


def match(name):
    s = load(name + "/summary.json")
    return {"a": s["a"], "b": s["b"], "games": s["games"], "w": s["a_wins"], "d": s["draws"], "l": s["a_losses"],
            "score": s["a_score"], "ci": s["a_score_ci95"], "margin": s["a_mean_marble_margin"],
            "reasons": s["reasons"], "plies": s["mean_plies"]}


MATCHES = ["e1-final-vs-a3-andor", "e1-final-vs-a3", "e1-hand-vs-a3-andor", "e1-a3-andor-vs-greedy", "e2-ablate-learning", "e2-ablate-quiescence", "e2-ablate-ordering", "e2-ablate-tt",
           "e2-ablate-killers", "e5-result-vs-hand", "e5-marbles-vs-result", "match-tdleaf-vs-hand-d2",
           "match-tdleaf-vs-hand-d3", "ch19-logistic-vs-hand", "ch19-logistic-vs-tdleaf", "ch22-net-vs-hand-d2",
           "ch22-net-vs-tdleaf-d2", "ch22-net-vs-tdleaf-time", "refit-logistic-vs-hand", "refit-logistic-vs-tdleaf",
           "refit-net-vs-hand", "refit-net-vs-tdleaf"]

cfg = load("tdleaf-d2-marbles/config.json")
log = [json.loads(l) for l in (EX / "tdleaf-d2-marbles/log.jsonl").read_text().splitlines() if l.strip()]
curve = []
for d in sorted((EX / "e4-learning-curve").glob("it*")):
    it = int(d.name[2:])
    m = match("e4-learning-curve/" + d.name)
    curve.append({"iteration": it, "games": it * cfg["games_per_iter"], "score": m["score"], "ci": m["ci"]})

# E6: the same training for 400 iterations with a decaying step, three seeds (the 150-iteration run stays final)
conv = load("convergence/summary.json")
conv_curve = []
for d in sorted((EX / "e4-learning-curve-400").glob("it*")):
    it = int(d.name[2:])
    conv_curve.append({"iteration": it, "games": it * cfg["games_per_iter"], "score": match("e4-learning-curve-400/" + d.name)["score"]})
cfg400 = load("tdleaf-d2-marbles-400/config.json")
convergence = {"games": cfg400["iterations"] * cfg400["games_per_iter"], "alpha_decay": cfg400["alpha_decay"],
               "seeds": len(conv["seeds"]), "curve": conv_curve,
               "vs150_d2": match("convergence-400-vs-150-d2"), "vs150_time": match("convergence-400-vs-150-time"),
               "positions": conv["positions"], "corr_edge_exposed": conv["corr"]["edge"]["exposed"],
               "eig_min": conv["eigenvalues"][0], "eig_max": conv["eigenvalues"][-1],
               "weakest": conv["weakest_direction"],
               "along_weakest": {k: [v["150"]["along_weakest"], v["400"]["along_weakest"]] for k, v in conv["seeds"].items()},
               "final": {k: v["400"]["weights"] for k, v in conv["seeds"].items()}}

ch19, ch22 = load("ch19-logistic/report.json"), load("ch22-valuenet/report.json")
refl, refn = load("refit-logistic-on-tdleaf-games/report.json"), load("refit-net-on-tdleaf-games/report.json")
tests = sum(1 for n in ast.walk(ast.parse((ROOT / "tests/test_search.py").read_text()))
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"))
src = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in sorted((ROOT / "src/abalone").glob("*.py"))}

A4 = {
    "meta": {"collected": datetime.date.today().isoformat(), "src_sha256": src, "tests": tests,
             "perft": [perft(*START["standard"], d) for d in (1, 2, 3)]},
    "m": {n: match(n) for n in MATCHES},
    "tdleaf": {k: cfg[k] for k in ("iterations", "games_per_iter", "depth", "alpha", "lambda", "reward", "K")}
              | {"games": log[-1]["games"], "seconds": log[-1]["seconds"], "init": cfg["init"],
                 "final": log[-1]["weights"], "features": cfg["features"],
                 "six_off_first": log[0]["results"]["six off"], "six_off_last10": sum(r["results"]["six off"] for r in log[-10:]),
                 "path": [{"games": r["games"], "w": r["weights"]} for r in log if r["iteration"] % 5 == 0 or r["iteration"] == 1]},
    "tdleaf_result_final": load("tdleaf-d2-result/weights-final.json"),
    "curve": curve,
    "convergence": convergence,
    "ch19": {"games": ch19["games"], "train_games": ch19["train_games"], "test_games": ch19["test_games"],
             "train_positions": ch19["train_positions"], "test_positions": ch19["test_positions"],
             "weights": ch19["weights"], "test": {k: {"log_loss": v["log_loss"], "acc": v["accuracy_decisive"]} for k, v in ch19["test"].items()},
             "coin": ch19["baseline_coin_flip_log_loss"]},
    "ch22": {"hidden": ch22["hidden"], "parameters": ch22["parameters"], "best_epoch": ch22["best_epoch"],
             "train_seconds": ch22["train_seconds"], "test": {"log_loss": ch22["test"]["log_loss"], "acc": ch22["test"]["accuracy_decisive"]}},
    "refit": {"logistic_weights": refl["weights"],
              "test": {k: v["log_loss"] for k, v in refl["test"].items()} | {"net": refn["test"]["log_loss"]}},
    "ch4": load("ch4-andor-vs-alphabeta/summary.json"),
    "cost": load("search-cost/summary.json"),
}

(HERE / "results").mkdir(exist_ok=True)
(HERE / "results/a4data.json").write_text(json.dumps(A4, indent=1, ensure_ascii=False))
deck = HERE / "deck.html"
if deck.exists():
    html = deck.read_text()
    line = "window.A4 = " + json.dumps(A4, separators=(",", ":"), ensure_ascii=False) + ";"
    new, n = re.subn(r"^window\.A4 = .*;$", lambda _: line, html, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("deck.html has no `window.A4 = ...;` line")
    deck.write_text(new)
print("matches %d · curve %d · convergence %d · tests %d · written results/a4data.json%s" % (
    len(A4["m"]), len(curve), len(conv_curve), tests, " and deck.html" if deck.exists() else ""))

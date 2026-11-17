"""Ek materyalin tablolarını projenin sonuç dosyalarından üretir (tables-*.tex). Hiçbir sayı elle kopyalanmaz.

Henüz bitmemiş deneyler tabloda "running" olarak görünür; deneyler bitince betik yeniden çalıştırılır.
Çalıştırma: python tablolar.py   (abalone-makale/ek-materyal/ içinden ya da herhangi bir yerden)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1] / "abalone-proje"


def load(rel):
    p = PROJ / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def n(x):
    return f"{x:,}"


def table(name, caption, colspec, header, rows, note=None, size=r"\footnotesize"):
    body = "\n".join(rows)
    tail = f"\\par\\vspace{{2pt}}{{\\scriptsize {note}}}\n" if note else ""
    return (f"\\begin{{table}}[!htbp]\n\\centering\\caption{{{caption}}}\\label{{tab:{name}}}\n{size}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}\n{tail}\\end{{table}}\n")


# ------------------------------------------------------------------ maçlar
MATCH_HEAD = r"Player A & Player B & Budget & W--D--L & Score (95\% CI) & Six off / rep. / cap & Plies"


def match_row(a, b, budget, s):
    if s is None:
        return f"{a} & {b} & {budget} & \\multicolumn{{4}}{{c}}{{(running)}} \\\\"
    lo, hi = s["a_score_ci95"]
    r = s["reasons"]
    return (f"{a} & {b} & {budget} & {s['a_wins']}--{s['draws']}--{s['a_losses']} & "
            f"{s['a_score']:.2f} ({lo:.2f}--{hi:.2f}) & {r['six off']} / {r['repetition']} / {r['ply cap']} & "
            f"{s['mean_plies']:.0f} \\\\")


A4_MATCHES = [
    ("e1-final-vs-a3-andor", "Final agent", "A3 agent", "0.5 s"),
    ("e1-final-vs-a3", "Final agent", "Greedy on $U$", "0.5 s"),
    ("e1-hand-vs-a3-andor", "Hand-set $U$", "A3 agent", "0.5 s"),
    ("e1-a3-andor-vs-greedy", "A3 agent", "Greedy on $U$", "--"),
    None,
    ("e2-ablate-learning", "Full agent", "Hand-set weights", "0.5 s"),
    ("e2-ablate-quiescence", "Full agent", "No quiescence", "0.5 s"),
    ("e2-ablate-ordering", "Full agent", "No move ordering", "0.5 s"),
    ("e2-ablate-tt", "Full agent", "No TT", "0.5 s"),
    ("e2-ablate-killers", "Full agent", "No killer moves", "0.5 s"),
    None,
    ("match-tdleaf-vs-hand-d2", "TDLeaf", "Hand-set", "depth 2"),
    ("match-tdleaf-vs-hand-d3", "TDLeaf", "Hand-set", "depth 3"),
    ("e5-result-vs-hand", "TDLeaf, result reward", "Hand-set", "depth 2"),
    ("e5-marbles-vs-result", "TDLeaf, marble reward", "TDLeaf, result reward", "depth 2"),
    ("convergence-400-vs-150-d2", "TDLeaf, 3,200 games", "TDLeaf, 1,200 games", "depth 2"),
    ("convergence-400-vs-150-time", "TDLeaf, 3,200 games", "TDLeaf, 1,200 games", "0.5 s"),
    None,
    ("ch19-logistic-vs-hand", "Logistic", "Hand-set", "depth 2"),
    ("ch19-logistic-vs-tdleaf", "Logistic", "TDLeaf", "depth 2"),
    ("refit-logistic-vs-hand", "Logistic, refit", "Hand-set", "depth 2"),
    ("refit-logistic-vs-tdleaf", "Logistic, refit", "TDLeaf", "depth 2"),
    ("ch22-net-vs-hand-d2", "Value network", "Hand-set", "depth 2"),
    ("ch22-net-vs-tdleaf-d2", "Value network", "TDLeaf", "depth 2"),
    ("ch22-net-vs-tdleaf-time", "Value network", "TDLeaf", "0.5 s"),
    ("refit-net-vs-hand", "Network, refit", "Hand-set", "depth 2"),
    ("refit-net-vs-tdleaf", "Network, refit", "TDLeaf", "depth 2"),
]

FINAL_MATCHES = [
    ("final-vs-a4", "Final agent", "A4 engine", "0.5 s"),
    ("e1-final-vs-a3-andor", "Final agent", "A3 agent", "0.5 s"),
    ("e1-hand-vs-a3-andor", "Hand-set $U$", "A3 agent", "0.5 s"),
    None,
    ("e2-ablate-learning", "Full agent", "Hand-set weights", "0.5 s"),
    ("e2-ablate-quiescence", "Full agent", "No quiescence", "0.5 s"),
    ("e2-ablate-ordering", "Full agent", "No move ordering", "0.5 s"),
    ("e2-ablate-tt", "Full agent", "No TT", "0.5 s"),
    ("e2-ablate-killers", "Full agent", "No killer moves", "0.5 s"),
    ("e2-ablate-ordering-200", "Full agent", "No ordering (200 games)", "0.5 s"),
    ("e2-ablate-tt-200", "Full agent", "No TT (200 games)", "0.5 s"),
    ("e2-ablate-killers-200", "Full agent", "No killers (200 games)", "0.5 s"),
    None,
    ("mcts-vs-final", "MCTS", "Final agent", "0.5 s"),
    ("minimax-vs-final", "Minimax", "Final agent", "0.5 s"),
]


def match_table(name, caption, entries, prefix, note):
    rows = []
    for e in entries:
        if e is None:
            rows.append(r"\midrule")
            continue
        folder, a, b, budget = e
        rows.append(match_row(a, b, budget, load(f"{prefix}{folder}/summary.json")))
    return table(name, caption, "@{}lllcccr@{}", MATCH_HEAD, rows, note)


def learning_curve():
    a4 = load("assignments/a4-final-agent/results/a4data.json")
    rows = []
    for c in a4["curve"]:
        s = load(f"experiments/e4-learning-curve/it{c['iteration']:04d}/summary.json")
        rows.append(f"{c['iteration']} & {n(c['games'])} & {s['a_wins']}--{s['draws']}--{s['a_losses']} & "
                    f"{c['score']:.2f} ({c['ci'][0]:.2f}--{c['ci'][1]:.2f}) \\\\")
    rows.append(r"\midrule")
    for c in a4["convergence"]["curve"]:
        s = load(f"experiments/e4-learning-curve-400/it{c['iteration']:04d}/summary.json")
        ci = s["a_score_ci95"] if s else (None, None)
        wdl = f"{s['a_wins']}--{s['draws']}--{s['a_losses']}" if s else "--"
        rows.append(f"{c['iteration']} (decaying $\\alpha$) & {n(c['games'])} & {wdl} & {c['score']:.2f}"
                    + (f" ({ci[0]:.2f}--{ci[1]:.2f})" if s else "") + r" \\")
    return table("curve", r"TDLeaf($\lambda$) checkpoints against the hand-set weights (depth 2, 50 games each)",
                 "@{}lrcc@{}", r"Iteration & Self-play games & W--D--L & Score (95\% CI)", rows,
                 "Top: the 1,200-game run of the final agent ($\\alpha=10$). Bottom: the 3,200-game convergence run, "
                 "seed 1, step size $\\alpha_t = 10/(1+(t-1)/200)$.")


def weights():
    a4 = load("assignments/a4-final-agent/results/a4data.json")
    feats = a4["tdleaf"]["features"]
    names = {"marbles": "Marbles", "centre": "Centrality", "cohesion": "Cohesion", "edge": "Edge",
             "exposed": "Exposed", "wedge": "Wedge"}
    cols = [("Hand-set", a4["tdleaf"]["init"]), ("TDLeaf", a4["tdleaf"]["final"]),
            ("TDLeaf, result", a4["tdleaf_result_final"]), ("Logistic", a4["ch19"]["weights"]),
            ("Logistic, refit", a4["refit"]["logistic_weights"])]
    cols += [(f"3,200 games, seed {i}", a4["convergence"]["final"][f"seed {i}"]) for i in (1, 2, 3)]
    rows = [names[f] + " & " + " & ".join(f"{c[1][f]:.2f}" for c in cols) + r" \\" for f in feats]
    head = "Feature & " + " & ".join(c[0] for c in cols)
    return table("weights", "Evaluation weights on the scaled features (scales 1, 10, 10, 5, 5, 3)",
                 "@{}l" + "r" * len(cols) + "@{}", head, rows, size=r"\scriptsize")


def prediction():
    r19 = load("experiments/ch19-logistic/report.json")
    r22 = load("experiments/ch22-valuenet/report.json")
    rows = []
    order = [("logistic", "Logistic regression"), ("tdleaf-marbles", "TDLeaf, marble reward"),
             ("tdleaf-result", "TDLeaf, result reward"), ("hand", "Hand-set weights")]
    for k, label in order:
        t = r19["test"].get(k)
        if t:
            rows.append(f"{label} & {t['log_loss']:.4f} & {100 * t['accuracy_decisive']:.1f}\\% \\\\")
    rows.append(f"Value network ({n(r22['parameters'])} parameters) & {r22['test']['log_loss']:.4f} & "
                f"{100 * r22['test']['accuracy_decisive']:.1f}\\% \\\\")
    rows.append(f"Coin flip & {r19['baseline_coin_flip_log_loss']:.4f} & 50.0\\% \\\\")
    return table("prediction", "Outcome prediction on 240 held-out games of the hand-set agent",
                 "@{}lcc@{}", r"Model & Log loss & Accuracy (decisive)", rows,
                 f"{n(r19['test_positions'])} quiet positions; fixed weight sets are scaled by one factor fitted on "
                 f"the {n(r19['train_positions'])} training positions.")


# ------------------------------------------------------------------ arama
def tree():
    t = load("assignments/a2-search/results/tree.json")
    rows = [f"{d} & {n(t['perft'][d - 1])} & {n(t['distinct'][d - 1])} & {n(t['reached'][d - 1])} & "
            f"{t['secs'][d - 1]} \\\\" for d in range(1, t["depth"] + 1)]
    return table("tree", "Growth of the game tree from the standard opening",
                 "@{}crrrr@{}", r"Plies & Move sequences (perft) & Distinct positions & Positions within & Time (s)",
                 rows, f"Iterative deepening to depth 4 re-expands {t['idsOverPct']}\\% more nodes than a single "
                       f"depth-limited pass; a position takes {t['bytesPerState']} bytes.")


def a2():
    b, m = load("assignments/a2-search/results/bench.json"), load("assignments/a2-search/results/more.json")
    rows = []
    for name in ("BFS", "UCS", "A* hR", "A* hF", "Greedy hF"):
        cells = [f"{n(r[name]['gen'])}{'$^{c}$' if r[name]['capped'] else ''} ({r[name]['bstar']})"
                 for r in b["table"]]
        s = b["summary"][name]
        rows.append(f"{name.replace('hR', '$h_R$').replace('hF', '$h_F$')} & " + " & ".join(cells)
                    + f" & {s['solved']} & {s['optimal']} \\\\")
    rows.append(r"\midrule")
    for name in ("IDA* hF", "IDA* hF + TT", "Weighted A* hF, w = 2"):
        cells = [f"{n(r[name]['gen'])} [{n(r[name]['mem'])}]" for r in m["table"]]
        s = m["summary"][name]
        label = name.replace("hF", "$h_F$").replace("w = 2", "$w=2$")
        rows.append(f"{label} & " + " & ".join(cells) + f" & {s['solved']} & {s['optimal']} \\\\")
    return table("a2", "Assignment 2 benchmark: 60 push-off puzzles (19, 37 and 4 at $d^*=2,3,4$)",
                 "@{}lrrrcc@{}", r"Strategy & $d^*=2$ & $d^*=3$ & $d^*=4$ & Solved & Optimal", rows,
                 "Top: mean generated nodes (effective branching factor $b^*$), limit $10^6$; $^c$ some runs stopped "
                 "at the limit. Bottom: mean generated nodes [mean peak memory in states], limit $2\\times10^6$.")


def a3():
    l = load("assignments/a3-complex-environments/results/local.json")
    names = [("steepest", "Steepest-ascent HC"), ("stochastic", "Stochastic HC"), ("firstChoice", "First-choice HC"),
             ("sideways", "HC, up to 100 sideways moves"), ("annealing", "Simulated annealing"), ("lrta", "LRTA*")]
    rows = [f"A*, $h_F$ & {l['astar']['solved']} & {l['n']} & {l['astar']['meanMoves']} & "
            f"{n(l['astar']['meanGen'])} (generated) & goal {l['astar']['solved']} \\\\"]
    for k, label in names:
        a = l["algos"][k]
        stops = ", ".join(f"{s} {v}" for s, v in a["stops"].items())
        rows.append(f"{label} & {a['solved']} & {a['optimal']} & {a['meanMoves']} & {n(a['meanEvals'])} & {stops} \\\\")
    rows.append(f"Stochastic HC, up to {l['restart']['R']} restarts & {l['restart']['solved']} & -- & -- & -- & "
                f"mean tries {l['restart']['meanTries']} \\\\")
    st, cu = l["stuck"], l["climbU"]
    note = (f"Steepest ascent stopped {st['n']} times, all on plateaus: {st['shoulder']} shoulders, {st['flatMin']} "
            f"flat minima, {st['localMin']} strict local minima, {st['escapeWorseFirstOnly']} ridges, {st['unknown']} "
            f"not classified within 500 boards. Hill climbing on the Assignment~1 utility pushed a marble off in "
            f"{cu['anyCapture']} of {cu['n']} puzzles. On the {l['commonSolved']['n']} puzzles solved by A*, "
            f"sideways HC, annealing and LRTA*, the mean solution lengths are "
            + ", ".join(f"{v}" for v in l["commonSolved"]["meanMoves"].values()) + ".")
    return table("a3", f"Assignment 3 benchmark: local and online search on {l['n']} push-off puzzles",
                 "@{}lccrrl@{}", r"Method & Solved & Optimal & Mean moves & Mean evaluations & Stop reasons", rows,
                 note, size=r"\scriptsize")


def cost(rel, name, caption):
    c = load(rel)
    if c is None:
        return table(name, caption, "@{}lc@{}", "Configuration & ", [r"\multicolumn{2}{c}{(running)} \\"])
    depths = sorted(c["depths"], key=int)
    confs = []
    for d in depths:
        for k in c["depths"][d]:
            if k not in confs:
                confs.append(k)
    labels = {"full": "Full agent", "no-ordering": "No move ordering", "no-tt": "No TT", "no-killers": "No killers",
              "no-quiescence": "No quiescence", "no-pruning": "No pruning (minimax)"}
    rows = []
    for k in confs:
        cells = []
        for d in depths:
            v = c["depths"][d].get(k)
            cells.append(f"{n(v['mean_nodes'])} & {v['mean_ms']:,.1f}" if v else "-- & --")
        rows.append(labels[k] + " & " + " & ".join(cells) + r" \\")
    head = "Configuration & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{Depth {d}}}" for d in depths)
    head += r" \\ & " + " & ".join("Nodes & ms" for _ in depths)
    cond = f"; one process, {c['conditions']}" if "conditions" in c else ""
    return table(name, caption, "@{}l" + "rr" * len(depths) + "@{}", head, rows,
                 f"Means over {c['positions']} middle-game positions{cond}.")


def speed():
    s = load("experiments/engine-speed/summary.json")
    if s is None:
        return table("speed", "Engine speed", "@{}lc@{}", "Engine & ", [r"\multicolumn{2}{c}{(running)} \\"])
    engines = (("a4", "A4 engine"), ("current", "Current engine"))
    rows = []
    for d, r in s["fixed_depth"].items():
        for k, label in engines:
            v = r[k]
            rows.append(f"depth {d} & {label} & {d} & {v['mean_ms']:.1f} & {n(v['mean_nodes'])} \\\\")
        same = "identical search" if r["identical_search"] else "search DIFFERS"
        rows.append(f"& \\multicolumn{{4}}{{l}}{{speed-up {r['speedup']:.2f}$\\times$, {same}}} \\\\")
        rows.append(r"\midrule")
    dists = []
    for k, label in engines:
        v = s["time_limit_0.5s"][k]
        rows.append(f"0.5 s & {label} & {v['mean_depth']:.2f} & -- & {n(v['mean_nodes'])} \\\\")
        dists.append(f"{label} " + ", ".join(f"{d}: {c}" for d, c in v["depths"].items()))
    return table("speed", f"The A4 engine and the current engine on the same {s['positions']} positions",
                 "@{}llrrr@{}", r"Limit & Engine & Depth & ms & Nodes", rows,
                 "Means per position. Identical search: the same move, value and node count on every position. "
                 "0.5 s: mean completed depth and nodes per move; positions per completed depth: "
                 + "; ".join(dists) + ". "
                 f"One process, one position at a time, {s.get('conditions', 'not pinned')}; Python {s['python']}.")


def depth():
    s = load("experiments/final/baselines-depth/summary.json")
    if s is None:
        return table("depth", "How far each searcher gets in 0.5 s", "@{}lc@{}", "Searcher & ",
                     [r"\multicolumn{2}{c}{(running)} \\"])
    rows = []
    for k, label in (("alpha-beta", "Alpha--beta, current engine"), ("alpha-beta, A4 engine", "Alpha--beta, A4 engine"),
                     ("minimax", "Minimax, no pruning")):
        v = s[k]
        dist = ", ".join(f"{d}: {c}" for d, c in v["depths"].items())
        rows.append(f"{label} & {v['mean_depth']} & positions per depth: {dist} \\\\")
    m = s["mcts"]
    rows.append(f"MCTS & -- & {n(m['mean_iterations'])} iterations per move (min {n(m['min'])}, max {n(m['max'])}) \\\\")
    return table("depth", f"How far each searcher gets in 0.5 s on {s['positions']} middle-game positions",
                 "@{}lcl@{}", "Searcher & Mean depth & Distribution", rows,
                 f"One process, one position at a time, {s.get('conditions', 'not pinned')}, nothing else running; "
                 "all four use the TDLeaf evaluation. Depth: last completed iteration of iterative deepening.")


def main():
    parts = {
        "tree": tree(), "a2": a2(), "a3": a3(),
        "matches-a4": match_table("matches-a4", "All Assignment 4 matches (50 games each, A4 engine)", A4_MATCHES,
                                  "experiments/", "Score = (W + D/2)/N for player A. End reasons: six marbles off, "
                                                  "third repetition, 200-ply cap (decided by marble count)."),
        "curve": learning_curve(), "weights": weights(), "prediction": prediction(),
        "cost-a4": cost("experiments/search-cost/summary.json", "cost-a4",
                        "Search cost of the A4 engine at fixed depth"),
        "speed": speed(),
        "depth": depth(),
        "matches-final": match_table("matches-final", "Matches with the current engine", FINAL_MATCHES,
                                     "experiments/final/", "50 games each, except the final agent against the A4 "
                                                           "engine (100) and the three 200-game ablations."),
        "cost-final": cost("experiments/final/search-cost/summary.json", "cost-final",
                           "Search cost of the current engine at fixed depth, with minimax without pruning"),
    }
    for name, text in parts.items():
        (HERE / f"tables-{name}.tex").write_text("% tablolar.py tarafından üretildi; elle düzenlemeyin.\n" + text,
                                                 encoding="utf-8")
    print("wrote", len(parts), "tables;", sum("(running)" in t for t in parts.values()), "still running")


if __name__ == "__main__":
    main()

"""Abalone through the lens of Chapter 4: the opponent's reply (AND–OR), what is unseen, what is irreversible, and the traps of local search.

Every number in the Assignment 3 deck comes from this file. The rules engine (ACTIONS, RESULT), both generators, hF and A*
are imported unchanged from Assignment 2: ../a2-search/abalone_search.py.

Usage:
  python abalone_ch4.py recheck           # Assignment 2's numbers again with the same engine (44/73/81, perft, 75/6, 439)
  python abalone_ch4.py test              # test board: 73 moves, 39 replies, AND–OR plan, U's trap, observation, reversibility
  python abalone_ch4.py opening           # opening puzzle: plateau with hF, local maximum with U
  python abalone_ch4.py local [N] [SEED]  # local search methods on N puzzles (~10 min)
  python abalone_ch4.py inject [HTML]     # results/*.json -> window.A3 in the deck
"""
import hashlib
import json
import math
import pathlib
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
A2 = HERE.parent / "a2-search"
sys.path.insert(0, str(A2))
from abalone_search import (CELLS, DIST, MID_B, MID_W, NAME, NB, NBMASK, RING, START_B, START_W, Puzzle,  # noqa: E402
                            best_first, bfs, bits, mask, move_name, naive_successors, perft, random_position,
                            successors)

OUT = HERE / "results"
DECK = HERE / "deck.html"
RING4 = sum(1 << c for c in range(len(CELLS)) if RING[c] == 4)

# Test board (slides 1–4): built from the opening, a position where one push knocks a white marble off. Black to move.
TEST_B = mask("A1 A2 A3 A4 A5 B2 B3 B4 B5 E5 E6 E7 D6 D7".split())
TEST_W = mask("I5 I6 I7 I8 I9 H4 H5 H6 H7 H8 H9 G7 E8 E9".split())
PLAN2 = "E6 E7 E8 → E"            # second push of the plan that ignores the opponent


def save(name, data):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def load(name, folder=OUT):
    return json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))


def st(B, W):
    """The board as the planner reads it: 61 cells, b / w / empty."""
    return "".join("b" if B >> i & 1 else "w" if W >> i & 1 else "." for i in range(len(CELLS)))


def names(m):
    return [NAME[c] for c in bits(m)]


def white_off(W):
    return 14 - W.bit_count()


# ------------------------------------------------------------------ U: the utility function from Assignment 1
def u_terms(B, W):
    centre = sum(4 - RING[c] for c in bits(B))                    # merkez 0. halka, kenar 4. halka
    coh = sum((NBMASK[c] & B).bit_count() for c in bits(B)) // 2  # pairs of touching marbles
    return {"off": white_off(W), "centre": centre, "coh": coh, "edge": (B & RING4).bit_count()}


def U(B, W):
    t = u_terms(B, W)
    return 10 * t["off"] + 2 * t["centre"] + t["coh"] - 3 * t["edge"]


# ------------------------------------------------------------------ Assignment 2's numbers again
def cmd_recheck():
    """Same engine, same inputs: every number Assignment 2 reported must come out again."""
    a2 = {k: load(k, A2 / "results") for k in ("count", "tree", "demo")}
    count = {}
    for key, P, O in [("start", START_B, START_W), ("midB", MID_B, MID_W), ("midW", MID_W, MID_B)]:
        kinds = Counter(f"{kind}-{len(mv[0])}" for mv, kind, *_ in successors(P, O))
        count[key] = sum(kinds.values())
        if key == "start":
            count["startKinds"] = dict(kinds)
    count["A2"] = [a2["count"][k]["total"] for k in ("start", "midB", "midW")]

    ours, distinct, layer = [], [], {(START_B, START_W)}
    for d in (1, 2, 3):
        ours.append(perft(START_B, START_W, d))
        layer = {(O2, P2) for P, O in layer for _, _, P2, O2, _ in successors(P, O)}
        distinct.append(len(layer))
    perft_ = {"ours": ours, "distinct": distinct, "A2": a2["tree"]["perft"][:3], "A2distinct": a2["tree"]["distinct"][:3]}

    # Assignment 2's "plan the opponent breaks" example: two whites from the mid-game, White's replies after the first move
    pz = Puzzle(MID_B, MID_W, 2)
    path = best_first(pz, "hF", path=True)["path"]
    rest = [next(mv for mv, _, P2, O2, _ in successors(*s) if (P2, O2) == t) for s, t in zip(path[1:], path[2:])]
    B1, W1 = path[1]
    replies, breakers = list(successors(W1, B1)), []
    for mvw, _, W2, B2, _ in replies:
        B, W, offs = B2, W2, 0
        for mv in rest:
            nxt = [x for x in successors(B, W) if x[0] == mv]
            if not nxt:
                offs = -1
                break
            _, _, B, W, o = nxt[0]
            offs += o
        if offs != len(rest):
            breakers.append(move_name(mvw))
    plan = {"firstOff": {"b": 14 - B1.bit_count(), "w": white_off(W1)}, "replies": len(replies), "breaks": len(breakers),
            "breakers": breakers, "A2": [a2["demo"]["plan"]["replies"], a2["demo"]["plan"]["breaks"]],
            "sameSet": set(breakers) == set(a2["demo"]["plan"]["breakers"])}

    # Assignment 2's slide 7 puzzle: BFS, A*, greedy on the same board
    q = a2["demo"]["puzzle"]
    pz = Puzzle(mask(q["B"]), mask(q["W"]), 1)
    runs = {"bfs": bfs(pz, 5_000_000), "astar": best_first(pz, "hF", limit=5_000_000),
            "greedy": best_first(pz, "hF", astar=False, limit=5_000_000)}
    puzzle = {"ours": {k: {"gen": v["gen"], "len": v["len"]} for k, v in runs.items()}, "A2": q["runs"]}

    data = {"count": count, "perft": perft_, "plan": plan, "puzzle": puzzle}
    same = [count["A2"] == [count["start"], count["midB"], count["midW"]],
            perft_["ours"] == perft_["A2"] and perft_["distinct"] == perft_["A2distinct"],
            plan["A2"] == [plan["replies"], plan["breaks"]] and plan["sameSet"],
            all(puzzle["ours"][a][x] == q["runs"][b][x] for a, b in (("bfs", "BFS"), ("astar", "A* hF"), ("greedy", "Greedy hF"))
                for x in ("gen", "len"))]
    data["identical"] = sum(same)
    print(f"moves {count['start']}/{count['midB']}/{count['midW']} · perft {ours} distinct {distinct} · "
          f"plan {plan['replies']} replies, {plan['breaks']} break · puzzle A* {puzzle['ours']['astar']} · identical {sum(same)} of 4")
    save("recheck", data)


# ------------------------------------------------------------------ AND–OR search (AIMA Figure 4.11)
class AndOr:
    """Black to move (OR node: we choose); each White reply is an outcome (AND node: all must be handled).

    Goal: `need` white marbles pushed off since the start, within at most `depth` of our moves. If the same board
    appears twice on a branch, that branch fails (the cycle check in Figure 4.11). Shallowest sub-plan first: we first
    check whether this move reaches the goal; the move named `prefer` is tried before the others.
    """

    def __init__(self, need, prefer=None):
        self.need, self.prefer = need, prefer
        self.or_calls = self.and_calls = self.boards = 0

    def ordered(self, B, W):
        kids = list(successors(B, W))
        self.boards += len(kids)
        if self.prefer:
            kids.sort(key=lambda x: move_name(x[0]) != self.prefer)    # stable sort: the generator's order is kept
        return kids

    def or_search(self, B, W, depth, path):
        self.or_calls += 1
        if (B, W) in path:
            return None
        kids = self.ordered(B, W)
        for mv, _, B2, W2, _ in kids:
            if white_off(W2) >= self.need:
                return (mv, None)
        if depth == 1:
            return None
        for mv, _, B2, W2, _ in kids:
            sub = self.and_search(B2, W2, depth - 1, path | {(B, W)})
            if sub is not None:
                return (mv, sub)
        return None

    def and_search(self, B, W, depth, path):
        self.and_calls += 1
        replies = list(successors(W, B))
        self.boards += len(replies)
        plans = []
        for rv, _, W3, B3, _ in replies:
            p = self.or_search(B3, W3, depth, path)
            if p is None:
                return None
            plans.append((rv, B3, W3, p))
        return plans


def forced_naive(B, W, need, depth):
    """Independent recheck: no plan object, and the second generator that follows the rulebook literally.

    Answers only "is it forced?": do we have a move after which the goal follows whatever White does.
    """
    for (B2, W2) in naive_successors(B, W):
        if white_off(W2) >= need:
            return True
        if depth > 1 and all(forced_naive(B3, W3, need, depth - 1) for (W3, B3) in naive_successors(W2, B2)):
            return True
    return False


def forced_fast(B, W, need, depth):
    for _, _, B2, W2, _ in successors(B, W):
        if white_off(W2) >= need:
            return True
        if depth > 1 and all(forced_fast(B3, W3, need, depth - 1) for _, _, W3, B3, _ in successors(W2, B2)):
            return True
    return False


def cmd_test():
    """Slides 1–4 and 9: White's replies on the test board, the AND–OR plan, U's trap, observation and reversibility."""
    B0, W0 = TEST_B, TEST_W
    moves = list(successors(B0, W0))
    push_offs = [x for x in moves if x[4]]
    pushes = [x for x in moves if x[1] == "sumito" and not x[4]]
    quiet = [x for x in moves if x[1] != "sumito"]
    reversible = sum(any((P3, O3) == (B0, W0) for _, _, P3, O3, _ in successors(P2, O2)) for _, _, P2, O2, _ in quiet)
    cap = push_offs[0]
    B1, W1 = cap[2], cap[3]

    # the plan that ignores the opponent: push, then push again. After each White reply, does the second push still knock a marble off?
    replies = list(successors(W1, B1))
    breakers, boards = [], {}
    for rv, _, W2, B2, _ in replies:
        if not any(move_name(m) == PLAN2 and d for m, _, _, _, d in successors(B2, W2)):
            breakers.append(move_name(rv))
            boards[move_name(rv)] = st(B2, W2)
    # plans that push two whites off in two moves while White is frozen (Chapter 3's assumption)
    frozen = sum(1 for _, _, P2, O2, _ in moves for _, _, P3, O3, _ in successors(P2, O2) if white_off(O3) >= 2)

    # AND–OR: is there a plan for each first move? Counters are totals over the whole search.
    t0 = time.perf_counter()
    ao = AndOr(2, prefer=PLAN2)
    ao.or_calls += 1                                   # root OR node
    root = ao.ordered(B0, W0)
    found = [(mv, ao.and_search(B2, W2, 1, {(B0, W0)})) for mv, _, B2, W2, _ in root]
    ms = round((time.perf_counter() - t0) * 1000)
    with_plan = [(mv, p) for mv, p in found if p is not None]
    mv1, branches = with_plan[0]
    groups, lost = {}, 0
    for rv, B3, W3, (mv2, _) in branches:
        groups.setdefault(move_name(mv2), []).append(move_name(rv))
        P4 = next(P for m, _, P, _, _ in successors(B3, W3) if m == mv2)
        lost = max(lost, 14 - P4.bit_count())
    big = max(len(v) for v in groups.values())
    answers = sorted(({"act": a, "replies": len(v), **({"which": v} if len(v) < big else {})} for a, v in groups.items()),
                     key=lambda x: x["replies"])
    andor = {"need": 2, "ourMoves": 2, "firstMovesWithPlan": [move_name(m) for m, _ in with_plan], "of": len(moves),
             "answers": answers, "nodes": 1 + 2 * len(branches), "leaves": len(branches), "lost": lost,
             "search": {"orCalls": ao.or_calls, "andCalls": ao.and_calls, "boards": ao.boards, "ms": ms}}

    t0 = time.perf_counter()
    good = [(B2, W2) for (B2, W2) in naive_successors(B0, W0)
            if all(forced_naive(B3, W3, 2, 1) for (W3, B3) in naive_successors(W2, B2))]
    forcing = [move_name(mv) for mv, _, B2, W2, _ in moves if (B2, W2) in good]   # name only for the report
    recheck = {"firstMovesForcing": forcing, "agrees": forcing == andor["firstMovesWithPlan"],
               "ms": round((time.perf_counter() - t0) * 1000)}

    # greedy trap: the best one-move choice by U (ties broken by generator order)
    base = U(B0, W0)
    du = [(U(P2, O2) - base, mv, P2, O2) for mv, _, P2, O2, _ in moves]
    best = max(du, key=lambda x: x[0])                # max returns the first largest
    cap_du = next(d for d, mv, *_ in du if mv == cap[0])
    rank = 1 + sum(d > cap_du for d, *_ in du)
    _, mvb, Bb, Wb = best
    after = list(successors(Wb, Bb))
    refuting = [move_name(rv) for rv, _, W2, B2, _ in after if not forced_fast(B2, W2, 2, 2)]
    greedy = {"best": move_name(mvb), "bestDU": best[0], "captureDU": cap_du, "rankOfCapture": rank,
              "replies": len(after), "refuting": len(refuting), "refutingExamples": refuting[:6]}

    # counterfactual: if D6 had stayed on its opening cell (B6)
    Bc = B0 & ~mask(["D6"]) | mask(["B6"])
    Bc1, Wc1 = next((P2, O2) for mv, _, P2, O2, _ in successors(Bc, W0) if move_name(mv) == move_name(cap[0]))
    unanswered = [move_name(rv) for rv, _, W3, B3, _ in successors(Wc1, Bc1) if not forced_fast(B3, W3, 2, 1)]
    counterfactual = {"change": "D6 stays home on B6", "forcedWithin2": forced_fast(Bc, W0, 2, 2), "unanswered": unanswered}

    # partial observation: if we only saw cells within r of our marbles, how many boards would be possible?
    belief = []
    for r in (1, 2):
        seen = 0
        for c in bits(B0):
            seen |= sum(1 << t for t in range(len(CELLS)) if DIST[c][t] <= r)
        n_seen, vis = seen.bit_count(), (W0 & seen).bit_count()
        hidden, unseen = W0.bit_count() - vis, len(CELLS) - n_seen
        belief.append({"radius": r, "seen": n_seen, "unseen": unseen, "hiddenWhite": hidden, "visibleWhite": vis,
                       "beliefSize": math.comb(unseen, hidden), "formula": f"C({unseen}, {hidden})"})

    pz = Puzzle(B0, W0, 1)
    data = {"board": {"b": names(B0), "w": names(W0)}, "st": st(B0, W0), "hF": pz.hF(pz.s0), "U": base,
            "moves": {"n": len(moves), "quiet": len(quiet), "pushes": len(pushes),
                      "pushOffs": [move_name(x[0]) for x in push_offs], "quietReversible": reversible},
            "frozenPlans": frozen,
            "afterCapture": {"st": st(B1, W1), "replies": len(replies), "keep": len(replies) - len(breakers),
                             "breakers": breakers, "breakerBoards": boards},
            "andor": andor, "recheck": recheck, "greedy": greedy, "counterfactual": counterfactual, "belief": belief,
            "backToOpening": back_to_opening()}
    print(json.dumps({k: v for k, v in data.items() if k not in ("board", "afterCapture")}, ensure_ascii=False))
    save("test", data)


def back_to_opening():
    """Number of paths that return to the opening after four plies (Black, White, Black, White)."""
    n = 0
    for _, _, B1, W1, _ in successors(START_B, START_W):
        for _, _, W2, B2, _ in successors(W1, B1):
            for _, _, B3, W3, _ in successors(B2, W2):
                if B3 != START_B:                     # White cannot return in one move unless Black is back home
                    continue
                n += sum(1 for _, _, W4, B4, _ in successors(W3, B3) if (B4, W4) == (START_B, START_W))
    return n


# ------------------------------------------------------------------ opening puzzle: hill climbing
def neighbours(pz, s):
    return [(mv, (P2, O2), pz.hF((P2, O2))) for mv, _, P2, O2, _ in successors(*s)]


def cmd_opening():
    """Slides 5–6: White frozen at the opening, push one white off. Steepest descent on hF stops on a plateau; climbing U stops on a local peak."""
    pz = Puzzle(START_B, START_W, 1)
    s = pz.s0
    ref = best_first(pz, "hF")
    trace, nb = [pz.hF(s)], []
    while True:                                       # steepest descent: best neighbour, ties by generator order
        h = pz.hF(s)
        kids = neighbours(pz, s)
        nb.append({"h": h, "n": len(kids), "lo": sum(k[2] < h for k in kids), "eq": sum(k[2] == h for k in kids),
                   "hi": sum(k[2] > h for k in kids)})
        best = min(kids, key=lambda k: k[2])
        if pz.is_goal(s) or best[2] >= h:
            break
        s = best[1]
        trace.append(best[2])
    h = pz.hF(s)
    pairs, down = 0, Counter()
    for _, s2, h2 in neighbours(pz, s):               # two moves together: which pairs go down?
        for _, s3, h3 in neighbours(pz, s2):
            pairs += 1
            if h3 < h:
                down[f"{h2} → {h3}"] += 1
    stuck = Puzzle(*s, 1)
    steepest = {"trace": trace, "nb": nb, "stuck": st(*s), "stuckH": h, "pairs": pairs, "pairsDown": sum(down.values()),
                "downVia": dict(down), "distStuck": best_first(stuck, "hF")["len"],
                "exit": flat_exit(s, pz.hF, h, higher=False)}

    climb = climb_u(START_B, START_W)
    top = Puzzle(*climb.pop("s"), 1)
    climb.update(hFTop=top.hF(top.s0), distTop=best_first(top, "hF")["len"])
    data = {"st": st(START_B, START_W), "hF": pz.hF(pz.s0), "U": U(START_B, START_W),
            "astar": {"d": ref["len"], "gen": ref["gen"]}, "steepest": steepest, "climbU": climb}
    print(json.dumps(data, ensure_ascii=False))
    save("opening", data)


def climb_u(B, W):
    """Steepest ascent on U (White frozen): best neighbour, ties by generator order; stop when none is better."""
    s, trace, captures = (B, W), [U(B, W)], 0
    while True:
        kids = [((P2, O2), U(P2, O2), d) for _, _, P2, O2, d in successors(*s)]
        best = max(kids, key=lambda k: k[1])
        if best[1] <= trace[-1]:
            break
        s, captures = best[0], captures + best[2]
        trace.append(best[1])
    u = trace[-1]
    f0, f = u_terms(B, W), u_terms(*s)
    exit_ = flat_exit(s, lambda x: U(*x), u, higher=True) if any(k[1] == u for k in kids) else None
    return {"trace": trace, "moves": len(trace) - 1, "stop": "flat top" if any(k[1] == u for k in kids) else "true peak",
            "exit": exit_,
            "n": len(kids), "higher": 0, "equal": sum(k[1] == u for k in kids), "lower": sum(k[1] < u for k in kids),
            "captures": captures, "f0": {k: f0[k] for k in ("centre", "coh", "edge")},
            "f": {k: f[k] for k in ("centre", "coh", "edge")}, "top": st(*s),
            "terms": {"centre": 2 * (f["centre"] - f0["centre"]), "edge": -3 * (f["edge"] - f0["edge"]),
                      "pairs": f["coh"] - f0["coh"], "off": 10 * (f["off"] - f0["off"])}, "s": s}


def flat_exit(s, score, v, higher, cap=500):
    """Exit from a flat region (equal-scoring neighbours): how many equal moves to the first board with a better neighbour?

    Returns the step count if found (never 0, since s itself has no better neighbour), None if the region exceeds cap, -1 if there is no exit.
    With an exit the plateau is a shoulder; without one it is a flat local optimum (Section 4.1.1).
    """
    better = (lambda x: x > v) if higher else (lambda x: x < v)
    seen, layer, d = {s}, [s], 0
    while layer:
        nxt = []
        for x in layer:                               # the whole layer is checked; the cap only cuts the next layer
            for c in kids(x):
                sc = score(c)
                if better(sc):
                    return d
                if sc == v and c not in seen:
                    seen.add(c)
                    nxt.append(c)
        if len(seen) > cap:
            return None
        layer, d = nxt, d + 1
    return -1


# ------------------------------------------------------------------ yerel arama: 200 bulmaca
class Run:
    """One local-search run: minimise hF (lower is better), goal hF = 0. evals: number of boards evaluated."""

    def __init__(self, pz, cache):
        self.pz, self.cache, self.evals, self.moves, self.visited, self.revisits = pz, cache, 0, 0, set(), 0

    def h(self, s):
        self.evals += 1
        v = self.cache.get(s)
        if v is None:
            v = self.cache[s] = self.pz.hF(s)
        return v

    def step(self, s):
        self.moves += 1
        if s in self.visited:
            self.revisits += 1
        self.visited.add(s)
        return s

    def done(self, s, stop):
        return {"solved": self.pz.is_goal(s), "moves": self.moves, "evals": self.evals, "stop": stop,
                "revisits": self.revisits, "s": s}


def kids(s):
    return [(P2, O2) for _, _, P2, O2, _ in successors(*s)]


def steepest(pz, cache, rng=None):
    """Figure 4.2: best neighbour (ties by generator order); stop when none is better."""
    r, s = Run(pz, cache), pz.s0
    while not pz.is_goal(s):
        h = r.h(s)
        best, bh = None, h
        for c in kids(s):
            hc = r.h(c)
            if hc < bh:
                best, bh = c, hc
        if best is None:
            return r.done(s, "plateau")
        s = r.step(best)
    return r.done(s, "goal")


def sideways(pz, cache, rng, max_side=100, cap=1000):
    """Best neighbour, random among equals; at most 100 sideways moves in a row, at most 1,000 moves in total."""
    r, s, side = Run(pz, cache), pz.s0, 0
    while not pz.is_goal(s):
        if r.moves >= cap:
            return r.done(s, "cap")
        h = r.h(s)
        scored = [(r.h(c), c) for c in kids(s)]
        bh = min(x[0] for x in scored)
        if bh > h:
            return r.done(s, "local-min")
        if bh == h:
            if side >= max_side:
                return r.done(s, "sideways-limit")
            side += 1
        else:
            side = 0
        s = r.step(rng.choice([c for v, c in scored if v == bh]))
    return r.done(s, "goal")


def stochastic(pz, cache, rng):
    """Stochastic hill climbing: uniformly among the better neighbours."""
    r, s = Run(pz, cache), pz.s0
    while not pz.is_goal(s):
        h = r.h(s)
        better = [c for c in kids(s) if r.h(c) < h]
        if not better:
            return r.done(s, "plateau")
        s = r.step(rng.choice(better))
    return r.done(s, "goal")


def first_choice(pz, cache, rng):
    """First choice: shuffle the successors, take the first better one."""
    r, s = Run(pz, cache), pz.s0
    while not pz.is_goal(s):
        h = r.h(s)
        order = kids(s)
        rng.shuffle(order)
        nxt = next((c for c in order if r.h(c) < h), None)
        if nxt is None:
            return r.done(s, "stuck")
        s = r.step(nxt)
    return r.done(s, "goal")


def annealing(pz, cache, rng, steps=1000, T0=2.0, alpha=0.995):
    """Figure 4.5: random successor; take it if better, otherwise with probability e^(ΔE/T). T = 2 · 0.995^t."""
    r, s = Run(pz, cache), pz.s0
    h = r.h(s)
    for t in range(steps):
        if pz.is_goal(s):
            return r.done(s, "goal")
        c = rng.choice(kids(s))
        hc = r.h(c)
        dE = h - hc                                   # improvement is positive
        if dE > 0 or rng.random() < math.exp(dE / (T0 * alpha ** t)):
            s, h = r.step(c), hc
    return r.done(s, "goal" if pz.is_goal(s) else "cooled")


def lrta(pz, cache, rng=None, cap=1000):
    """Figure 4.24 with a known model: H(s) ← min (1 + H(s')); H starts as hF; move to the cheapest neighbour."""
    r, s, H = Run(pz, cache), pz.s0, {}

    def Hs(x):
        if x not in H:
            H[x] = r.h(x)
        return H[x]

    while not pz.is_goal(s):
        if r.moves >= cap:
            return r.done(s, "cap")
        costs = [(1 + Hs(c), c) for c in kids(s)]
        best = min(costs, key=lambda x: x[0])
        H[s] = best[0]
        s = r.step(best[1])
    return r.done(s, "goal")


METHODS = [("steepest", steepest), ("sideways", sideways), ("stochastic", stochastic), ("firstChoice", first_choice),
           ("annealing", annealing), ("lrta", lrta)]


def stuck_profile(pz, s, cache):
    """The board it got stuck on: how many neighbours are better / equal / worse; is there a way down in two moves, and is its first move equal or worse?"""
    h = pz.hF(s)
    first = [(c, cache.get(c) if c in cache else pz.hF(c)) for c in kids(s)]
    side_first = worse_first = False
    for c, hc in first:
        if any(pz.hF(d) < h for d in kids(c)):
            if hc == h:
                side_first = True
            elif hc > h:
                worse_first = True
    exit_ = flat_exit(s, lambda x: cache[x] if x in cache else cache.setdefault(x, pz.hF(x)), h, higher=False)
    return {"h": h, "lo": sum(x[1] < h for x in first), "eq": sum(x[1] == h for x in first),
            "hi": sum(x[1] > h for x in first), "sideFirst": side_first, "worseFirst": worse_first, "exit": exit_}


def cmd_local(n=200, seed=2027, restarts=20):
    """Slides 5–7, backup 12: random mid-game puzzles (White frozen, push one white off, A* depth 2–4)."""
    rng, rows, tries, t0 = random.Random(seed), [], 0, time.time()
    while len(rows) < n:                              # Assignment 2's puzzle rule: 12–28 random plies, no marble pushed off
        tries += 1
        B, W = random_position(rng, 2 * rng.randint(6, 14))
        pz = Puzzle(B, W, 1)
        ref = best_first(pz, "hF", limit=3_000_000)
        if ref["len"] is None or not 2 <= ref["len"] <= 4:
            continue
        rows.append((pz, ref))
        if len(rows) % 20 == 0:
            print(f"puzzles {len(rows)}/{n} ({tries} drawn) {time.time() - t0:.0f}s", flush=True)
    gen_secs = time.time() - t0
    save("puzzles", [{"st": st(*pz.s0), "d": ref["len"], "gen": ref["gen"]} for pz, ref in rows])
    res = {name: [] for name, _ in METHODS}
    stuck, restart, climbs = [], [], []
    for i, (pz, ref) in enumerate(rows):
        cache = {}
        for name, f in METHODS:
            out = f(pz, cache, random.Random(seed * 1000 + i))
            res[name].append(out)
            if name == "steepest" and not out["solved"]:
                stuck.append(stuck_profile(pz, out.pop("s"), cache))
            out.pop("s", None)
        tries_needed = None                           # the same puzzle again with a new seed (not AIMA's random restart)
        for k in range(restarts):
            if stochastic(pz, cache, random.Random((seed * 1000 + i) * 100 + k))["solved"]:
                tries_needed = k + 1
                break
        restart.append(tries_needed)
        climbs.append(climb_u(*pz.s0))
        print(f"[{i + 1}/{n}] d={ref['len']} " + " ".join(f"{k}={'✓' if v[-1]['solved'] else '·'}{v[-1]['moves']}"
                                                         for k, v in res.items())
              + f" restart={tries_needed} {time.time() - t0:.0f}s", flush=True)

    d = [ref["len"] for _, ref in rows]
    algos = {}
    for name, _ in METHODS:
        v, ok = res[name], [x for x in res[name] if x["solved"]]
        algos[name] = {"solved": len(ok), "n": n, "meanMoves": round(sum(x["moves"] for x in ok) / max(1, len(ok)), 2),
                       "meanEvals": round(sum(x["evals"] for x in v) / n),
                       "optimal": sum(x["solved"] and x["moves"] == d[i] for i, x in enumerate(v)),
                       "stops": dict(Counter(x["stop"] for x in v))}
        if name in ("sideways", "lrta"):
            algos[name]["meanRevisits"] = round(sum(x["revisits"] for x in v) / n, 2)
    common = ["astar", "sideways", "annealing", "lrta"]
    both = [i for i in range(n) if all(res[m][i]["solved"] for m in common[1:])]
    mean = lambda xs: round(sum(xs) / max(1, len(xs)), 2)
    ok_r = [t for t in restart if t]
    ls = {"seed": seed, "n": n, "tries": tries, "genMs": round(gen_secs * 1000), "dHist": dict(sorted(Counter(map(str, d)).items())),
          "astar": {"solved": n, "meanMoves": mean(d), "meanGen": round(sum(ref["gen"] for _, ref in rows) / n)},
          "algos": algos,
          "stuck": {"n": len(stuck), "plateau": sum(x["eq"] > 0 for x in stuck), "localMin": sum(x["eq"] == 0 for x in stuck),
                    "meanEq": mean([x["eq"] for x in stuck]), "meanLo": mean([x["lo"] for x in stuck]),
                    "meanHi": mean([x["hi"] for x in stuck]),
                    "hAtStop": dict(sorted(Counter(str(x["h"]) for x in stuck).items())),
                    "twoStepEscape": sum(x["sideFirst"] or x["worseFirst"] for x in stuck),
                    "escapeSidewaysFirst": sum(x["sideFirst"] for x in stuck),
                    "escapeWorseFirstOnly": sum(x["worseFirst"] and not x["sideFirst"] for x in stuck),
                    "shoulder": sum(x["exit"] is not None and x["exit"] >= 0 for x in stuck),
                    "flatMin": sum(x["exit"] == -1 for x in stuck), "unknown": sum(x["exit"] is None for x in stuck),
                    "exitSteps": dict(sorted(Counter(str(x["exit"]) for x in stuck if x["exit"] is not None and x["exit"] >= 0).items()))},
          "restart": {"R": restarts, "solved": len(ok_r), "meanTries": mean(ok_r)},
          "climbU": {"n": n, "anyCapture": sum(c["captures"] > 0 for c in climbs),
                     "stops": {"local-max": sum(c["stop"] == "true peak" for c in climbs),
                               "plateau": sum(c["stop"] == "flat top" for c in climbs)},
                     "flatTop": {"shoulder": sum(c["exit"] is not None and c["exit"] >= 0 for c in climbs),
                                 "flatMax": sum(c["exit"] == -1 for c in climbs),
                                 "unknown": sum(c["stop"] == "flat top" and c["exit"] is None for c in climbs)},
                     "meanMoves": mean([c["moves"] for c in climbs]),
                     **{f"mean{k}{w}": mean([c[f][key] for c in climbs])
                        for k, key in (("Edge", "edge"), ("Coh", "coh"), ("Centre", "centre"))
                        for w, f in (("End", "f"), ("Start", "f0"))}},
          "commonSolved": {"methods": common, "n": len(both),
                           "meanMoves": {"astar": mean([d[i] for i in both]),
                                         **{m: mean([res[m][i]["moves"] for i in both]) for m in common[1:]}}},
          "secs": round(time.time() - t0)}
    print(json.dumps({k: v for k, v in ls.items()}, ensure_ascii=False))
    save("local", ls)


# ------------------------------------------------------------------ write into the deck
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def cmd_inject(html=None):
    """Writes results/*.json into the deck's window.A3 (same schema the deck expects)."""
    html = pathlib.Path(html) if html else DECK
    test, opening, local, recheck = (load(k) for k in ("test", "opening", "local", "recheck"))
    secs = local.pop("secs", 0)
    data = {"meta": {"python": platform.python_version(), "ranAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "labSecs": secs, "files": {p.name: sha(p) for p in (A2 / "abalone_search.py", HERE / "abalone_ch4.py")}},
            "verify": recheck, "ls": local, "ex": {"test": test, "opening": opening}}
    s = html.read_text(encoding="utf-8")
    a = s.index("window.A3 = ") + len("window.A3 = ")
    b = s.index("window.LOGO", a)
    with html.open("w", encoding="utf-8", newline="") as fh:
        fh.write(s[:a] + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n" + s[b:])
    save("a3data", data)
    print("injected into", html.name)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    args = sys.argv[2:]
    if cmd == "inject":
        cmd_inject(*args)
    else:
        {"recheck": cmd_recheck, "test": cmd_test, "opening": cmd_opening, "local": cmd_local}[cmd](*map(int, args))

"""Abalone through the lens of Chapter 3: state, ACTIONS, RESULT, ACTION-COST, IS-GOAL and search experiments.

Every number in the Assignment 2 deck comes from this file.

Usage:
  python abalone_search.py count            # legal moves at the start and in a mid-game position
  python abalone_search.py states           # state-space size
  python abalone_search.py verify [N]       # compare two independently written generators on N positions
  python abalone_search.py tree [D]         # D plies from the start: paths, distinct boards, speed
  python abalone_search.py bench [N] [SEED] # BFS, UCS, A* and greedy on frozen-opponent puzzles
  python abalone_search.py demo             # the two examples shown on the slides
  python abalone_search.py more [N]         # IDA*, IDA* + TT and weighted A* (w = 2) on the same puzzles
  python abalone_search.py inject           # results/*.json -> the DATA block in deck.html
"""
import heapq
import itertools
import json
import math
import pathlib
import random
import sys
import time
from collections import deque, Counter

ROWS = "ABCDEFGHI"
# directions ordered so that the opposite of d is (d + 3) % 6; the first three are a line's "forward" directions
DIRS = [("E", 0, 1), ("NE", 1, 1), ("NW", 1, 0), ("W", 0, -1), ("SW", -1, -1), ("SE", -1, 0)]


def ring(r, n):
    a, b = r - 4, n - 5
    return max(abs(a), abs(b), abs(b - a))


CELLS = [(r, n) for r in range(9) for n in range(1, 10) if ring(r, n) <= 4]
IDX = {c: i for i, c in enumerate(CELLS)}
NAME = [ROWS[r] + str(n) for r, n in CELLS]
RING = [ring(*c) for c in CELLS]
NB = [[IDX.get((r + dr, n + dn), -1) for _, dr, dn in DIRS] for r, n in CELLS]
DIST = [[max(abs(r2 - r1), abs(n2 - n1), abs((n2 - n1) - (r2 - r1))) for r2, n2 in CELLS] for r1, n1 in CELLS]
EDGE = [5 - g for g in RING]          # fewest steps for a marble to leave the board
NBMASK = [sum(1 << t for t in row if t >= 0) for row in NB]


def line(c, m, steps):
    """The cell `steps` cells from c in direction m; -1 if off the board."""
    for _ in range(steps):
        c = NB[c][m] if c >= 0 else -1
    return c


# the two cells the pushers must stand on to push a marble on edge cell e off in direction m (1 or 2 opponents in front)
PUSH_SPOTS = [[] for _ in CELLS]
for e in range(len(CELLS)):
    for m in range(6):
        if NB[e][m] < 0:
            back = (m + 3) % 6
            for r in (1, 2):
                a, b = line(e, back, r), line(e, back, r + 1)
                if a >= 0 and b >= 0:
                    PUSH_SPOTS[e].append((a, b))
EDGE_CELLS = [e for e in range(len(CELLS)) if PUSH_SPOTS[e]]


def mask(names):
    return sum(1 << IDX[(ROWS.index(s[0]), int(s[1:]))] for s in names)


def bits(x):
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


START_B = mask("A1 A2 A3 A4 A5 B1 B2 B3 B4 B5 B6 C3 C4 C5".split())
START_W = mask("I5 I6 I7 I8 I9 H4 H5 H6 H7 H8 H9 G5 G6 G7".split())
# the 14-vs-14 mid-game position from Assignment 1 (Week 03)
MID_B = mask("B2 B4 B5 C2 C3 C4 C5 D3 E1 E5 E6 E7 F2 F4".split())
MID_W = mask("B3 C6 D6 E8 E9 F7 G5 G6 G7 G8 H5 H6 H7 H8".split())


# ------------------------------------------------------------------ ACTIONS and RESULT
def successors(P, O):
    """Player to move P, opponent O (bitmasks). Yields (move, kind, P', O', pushed_off) for every legal move.

    Move = (marble group, direction). A group is 1–3 adjacent marbles on one line.
    Kind: 'in-line' (along the line), 'broadside' (sideways), 'sumito' (push).
    """
    occ = P | O
    for i in bits(P):
        for m in range(6):                        # single marble: one step to an empty neighbour, cannot push
            t = NB[i][m]
            if t >= 0 and not occ >> t & 1:
                yield ((i,), m), "in-line", P ^ (1 << i) ^ (1 << t), O, 0
        for d in range(3):                        # groups of 2 and 3, each once: forward from i
            j = NB[i][d]
            if j < 0 or not P >> j & 1:
                continue
            k = NB[j][d]
            groups = [(i, j)] + ([(i, j, k)] if k >= 0 and P >> k & 1 else [])
            for g in groups:
                gm = sum(1 << c for c in g)
                for m in range(6):
                    if m % 3 == d:                # along the line
                        lead, tail = (g[-1], g[0]) if m == d else (g[0], g[-1])
                        f = NB[lead][m]
                        if f < 0 or P >> f & 1:   # cannot push own marble off; own marble in front means a different group
                            continue
                        if not O >> f & 1:
                            yield (g, m), "in-line", P ^ (1 << tail) ^ (1 << f), O, 0
                            continue
                        run, x = 1, NB[f][m]      # consecutive opponent marbles in front
                        while x >= 0 and O >> x & 1 and run < len(g):
                            run, x = run + 1, NB[x][m]
                        if run >= len(g) or (x >= 0 and occ >> x & 1):
                            continue              # not outnumbered, or the cell behind is occupied
                        O2 = O ^ (1 << f)
                        if x >= 0:
                            O2 |= 1 << x
                        yield (g, m), "sumito", P ^ (1 << tail) ^ (1 << f), O2, int(x < 0)
                    else:                         # broadside: every target must be empty
                        ts = [NB[c][m] for c in g]
                        if all(t >= 0 and not occ >> t & 1 for t in ts):
                            yield (g, m), "broadside", P & ~gm | sum(1 << t for t in ts), O, 0


def move_name(mv):
    g, m = mv
    return " ".join(NAME[c] for c in g) + " → " + DIRS[m][0]


def naive_successors(P, O):
    """Second generator for cross-checking: follows the rulebook literally, with no tables or bit tricks.

    Tries every subset of 1–3 marbles and all 6 directions; returns the multiset of resulting positions.
    """
    own = [CELLS[i] for i in bits(P)]
    opp = {CELLS[i] for i in bits(O)}
    mine = set(own)
    on_board = lambda c: c in IDX
    vecs = [(dr, dn) for _, dr, dn in DIRS]
    out = Counter()
    for size in (1, 2, 3):
        for grp in itertools.combinations(own, size):
            axis = None
            if size > 1:                          # adjacent and on one line?
                rs = sorted(grp)
                steps = {(b[0] - a[0], b[1] - a[1]) for a, b in zip(rs, rs[1:])}
                if len(steps) != 1 or next(iter(steps)) not in vecs:
                    continue
                axis = next(iter(steps))
            for v in vecs:
                inline = size == 1 or v == axis or v == (-axis[0], -axis[1])
                new_own = set(mine) - set(grp)
                new_opp = set(opp)
                moved = [(c[0] + v[0], c[1] + v[1]) for c in grp]
                if not inline:
                    if all(on_board(t) and t not in mine and t not in opp for t in moved):
                        out[(mask_of(new_own | set(moved)), mask_of(new_opp))] += 1
                    continue
                front = max(moved, key=lambda t: t[0] * v[0] + t[1] * v[1])   # the cell the group moves into
                if not on_board(front) or front in mine:
                    continue
                run, cur = [], front
                while on_board(cur) and cur in opp:
                    run.append(cur)
                    cur = (cur[0] + v[0], cur[1] + v[1])
                if run:
                    if len(run) >= size or (on_board(cur) and cur in mine):
                        continue
                    for c in run:
                        new_opp.discard(c)
                    for c in run:
                        t = (c[0] + v[0], c[1] + v[1])
                        if on_board(t):
                            new_opp.add(t)
                out[(mask_of(new_own | set(moved)), mask_of(new_opp))] += 1
    return out


def mask_of(cells):
    return sum(1 << IDX[c] for c in cells)


# ------------------------------------------------------------------ search problem: frozen opponent
class Puzzle:
    """The version that fits Chapter 3's single-agent assumption: Black moves, White never moves (it is only pushed).

    State (B, W) · ACTIONS = successors · ACTION-COST = 1 · IS-GOAL: k white marbles pushed off since the start.
    """

    def __init__(self, B, W, k):
        self.s0, self.k, self.w0 = (B, W), k, W.bit_count()

    def off(self, s):
        return self.w0 - s[1].bit_count()

    def is_goal(self, s):
        return self.off(s) >= self.k

    # h1: marbles still to push off. RESULT pushes off at most one opponent marble per move.
    def h1(self, s):
        return max(0, self.k - self.off(s))

    # h2: sum of edge distances of the h1 easiest opponent marbles / 2. RESULT pushes at most two opponent
    # marbles one cell per move (3-on-2 sumito).
    def h2(self, s):
        need = self.h1(s)
        if not need:
            return 0
        return (sum(sorted(EDGE[c] for c in bits(s[1]))[:need]) + 1) // 2

    # hR: an estimate that looks only at the opponent
    def hR(self, s):
        return max(self.h1(s), self.h2(s))

    # hF (formation): before the final push the opponent marble must sit on an edge cell (e) with two of our
    # marbles lined up behind it. Relaxed problem: our marbles pass through each other and all may move one
    # cell per move; an opponent marble is pushed at most one cell per move.
    # hF = 1 + min over (opponent p, edge e) max(distance to formation, DIST(p, e)); at least hR.
    def hF(self, s):
        h = self.hR(s)
        if not h:
            return 0
        B, W = s
        nd, seen, layer, t = {}, B, B, 0         # nd: distance from each cell to our nearest marble (ignoring blockers)
        while layer:
            for c in bits(layer):
                nd[c] = t
            nxt = 0
            for c in bits(layer):
                nxt |= NBMASK[c]
            layer, seen, t = nxt & ~seen, seen | nxt, t + 1
        spot = [(e, min(max(nd[a], nd[b]) for a, b in PUSH_SPOTS[e])) for e in EDGE_CELLS]
        best = min(1 + max(t, DIST[p][e]) for p in bits(W) for e, t in spot)
        return max(h, best)

    def h0(self, s):
        return 0


def bfs(pz, limit):
    """Breadth-first graph search with the goal test on generation (as in Figure 3.9)."""
    s0 = pz.s0
    if pz.is_goal(s0):
        return {"len": 0, "gen": 1, "exp": 0}
    frontier, reached, generated, expanded = deque([s0]), {s0: 0}, 1, 0
    while frontier:
        s = frontier.popleft()
        expanded += 1
        for _, _, P2, O2, _ in successors(*s):
            c = (P2, O2)
            generated += 1
            if pz.is_goal(c):
                return {"len": reached[s] + 1, "gen": generated, "exp": expanded}
            if c not in reached:
                reached[c] = reached[s] + 1
                frontier.append(c)
        if generated > limit:
            break
    return {"len": None, "gen": generated, "exp": expanded}


def best_first(pz, hname, astar=True, limit=10**9, check=False, path=False, w=1):
    """Best-first graph search (Figure 3.7). f = g + w·h (A*; w > 1 weighted A*, h = 0 for UCS) or f = h (greedy).

    check: counts the consistency condition h(n) <= 1 + h(n') on every edge.
    """
    h = getattr(pz, hname)
    s0, tie = pz.s0, itertools.count()
    h0 = h(s0)
    frontier = [(h0, 0, next(tie), s0, h0)]
    reached, parent = {s0: 0}, {s0: None}
    generated, expanded, edges, bad = 1, 0, 0, 0
    while frontier:
        _, ng, _, s, hs = heapq.heappop(frontier)
        g = -ng
        if g > reached[s]:
            continue
        if pz.is_goal(s):
            out = {"len": g, "gen": generated, "exp": expanded, "h0": h0, "mem": len(reached)}
            if check:
                out.update(edges=edges, bad=bad)
            if path:
                seq = []
                while s is not None:
                    seq.append(s)
                    s = parent[s][0] if parent[s] else None
                out["path"] = seq[::-1]
            return out
        expanded += 1
        for mv, _, P2, O2, _ in successors(*s):
            c, g2 = (P2, O2), g + 1               # ACTION-COST(s, a, s') = 1
            generated += 1
            if c not in reached or g2 < reached[c]:
                hc = h(c)
                if check:
                    edges += 1
                    bad += hs > 1 + hc
                reached[c] = g2
                parent[c] = (s, mv)
                heapq.heappush(frontier, ((g2 + w * hc) if astar else hc, -g2, next(tie), c, hc))
        if generated > limit:
            break
    return {"len": None, "gen": generated, "exp": expanded, "h0": h0, "mem": len(reached)}


def ida_star(pz, hname, limit=2_000_000, tt=False):
    """IDA* (Korf 1985): depth-first with an f = g + h bound; each iteration raises the bound to the smallest f that exceeded it.

    Only the path is kept in memory (cycle check along the path). tt: does not re-expand a board already seen
    by a shorter path in the same iteration (transposition table, Reinefeld & Marsland 1994); memory grows with the table.
    """
    h = getattr(pz, hname)
    s0 = pz.s0
    bound, gen, iters, peak = h(s0), 1, 0, 1
    while True:
        iters += 1
        nxt, seen, on_path = math.inf, {}, {s0}

        def dfs(s, g, hs):
            nonlocal gen, nxt, peak
            if g + hs > bound:
                nxt = min(nxt, g + hs)
                return None
            if pz.is_goal(s):
                return g
            if tt:
                if seen.get(s, math.inf) <= g:
                    return None
                seen[s] = g
            for _, _, P2, O2, _ in successors(*s):
                c = (P2, O2)
                gen += 1
                if c in on_path or gen > limit:
                    continue
                on_path.add(c)
                peak = max(peak, len(on_path) + len(seen))
                r = dfs(c, g + 1, h(c))
                on_path.discard(c)
                if r is not None:
                    return r
            return None

        r = dfs(s0, 0, h(s0))
        if r is not None:
            return {"len": r, "gen": gen, "iters": iters, "mem": peak}
        if gen > limit or nxt == math.inf:
            return {"len": None, "gen": gen, "iters": iters, "mem": peak}
        bound = nxt


def eff_branching(N, d):
    """b* from N + 1 = 1 + b* + b*^2 + ... + b*^d (bisection)."""
    if d is None or d <= 0:
        return None
    lo, hi = 1.0, float(N)
    for _ in range(100):
        b = (lo + hi) / 2
        tot = (b ** (d + 1) - 1) / (b - 1) if b > 1 else d + 1
        lo, hi = (b, hi) if tot < N + 1 else (lo, b)
    return (lo + hi) / 2


# ------------------------------------------------------------------ commands
OUT = pathlib.Path(__file__).with_name("results")


def save(name, data):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_count():
    data = {}
    for key, label, P, O in [("start", "start, Black to move", START_B, START_W),
                             ("midB", "mid-game (Week 03), Black to move", MID_B, MID_W),
                             ("midW", "mid-game (Week 03), White to move", MID_W, MID_B)]:
        kinds = Counter(f"{kind}-{len(mv[0])}" for mv, kind, *_ in successors(P, O))
        data[key] = {"total": sum(kinds.values()), "kinds": dict(sorted(kinds.items()))}
        print(f"{label}: {data[key]['total']} legal moves {data[key]['kinds']}")
    data["midPushes"] = [move_name(mv) for mv, kind, *_ in successors(MID_B, MID_W) if kind == "sumito"]
    print("mid-game sumito moves:", ", ".join(data["midPushes"]))
    save("count", data)


def cmd_states():
    ours = sum(math.comb(61, b) * math.comb(61 - b, w) for b in range(8, 15) for w in range(8, 15) if b > 8 or w > 8)
    lem = sum(math.comb(61, k) * math.comb(61 - k, m) for k in range(8, 15) for m in range(9, 15))
    data = {"naive": float(3 ** 61), "ours": float(ours), "oursSide": float(2 * ours),
            "full14": float(math.comb(61, 14) * math.comb(47, 14)),
            "lemmensFormula": float(lem), "lemmensFormula12": lem / 12, "lemmensPaper": 6.5e23,
            "gameTree": float(60 ** 87)}
    for k, v in data.items():
        print(f"{k:18s} {v:.3e}")
    save("states", data)


def cmd_verify(n=2000, seed=11):
    """Do both generators give the same positions? Start, mid-game and random positions."""
    rng = random.Random(seed)
    tests = [(START_B, START_W), (MID_B, MID_W), (MID_W, MID_B)]
    while len(tests) < n:                          # random games, including pushes and pushed-off marbles
        P, O = START_B, START_W
        for _ in range(rng.randint(1, 120)):
            moves = list(successors(P, O))
            pushes = [x for x in moves if x[1] == "sumito"]
            _, _, P2, O2, _ = rng.choice(pushes if pushes and rng.random() < .5 else moves)
            P, O = O2, P2
            if O.bit_count() <= 8:
                break
        tests.append((P, O))
    same = moves_total = pushes_total = 0
    for P, O in tests:
        fast = Counter((P2, O2) for _, _, P2, O2, _ in successors(P, O))
        slow = naive_successors(P, O)
        same += fast == slow
        moves_total += sum(fast.values())
        pushes_total += sum(1 for x in successors(P, O) if x[1] == "sumito")
    print(f"{same}/{len(tests)} positions: both generators give the same multiset of resulting boards;"
          f" {moves_total} moves checked, {pushes_total} of them pushes")
    p3 = (perft_with(naive_successors, START_B, START_W, 3), perft(START_B, START_W, 3))
    print(f"perft(3) from the start: naive {p3[0]}, fast {p3[1]}")
    save("verify", {"positions": len(tests), "same": same, "moves": moves_total, "pushes": pushes_total,
                    "perft3naive": p3[0], "perft3fast": p3[1], "seed": seed})


def perft(P, O, d):
    if d == 1:
        return sum(1 for _ in successors(P, O))
    return sum(perft(O2, P2, d - 1) for _, _, P2, O2, _ in successors(P, O))


def perft_with(gen, P, O, d):
    kids = gen(P, O)
    if d == 1:
        return sum(kids.values())
    return sum(c * perft_with(gen, O2, P2, d - 1) for (P2, O2), c in kids.items())


def cmd_tree(depth=4):
    """Tree (perft) and graph (distinct position) size from the start, speed, cycles."""
    data = {"perft": [], "distinct": [], "reached": [], "secs": []}
    layer, reached = {(START_B, START_W): 1}, {(START_B, START_W)}
    back_to_start = 0
    for d in range(1, depth + 1):
        t0 = time.time()
        nxt, leaves = Counter(), 0
        for (P, O), paths in layer.items():
            for _, _, P2, O2, _ in successors(P, O):
                nxt[(O2, P2)] += paths            # the turn passes to the other side
                leaves += paths
        secs = time.time() - t0
        if d % 2 == 0:
            back_to_start = nxt.get((START_B, START_W), 0)
        reached |= nxt.keys()
        data["perft"].append(leaves)
        data["distinct"].append(len(nxt))
        data["reached"].append(len(reached))
        data["secs"].append(round(secs, 2))
        print(f"ply {d}: {leaves:,} paths, {len(nxt):,} distinct boards ({leaves / len(nxt):.1f} paths/board),"
              f" {len(reached):,} reached so far, {secs:.1f}s; paths back to the start: {nxt.get((START_B, START_W), 0)}",
              flush=True)
        layer = nxt
    # speed: nodes generated per second on one core
    t0, gen = time.time(), 0
    for (P, O) in list(layer)[:20000]:
        gen += sum(1 for _ in successors(P, O))
    rate = gen / (time.time() - t0)
    P = data["perft"]
    bfs_nodes = sum(P)
    ids_nodes = sum(sum(P[:l]) for l in range(1, depth + 1))
    data.update(rate=round(rate), backToStart=back_to_start, idsOverPct=round((ids_nodes / bfs_nodes - 1) * 100, 2),
                bytesPerState=16, depth=depth)
    print(f"{rate:,.0f} nodes/s; IDS generates {data['idsOverPct']}% more nodes than BFS up to ply {depth}")
    save("tree", data)


def random_position(rng, plies):
    """A position reached by random play from the start; Black to move, no marble pushed off."""
    P, O = START_B, START_W
    for _ in range(plies):
        moves = [x for x in successors(P, O) if not x[4]]
        _, _, P2, O2, _ = rng.choice(moves)
        P, O = O2, P2
    return P, O


ALGOS = [("BFS", None, None), ("UCS", "h0", True), ("A* hR", "hR", True), ("A* hF", "hF", True), ("Greedy hF", "hF", False)]


def cmd_bench(n=60, seed=314, limit=1_000_000, dmin=2, dmax=5):
    """Frozen-opponent puzzles: push one white marble off from a random mid-game position (k = 1)."""
    rng = random.Random(seed)
    rows, tried, t0 = [], 0, time.time()
    while len(rows) < n:
        tried += 1
        B, W = random_position(rng, 2 * rng.randint(6, 14))
        pz = Puzzle(B, W, 1)
        ref = best_first(pz, "hF", limit=3 * limit, check=True)    # shortest solution (hF is admissible)
        d = ref["len"]
        if d is None or not dmin <= d <= dmax:
            continue
        res = {}
        for name, h, astar in ALGOS:
            res[name] = bfs(pz, limit) if h is None else (ref if name == "A* hF" else best_first(pz, h, astar, limit))
        rows.append({"d": d, "h0": ref["h0"], "edges": ref["edges"], "bad": ref["bad"],
                     "res": {k: {x: v[x] for x in ("len", "gen", "exp")} for k, v in res.items()}})
        print(f"[{len(rows)}/{n}] d={d} hF(s0)={ref['h0']} " +
              " ".join(f"{k}={v['gen']}{'+' if v['len'] is None else ''}"
                       f"{'' if v['len'] in (None, d) else '(len ' + str(v['len']) + ')'}"
                       for k, v in res.items()) + f" {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for name, *_ in ALGOS:
        done = [r for r in rows if r["res"][name]["len"] is not None]
        summary[name] = {"solved": len(done), "optimal": sum(r["res"][name]["len"] == r["d"] for r in done),
                         "meanLen": round(sum(r["res"][name]["len"] for r in done) / max(1, len(done)), 2)}
    table = []
    for d in sorted({r["d"] for r in rows}):
        grp = [r for r in rows if r["d"] == d]
        row = {"d": d, "n": len(grp)}
        for name, *_ in ALGOS:
            gens = [r["res"][name]["gen"] for r in grp]
            capped = sum(r["res"][name]["len"] is None for r in grp)
            mean = sum(gens) / len(gens)
            row[name] = {"gen": round(mean), "capped": capped, "bstar": round(eff_branching(mean, d), 2)}
        table.append(row)
        print(f"d={d} n={len(grp)} | " + " | ".join(f"{k}: {v['gen']:,}{'+' if v['capped'] else ''} b*={v['bstar']}"
                                                   for k, v in row.items() if isinstance(v, dict)))
    adm = sum(r["h0"] <= r["d"] for r in rows)
    edges, bad = sum(r["edges"] for r in rows), sum(r["bad"] for r in rows)
    print(f"admissible at the start: {adm}/{len(rows)}; consistency violations: {bad} of {edges} edges")
    print("summary:", summary)
    save("bench", {"n": n, "tried": tried, "seed": seed, "limit": limit, "rows": rows, "table": table, "summary": summary,
                   "admissible": adm, "edges": edges, "bad": bad, "secs": round(time.time() - t0)})


MORE = [("A* hF", lambda pz, L: best_first(pz, "hF", limit=L)),
        ("IDA* hF", lambda pz, L: ida_star(pz, "hF", L)),
        ("IDA* hF + TT", lambda pz, L: ida_star(pz, "hF", L, tt=True)),
        ("Weighted A* hF, w = 2", lambda pz, L: best_first(pz, "hF", limit=L, w=2))]


def cmd_more(n=60, seed=314, limit=2_000_000):
    """The same 60 puzzles as bench: IDA*, IDA* + transposition table and weighted A* (w = 2) against A* hF."""
    rng = random.Random(seed)
    rows, t0 = [], time.time()
    while len(rows) < n:
        B, W = random_position(rng, 2 * rng.randint(6, 14))
        pz = Puzzle(B, W, 1)
        d = best_first(pz, "hF", limit=3 * 1_000_000)["len"]      # same order and same filter as bench
        if d is None or not 2 <= d <= 5:
            continue
        res = {name: f(pz, limit) for name, f in MORE}
        rows.append({"d": d, "res": {k: {x: v.get(x) for x in ("len", "gen", "mem", "iters")} for k, v in res.items()}})
        print(f"[{len(rows)}/{n}] d={d} " + " ".join(f"{k}: gen={v['gen']} mem={v['mem']} len={v['len']}" for k, v in res.items())
              + f" {time.time() - t0:.0f}s", flush=True)
    table = []
    for d in sorted({r["d"] for r in rows}):
        grp = [r for r in rows if r["d"] == d]
        row = {"d": d, "n": len(grp)}
        for name, _ in MORE:
            v = [r["res"][name] for r in grp]
            row[name] = {"gen": round(sum(x["gen"] for x in v) / len(v)), "mem": round(sum(x["mem"] for x in v) / len(v)),
                         "capped": sum(x["len"] is None for x in v), "optimal": sum(x["len"] == d for x in v),
                         "meanLen": round(sum(x["len"] or 0 for x in v) / len(v), 2)}
        table.append(row)
        print(f"d={d} n={len(grp)} | " + " | ".join(f"{k}: gen {v['gen']:,} mem {v['mem']:,} opt {v['optimal']}/{len(grp)}{' cut ' + str(v['capped']) if v['capped'] else ''}"
                                                   for k, v in row.items() if isinstance(v, dict)))
    summary = {name: {"optimal": sum(r["res"][name]["len"] == r["d"] for r in rows),
                      "solved": sum(r["res"][name]["len"] is not None for r in rows),
                      "worst": max((r["res"][name]["len"] or 0) - r["d"] for r in rows)} for name, _ in MORE}
    print("summary:", summary)
    save("more", {"n": n, "seed": seed, "limit": limit, "table": table, "summary": summary, "secs": round(time.time() - t0)})


def path_moves(path):
    """Recovers move names from the states along a path."""
    out = []
    for s, t in zip(path, path[1:]):
        out.append(next(move_name(mv) for mv, _, P2, O2, _ in successors(*s) if (P2, O2) == t))
    return out


def cmd_demo():
    """The two examples on the slides: the plan the opponent breaks, and the heuristic example."""
    # 1) push two white marbles off from the mid-game position; which White replies after the first move break the plan?
    pz = Puzzle(MID_B, MID_W, 2)
    r = best_first(pz, "hF", path=True)
    plan = r["path"]
    rest = [next(mv for mv, _, P2, O2, _ in successors(*s) if (P2, O2) == t) for s, t in zip(plan[1:], plan[2:])]
    B1, W1 = plan[1]
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
    demo = {"plan": {"moves": path_moves(plan), "replies": len(replies), "breaks": len(breakers), "breakers": breakers}}
    print(demo["plan"])
    # 2) heuristic example: the first random puzzle with d = 3 and hF = 3
    rng = random.Random(2026)
    while True:
        B, W = random_position(rng, 2 * rng.randint(6, 14))
        pz = Puzzle(B, W, 1)
        r = best_first(pz, "hF", limit=400_000, path=True)
        if r["len"] == 3 and pz.hF(pz.s0) == 3:
            break
    runs = {"BFS": bfs(pz, 5_000_000), "UCS": best_first(pz, "h0", limit=5_000_000),
            "A* hR": best_first(pz, "hR", limit=5_000_000), "A* hF": r,
            "Greedy hF": best_first(pz, "hF", astar=False, limit=5_000_000)}
    demo["puzzle"] = {"B": [NAME[c] for c in bits(B)], "W": [NAME[c] for c in bits(W)], "path": path_moves(r["path"]),
                      "d": r["len"], "hF": pz.hF(pz.s0), "hR": pz.hR(pz.s0),
                      "runs": {k: {"gen": v["gen"], "len": v["len"]} for k, v in runs.items()}}
    print(demo["puzzle"])
    save("demo", demo)


def cmd_inject():
    """Writes results/*.json into the deck's DATA block."""
    data = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json"))}
    data.get("bench", {}).pop("rows", None)
    html = pathlib.Path(__file__).with_name("deck.html")
    s = html.read_text(encoding="utf-8")
    a, b = s.index("/*DATA*/") + len("/*DATA*/"), s.index("/*END*/")
    with html.open("w", encoding="utf-8", newline="") as fh:
        fh.write(s[:a] + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + s[b:])
    print("injected:", ", ".join(data))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "count"
    args = [int(a) for a in sys.argv[2:]]
    {"count": cmd_count, "states": cmd_states, "verify": cmd_verify, "tree": cmd_tree, "bench": cmd_bench,
     "demo": cmd_demo, "inject": cmd_inject, "more": cmd_more}[cmd](*args)

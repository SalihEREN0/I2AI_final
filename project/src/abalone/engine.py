"""Rules engine: board geometry, the two move generators and positions.

Ported unchanged in behaviour from Assignment 2 (assignments/a2-search/abalone_search.py), where the two
independently written generators were shown to agree on 2,000 positions. Past assignment code stays frozen,
so the final agent carries its own copy; tests/test_engine.py re-checks perft and the cross-check here.

A position is a pair of bitmasks (P, O): P = the side to move, O = the opponent. Bit i is cell CELLS[i].
"""
import itertools
from collections import Counter

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
NBMASK = [sum(1 << t for t in row if t >= 0) for row in NB]
RING_MASK = [sum(1 << i for i, g in enumerate(RING) if g == k) for k in range(5)]
EDGE_MASK = RING_MASK[4]
FULL = (1 << len(CELLS)) - 1
WIN_COUNT = 8                       # a side with 8 marbles left has lost 6: the game is over


def bits(x):
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


def mask(names):
    return sum(1 << IDX[(ROWS.index(s[0]), int(s[1:]))] for s in names)


def names(m):
    return [NAME[c] for c in bits(m)]


START = {
    "standard": (mask("A1 A2 A3 A4 A5 B1 B2 B3 B4 B5 B6 C3 C4 C5".split()),
                 mask("I5 I6 I7 I8 I9 H4 H5 H6 H7 H8 H9 G5 G6 G7".split())),
    "belgian-daisy": (mask("A1 A2 B1 B2 B3 C2 C3 I8 I9 H7 H8 H9 G7 G8".split()),
                      mask("A4 A5 B4 B5 B6 C5 C6 I5 I6 H4 H5 H6 G4 G5".split())),
}


# ------------------------------------------------------------------ ACTIONS and RESULT
def successors_plain(P, O):
    """The move generator as written for Assignment 2, kept as the reference for successors() below.

    Player to move P, opponent O (bitmasks). Yields (move, kind, P', O', pushed_off) for every legal move.
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


# The same generator with everything that does not depend on the position worked out once: for each cell its
# single steps, and for each group starting there its in-line and broadside moves with their bit masks and the
# cells a push would run into. successors() yields exactly what successors_plain() yields, in the same order
# (tests/test_speed.py): the search breaks ties between equally ordered moves by this order and the random
# openings draw from it, so the same order keeps every result reproducible.
BIT = [1 << c for c in range(len(CELLS))]


def _tables():
    single, groups, pushes = [], [], []
    for i in range(len(CELLS)):
        single.append(tuple((((i,), m), BIT[t]) for m, t in enumerate(NB[i]) if t >= 0))
        gs, ps = [], []
        for d in range(3):
            j = NB[i][d]
            if j < 0:
                continue
            k = NB[j][d]
            per_size, push_size = [], []
            for g in ((i, j), (i, j, k)) if k >= 0 else ((i, j),):
                moves, push = [], []
                for m in range(6):
                    if m % 3 == d:                                   # along the line
                        lead, tail = (g[-1], g[0]) if m == d else (g[0], g[-1])
                        f = NB[lead][m]
                        if f < 0:
                            continue                                 # would push our own marble off
                        chain, x = [], f                             # the cells beyond f, for a push
                        for _ in range(3):
                            x = NB[x][m] if x >= 0 else -1
                            chain.append(x)
                        moves.append((True, (g, m), BIT[tail], BIT[f], tuple(chain), len(g)))
                        # a push-off: f and the next opponent marbles, then the edge, fewer than the group
                        for q in range(len(g) - 1):
                            if chain[q] < 0:
                                need = BIT[f] | sum(BIT[c] for c in chain[:q])
                                push.append(((g, m), BIT[tail], BIT[f], need))
                                break
                    else:                                            # broadside: every target on the board
                        ts = [NB[c][m] for c in g]
                        if all(t >= 0 for t in ts):
                            moves.append((False, (g, m), sum(BIT[c] for c in g), sum(BIT[t] for t in ts), None, 0))
                per_size.append(tuple(moves))
                push_size.append(tuple(push))
            kbit = BIT[k] if k >= 0 else 0
            gs.append((BIT[j], kbit, per_size[0], per_size[1] if k >= 0 else ()))
            ps.append((BIT[j], kbit, push_size[0], push_size[1] if k >= 0 else ()))
        groups.append(tuple(gs))
        pushes.append(tuple(ps))
    return single, groups, pushes


SINGLE, GROUPS, PUSHES = _tables()


def successors(P, O):
    """Player to move P, opponent O (bitmasks). Yields (move, kind, P', O', pushed_off) for every legal move.

    Move = (marble group, direction). A group is 1–3 adjacent marbles on one line.
    Kind: 'in-line' (along the line), 'broadside' (sideways), 'sumito' (push).
    """
    occ = P | O
    rest = P
    while rest:
        low = rest & -rest
        rest ^= low
        i = low.bit_length() - 1
        for mv, tb in SINGLE[i]:                  # single marble: one step to an empty neighbour, cannot push
            if not occ & tb:
                yield mv, "in-line", P ^ low ^ tb, O, 0
        for jb, kb, two, three in GROUPS[i]:     # groups of 2 and 3, each once: forward from i
            if not P & jb:
                continue
            for moves in (two, three) if kb and P & kb else (two,):
                for inline, mv, a, b, chain, size in moves:
                    if inline:                    # a = tail bit, b = bit of the cell in front
                        if P & b:
                            continue              # own marble in front means a different group
                        if not O & b:
                            yield mv, "in-line", P ^ a ^ b, O, 0
                            continue
                        run, x = 1, chain[0]      # consecutive opponent marbles in front
                        while x >= 0 and O >> x & 1 and run < size:
                            x = chain[run]
                            run += 1
                        if run >= size or (x >= 0 and occ >> x & 1):
                            continue              # not outnumbered, or the cell behind is occupied
                        yield mv, "sumito", P ^ a ^ b, O ^ b | (BIT[x] if x >= 0 else 0), int(x < 0)
                    elif not occ & b:             # broadside: a = group mask, b = target mask
                        yield mv, "broadside", P & ~a | b, O, 0


def pushoffs(P, O):
    """Only the moves that push a marble off, as (move, P', O'), in the order successors() yields them.

    Quiescence search needs nothing else; generating all ~60 moves to keep a few was most of its cost.
    """
    rest = P
    while rest:
        low = rest & -rest
        rest ^= low
        for jb, kb, two, three in PUSHES[low.bit_length() - 1]:
            if not P & jb:
                continue
            for moves in (two, three) if kb and P & kb else (two,):
                for mv, tail, front, need in moves:
                    if O & need == need:
                        yield mv, P ^ tail ^ front, O ^ front


def move_name(mv):
    g, m = mv
    return " ".join(NAME[c] for c in g) + " " + DIRS[m][0]


def parse_move(text):
    """Inverse of move_name: 'C3 C4 NE' -> ((i, j), 1)."""
    *cells, d = text.split()
    return tuple(IDX[(ROWS.index(c[0]), int(c[1:]))] for c in cells), [x[0] for x in DIRS].index(d)


def apply(P, O, mv):
    """RESULT for one move given as (group, direction). Returns (P', O', pushed_off); raises if illegal."""
    for m, _, P2, O2, off in successors(P, O):
        if m == mv:
            return P2, O2, off
    raise ValueError("illegal move " + move_name(mv))


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


def can_push_off(P, O):
    """Cheap prefilter for quiescence: could P possibly push an O marble off? (no false negatives)

    A push-off needs an opponent marble on the edge ring and one of P's marbles touching it, or touching
    the opponent marble next to it (the 3-vs-2 push).
    """
    for c in bits(O & EDGE_MASK):
        near = NBMASK[c]
        if near & P:
            return True
        for o in bits(near & O):
            if NBMASK[o] & P:
                return True
    return False


def perft(P, O, d):
    if d == 0:
        return 1
    return sum(perft(O2, P2, d - 1) for _, _, P2, O2, _ in successors(P, O))


def ascii_board(B, W, marks=()):
    """Text board, row I at the top. B = 'x', W = 'o', empty = '.', cells in `marks` = '*'."""
    out = []
    for r in range(8, -1, -1):
        cells = [i for i, (rr, _) in enumerate(CELLS) if rr == r]
        row = []
        for i in cells:
            row.append("*" if i in marks else "x" if B >> i & 1 else "o" if W >> i & 1 else ".")
        out.append(" " * abs(r - 4) + ROWS[r] + " " + " ".join(row))
    return "\n".join(out)

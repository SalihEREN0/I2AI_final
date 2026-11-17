"""Evaluation features and the linear evaluation U(s) = Σ wᵢ·fᵢ(s).

Every feature is a difference "side to move minus opponent", so f(O, P) = −f(P, O) and the evaluation is
zero-sum, as negamax needs. Features are divided by a fixed scale so that weights of different features are
comparable and one learning rate works for all of them in TDLeaf(λ).

Sources for the features:
  marbles, centre, cohesion, edge   — our utility U from Assignment 1; centre and cohesion are also the two
                                      features of ABLA (Özcan & Hülagu 2004) and Lemmens (2005)
  exposed                           — own edge marbles touching an opponent marble: a cheap stand-in for
                                      "marbles in danger", the weakness Lemmens reports for his agent
  wedge                             — Lemmens' "break strong group": a marble with opponents on both sides
                                      along one line, splitting their group
"""
from .engine import NBMASK, NB, RING_MASK, EDGE_MASK, bits

FEATURES = ["marbles", "centre", "cohesion", "edge", "exposed", "wedge"]
SCALE = {"marbles": 1.0, "centre": 10.0, "cohesion": 10.0, "edge": 5.0, "exposed": 5.0, "wedge": 3.0}

# Assignment 1's U = 10·off + 2·centre + cohesion − 3·edge, rewritten as zero-sum differences and rescaled;
# the two new features start at 0, so HAND is exactly A1's judgement and learning starts from it
HAND = {"marbles": 10.0, "centre": 2.0 * 10, "cohesion": 1.0 * 10, "edge": -3.0 * 5, "exposed": 0.0, "wedge": 0.0}


def _centre(X):
    return sum((4 - k) * (X & RING_MASK[k]).bit_count() for k in range(4))


def _pairs(X):
    return sum((NBMASK[c] & X).bit_count() for c in bits(X)) // 2


def _exposed(X, Y):
    return sum(1 for c in bits(X & EDGE_MASK) if NBMASK[c] & Y)


def _wedge(X, Y):
    n = 0
    for c in bits(X):
        row = NB[c]
        for d in range(3):
            a, b = row[d], row[d + 3]
            if a >= 0 and b >= 0 and Y >> a & 1 and Y >> b & 1:
                n += 1
    return n


def raw_plain(P, O):
    """The features one marble at a time, as written for Assignment 4: the reference for raw_tuple() below."""
    return {
        "marbles": P.bit_count() - O.bit_count(),
        "centre": _centre(P) - _centre(O),
        "cohesion": _pairs(P) - _pairs(O),
        "edge": (P & EDGE_MASK).bit_count() - (O & EDGE_MASK).bit_count(),
        "exposed": _exposed(P, O) - _exposed(O, P),
        "wedge": _wedge(P, O) - _wedge(O, P),
    }


# ------------------------------------------------------------------ the same features, all marbles at once
# Moving every marble of X one cell in direction m is a handful of shifts: within a row a neighbour is one index
# away, and between two rows it is a fixed number of indices away (the rows have different lengths, so the
# number depends on the row; rows with the same number share one shift). With X shifted six ways, each feature
# is a few AND and popcount operations, and the evaluation no longer loops over marbles. raw_tuple() returns
# exactly the integers of raw_plain() (tests/test_speed.py), so the search sees the same values.
def _shift_source():
    terms = []
    for m in range(6):
        by_offset = {}
        for c, row in enumerate(NB):
            if row[m] >= 0:
                by_offset[row[m] - c] = by_offset.get(row[m] - c, 0) | 1 << c
        terms.append(" | ".join("((X & %d) << %d)" % (mask, o) if o > 0 else "((X & %d) >> %d)" % (mask, -o)
                                for o, mask in sorted(by_offset.items())))
    return "def shifts(X):\n    return (%s)\n" % ",\n            ".join(terms)


_ns = {}
exec(_shift_source(), _ns)
shifts = _ns["shifts"]      # shifts(X) = X moved E, NE, NW, W, SW, SE (marbles leaving the board dropped)
SCALES = [SCALE[k] for k in FEATURES]
_R0, _R1, _R2, _R3 = RING_MASK[:4]


def raw_tuple(P, O):
    """The six unscaled features of raw_plain(), in FEATURES order, computed for all marbles at once."""
    pE, pNE, pNW, pW, pSW, pSE = shifts(P)
    oE, oNE, oNW, oW, oSW, oSE = shifts(O)
    pe, oe = P & EDGE_MASK, O & EDGE_MASK
    return (P.bit_count() - O.bit_count(),
            4 * (P & _R0).bit_count() + 3 * (P & _R1).bit_count() + 2 * (P & _R2).bit_count() + (P & _R3).bit_count()
            - 4 * (O & _R0).bit_count() - 3 * (O & _R1).bit_count() - 2 * (O & _R2).bit_count() - (O & _R3).bit_count(),
            # a pair is a marble whose neighbour in a forward direction (E, NE, NW) is also ours
            (P & pE).bit_count() + (P & pNE).bit_count() + (P & pNW).bit_count()
            - (O & oE).bit_count() - (O & oNE).bit_count() - (O & oNW).bit_count(),
            pe.bit_count() - oe.bit_count(),
            # exposed: an edge marble on a cell next to an opposing marble
            (pe & (oE | oNE | oNW | oW | oSW | oSE)).bit_count() - (oe & (pE | pNE | pNW | pW | pSW | pSE)).bit_count(),
            # wedge: opposing marbles on both sides of a marble along one line, counted per line
            (P & oW & oE).bit_count() + (P & oSW & oNE).bit_count() + (P & oSE & oNW).bit_count()
            - (O & pW & pE).bit_count() - (O & pSW & pNE).bit_count() - (O & pSE & pNW).bit_count())


def raw(P, O):
    """Unscaled feature differences, for display."""
    return dict(zip(FEATURES, raw_tuple(P, O)))


def vector(P, O):
    return [r / s for r, s in zip(raw_tuple(P, O), SCALES)]


def make_evaluator(wl):
    """evaluate(·, ·, wl) for fixed weights: the same six products, summed by sum() as evaluate() does (since
    Python 3.12 sum() compensates float rounding, so plain left-to-right addition can differ in the last bit)."""
    w0, w1, w2, w3, w4, w5 = wl
    s0, s1, s2, s3, s4, s5 = SCALES

    def ev(P, O):
        r0, r1, r2, r3, r4, r5 = raw_tuple(P, O)
        return sum((w0 * (r0 / s0), w1 * (r1 / s1), w2 * (r2 / s2), w3 * (r3 / s3), w4 * (r4 / s4), w5 * (r5 / s5)))
    return ev


def weights_list(w):
    return [float(w.get(k, 0.0)) for k in FEATURES]


def evaluate(P, O, wl):
    """Static evaluation from P's point of view; wl = weights as a list in FEATURES order."""
    return sum(a * b for a, b in zip(wl, vector(P, O)))


def breakdown(P, O, w):
    """Per-feature contribution wᵢ·fᵢ, the 'why' panel of the developer view."""
    r = raw(P, O)
    return {k: {"raw": r[k], "weight": float(w.get(k, 0.0)), "contribution": float(w.get(k, 0.0)) * r[k] / SCALE[k]}
            for k in FEATURES}


# ------------------------------------------------------------------ the Assignment 3 agent's utility, unchanged
def u_a1(P, O):
    """U from Assignment 1 for the side to move P: 10·(opponent marbles off) + 2·centre + cohesion − 3·edge.

    Not zero-sum and only looks at the mover's own marbles; used only by the A3 baseline (greedy) agent.
    """
    return 10 * (14 - O.bit_count()) + 2 * _centre(P) + _pairs(P) - 3 * (P & EDGE_MASK).bit_count()

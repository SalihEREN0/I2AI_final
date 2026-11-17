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


def raw(P, O):
    """Unscaled feature differences, for display."""
    return {
        "marbles": P.bit_count() - O.bit_count(),
        "centre": _centre(P) - _centre(O),
        "cohesion": _pairs(P) - _pairs(O),
        "edge": (P & EDGE_MASK).bit_count() - (O & EDGE_MASK).bit_count(),
        "exposed": _exposed(P, O) - _exposed(O, P),
        "wedge": _wedge(P, O) - _wedge(O, P),
    }


def vector(P, O):
    r = raw(P, O)
    return [r[k] / SCALE[k] for k in FEATURES]


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

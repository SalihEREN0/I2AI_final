"""Playing games: agents, the game loop and the match rules from docs/a4-research.md §6.

Rules fixed before any training or match (the "open point" in the research note):
  - Black moves first; a game is won by pushing off 6 marbles.
  - The same position with the same side to move for the third time: draw by repetition.
  - After MAX_PLIES plies the game is adjudicated by marble difference; equal marbles = draw.
"""
import json
import random

from .engine import START, successors, move_name, WIN_COUNT
from .andor import forced
from . import features as F

MAX_PLIES = 200


class RandomAgent:
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose(self, P, O, ply=0, avoid=()):
        mv = self.rng.choice(list(successors(P, O)))[0]
        return mv, {"ply": ply, "agent": self.name, "move": move_name(mv)}


class GreedyU:
    """One ply, picks the move with the highest Assignment-1 utility U for itself (Assignment 3's greedy choice)."""
    name = "a3-greedy-U"

    def choose(self, P, O, ply=0, avoid=()):
        scored = [(F.u_a1(P2, O2), mv) for mv, _, P2, O2, _ in successors(P, O)]
        best = max(scored, key=lambda x: x[0])
        return best[1], {"ply": ply, "agent": self.name, "move": move_name(best[1]), "value": best[0],
                         "candidates": [{"move": move_name(m), "value": u}
                                        for u, m in sorted(scored, key=lambda x: -x[0])[:8]]}


class GreedyAndOr(GreedyU):
    """The Assignment 3 agent as presented there: greedy U ranked quiet moves above a capture, so A3 chose AND–OR.

    When AND–OR proves a push-off can be forced within two of our moves whatever the opponent replies, it plays a
    forcing first move (the one U likes best); otherwise it is greedy on U.
    """
    name = "a3-greedy-U+and-or"

    def choose(self, P, O, ply=0, avoid=()):
        f = forced(P, O, need=1, moves=2)
        if not f["forcing"]:
            mv, t = super().choose(P, O, ply, avoid)
            t["agent"], t["and_or_ms"] = self.name, f["search"]["ms"]
            return mv, t
        forcing = set(f["forcing"])
        scored = [(F.u_a1(P2, O2), mv) for mv, _, P2, O2, _ in successors(P, O) if move_name(mv) in forcing]
        best = max(scored, key=lambda x: x[0])
        return best[1], {"ply": ply, "agent": self.name, "move": move_name(best[1]), "value": best[0],
                         "forcing": f["forcing"], "and_or_ms": f["search"]["ms"]}


def play(black, white, start="standard", opening=(), max_plies=MAX_PLIES, trace=None, on_move=None):
    """Play one game. `opening` = moves played first (e.g. random opening plies), recorded but not searched.

    Returns a result dict; per-move traces go to `trace` (a list) if given, and to on_move(trace) if given.
    """
    B, W = START[start]
    agents = (black, white)
    seen, moves = {}, []
    side = 0                                   # 0 = Black to move
    result = None
    for ply in range(max_plies):
        P, O = (B, W) if side == 0 else (W, B)
        key = (P, O, side)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            result = ("draw", "repetition")
            break
        if ply < len(opening):
            mv, t = opening[ply], {"ply": ply, "agent": "opening", "move": move_name(opening[ply])}
        else:
            avoid = {(a, b) for (a, b, s), n in seen.items() if s != side}
            mv, t = agents[side].choose(P, O, ply, avoid)
            if mv is None:
                result = ("draw", "no legal move")
                break
        P2, O2 = None, None
        for m, _, a, b, _ in successors(P, O):
            if m == mv:
                P2, O2 = a, b
                break
        if P2 is None:
            raise ValueError("agent %s played an illegal move" % agents[side].name)
        B, W = (P2, O2) if side == 0 else (O2, P2)
        t["side"] = "black" if side == 0 else "white"
        t["marbles_after"] = {"black": B.bit_count(), "white": W.bit_count()}
        moves.append(t["move"])
        if trace is not None:
            trace.append(t)
        if on_move:
            on_move(t)
        if W.bit_count() <= WIN_COUNT:
            result = ("black", "six off")
            break
        if B.bit_count() <= WIN_COUNT:
            result = ("white", "six off")
            break
        side ^= 1
    if result is None:
        nb, nw = B.bit_count(), W.bit_count()
        result = ("black" if nb > nw else "white" if nw > nb else "draw", "ply cap")
    return {"winner": result[0], "reason": result[1], "plies": len(moves), "start": start,
            "black": black.name, "white": white.name,
            "off": {"black": 14 - B.bit_count(), "white": 14 - W.bit_count()}, "moves": moves,
            "final": {"black": B, "white": W}}


def random_opening(rng, start="standard", plies=2):
    """A few uniformly random plies to vary the games (both colours get the same openings in a match)."""
    B, W = START[start]
    P, O, out = B, W, []
    for _ in range(plies):
        mv, _, P2, O2, _ = rng.choice(list(successors(P, O)))
        out.append(mv)
        P, O = O2, P2
    return tuple(out)


def save_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

"""AND–OR search for forced push-offs (Chapter 4, AIMA Figure 4.11), carried over from Assignment 3.

Our move is an OR node (we choose one); each opponent reply is an AND node (every reply must be answered).
The question for the position to move: can we push `need` more opposing marbles off within `moves` of our own
moves, whatever the opponent does? A plan is a first move plus, for each reply, our answer.

As in Assignment 3, a branch that repeats a board fails (the cycle check of Figure 4.11), the shallowest
sub-plan is tried first, and every first move is examined so the panel can say how many of them force the
goal. One addition for live play: a reply that pushes off our sixth marble ends the game, so that branch fails.
Assignment 3's test board is a regression test (tests/test_search.py): exactly one of its 73 first moves forces
two marbles off within two moves.
"""
import time

from .engine import successors, move_name, WIN_COUNT


class AndOr:
    def __init__(self, target):
        self.target = target                      # the opponent's marble count that reaches the goal
        self.or_calls = self.and_calls = self.boards = 0

    def or_search(self, U, T, depth, path):
        """We (U) are to move. Returns a plan (move, sub-plan or None) or None."""
        self.or_calls += 1
        if (U, T) in path:
            return None
        kids = list(successors(U, T))
        self.boards += len(kids)
        for mv, _, U2, T2, _ in kids:
            if T2.bit_count() <= self.target:
                return (mv, None)
        if depth == 1:
            return None
        for mv, _, U2, T2, _ in kids:
            sub = self.and_search(U2, T2, depth - 1, path | {(U, T)})
            if sub is not None:
                return (mv, sub)
        return None

    def and_search(self, U, T, depth, path):
        """The opponent (T) is to move. Returns [(reply, plan)] covering every reply, or None."""
        self.and_calls += 1
        replies = list(successors(T, U))
        self.boards += len(replies)
        if not replies:
            return None
        plans = []
        for rv, _, T3, U3, _ in replies:
            if U3.bit_count() <= WIN_COUNT:       # the reply wins the game for them
                return None
            p = self.or_search(U3, T3, depth, path)
            if p is None:
                return None
            plans.append((rv, p))
        return plans


def forced(P, O, need=1, moves=2):
    """For the side to move P: which first moves force `need` push-offs within `moves` of our moves?

    Returns a JSON-ready record for the developer view: every first move with a guaranteed plan, the plan of the
    first one (each reply grouped under our answer), and the search effort.
    """
    t0 = time.perf_counter()
    target = O.bit_count() - need
    ao = AndOr(target)
    ao.or_calls += 1                              # the root OR node, examined move by move
    root = list(successors(P, O))
    ao.boards += len(root)
    found = []
    for mv, kind, P2, O2, off in root:
        if O2.bit_count() <= target:
            found.append((mv, None))
        elif moves > 1:
            sub = ao.and_search(P2, O2, moves - 1, {(P, O)})
            if sub is not None:
                found.append((mv, sub))
    plan = None
    if found:
        mv, sub = found[0]
        if sub is None:
            plan = {"first": move_name(mv), "immediate": True, "answers": []}
        else:
            groups = {}
            for rv, (ans, _) in sub:
                groups.setdefault(move_name(ans), []).append(move_name(rv))
            plan = {"first": move_name(mv), "immediate": False, "replies": len(sub),
                    "answers": sorted(({"answer": a, "replies": r} for a, r in groups.items()),
                                      key=lambda x: -len(x["replies"]))}
    return {"need": need, "moves": moves, "first_moves": len(root),
            "forcing": [move_name(m) for m, _ in found], "plan": plan,
            "search": {"or_calls": ao.or_calls, "and_calls": ao.and_calls, "boards": ao.boards,
                       "ms": round(1000 * (time.perf_counter() - t0), 1)}}

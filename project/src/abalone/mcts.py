"""Monte Carlo tree search (Chapter 6, AIMA §6.4), as the alternative to alpha-beta that the paper compares with.

UCT: each iteration selects a path from the root by UCB1, expands one untried move, estimates the new leaf and
backs the estimate up the path. A random playout to the end of the game is useless in Abalone (Chorus 2009:
random games can last 4,000 plies), so the leaf is estimated with the final agent's own evaluation, after its
quiescence search, turned into a win probability the way TDLeaf reads values: P(win) = (1 + tanh(v / K)) / 2.
With the same evaluation and the same clock, a match against the alpha-beta agent compares only the two ways
of searching the game tree.

Rules are the game's: pushing off a sixth marble wins at once, and a root move back to a position already seen
in this game counts as a draw, as it does for the alpha-beta agent.
"""
import math
import random
import time

from .engine import successors, move_name, WIN_COUNT
from .search import Searcher, INF
from .tdleaf import K


class _Node:
    __slots__ = ("P", "O", "untried", "children", "n", "w", "fixed")

    def __init__(self, P, O, fixed=None):
        self.P, self.O = P, O              # position with P to move
        self.untried = None                # moves not expanded yet, generated on the first visit
        self.children = []                 # (move, node)
        self.n = 0                         # visits
        self.w = 0.0                       # summed result for the player who moved into this node
        self.fixed = fixed                 # a known result for that player (win 1, draw 0.5): no search below


class MCTS:
    def __init__(self, weights=None, time_limit=0.5, iterations=None, c=math.sqrt(2), seed=0, name="mcts"):
        self.leaf = Searcher(weights=weights, depth=1)       # its evaluation and quiescence estimate the leaves
        self.time_limit, self.iterations = time_limit, iterations
        self.c = c
        self.rng = random.Random(seed)
        self.name = name

    def _estimate(self, P, O):
        """Probability that the side to move P wins, from the quiescence value of the position."""
        s = self.leaf
        s.deadline, s.nodes, s.qnodes = None, 0, 0
        v, _ = s._quiesce(P, O, -INF, INF, 0, s.qdepth)
        return (1 + math.tanh(v / K)) / 2

    def _expand(self, node, avoid, at_root):
        mv, P2, O2 = node.untried.pop()
        if O2.bit_count() <= WIN_COUNT:                       # the move pushed off the sixth marble
            child = _Node(O2, P2, fixed=1.0)
        elif at_root and (O2, P2) in avoid:                  # back to an earlier position of the game
            child = _Node(O2, P2, fixed=0.5)
        else:
            child = _Node(O2, P2)
        node.children.append((mv, child))
        return child

    def choose(self, P, O, ply=0, avoid=()):
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit if self.time_limit else None
        if len(self.leaf.ecache) > 300_000:                 # the leaf searcher's choose() never runs to trim it
            self.leaf.ecache.clear()
        root, c, it = _Node(P, O), self.c, 0
        while True:
            if it and (it >= self.iterations if self.iterations else time.perf_counter() > deadline):
                break
            it += 1
            node, path = root, [root]
            # selection: follow UCB1 while the node is fully expanded
            while node.fixed is None and node.untried is not None and not node.untried and node.children:
                log_n = math.log(node.n)
                node = max(node.children, key=lambda ch: ch[1].w / ch[1].n + c * math.sqrt(log_n / ch[1].n))[1]
                path.append(node)
            # expansion: one untried move, in random order
            if node.fixed is None:
                if node.untried is None:
                    node.untried = [(mv, P2, O2) for mv, _, P2, O2, _ in successors(node.P, node.O)]
                    self.rng.shuffle(node.untried)
                if node.untried:
                    node = self._expand(node, avoid, node is root)
                    path.append(node)
            # estimate: the result for the player who moved into the leaf
            if node.fixed is not None:
                r = node.fixed
            elif node.untried == [] and not node.children:
                r = 0.5                                       # no legal move: a draw
            else:
                r = 1 - self._estimate(node.P, node.O)
            # back-up: each node keeps the result of the player who moved into it
            for nd in reversed(path):
                nd.n += 1
                nd.w += r
                r = 1 - r
        if not root.children:
            return None, {"error": "no legal move"}
        mv, best = max(root.children, key=lambda ch: (ch[1].n, ch[1].w / ch[1].n))
        ranked = sorted(root.children, key=lambda ch: -ch[1].n)[:8]
        return mv, {"ply": ply, "agent": self.name, "move": move_name(mv), "value": round(best.w / best.n, 4),
                    "candidates": [{"move": move_name(m), "visits": ch.n, "score": round(ch.w / ch.n, 4)}
                                   for m, ch in ranked],
                    "stats": {"iterations": it, "ms": round(1000 * (time.perf_counter() - t0), 1),
                              "root_moves": len(root.children)}}

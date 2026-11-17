"""The faster engine must search exactly like the agent presented in Assignment 4.

The move generator, the features and the evaluation were rewritten for speed (2026-11). Every result of the
paper that does not depend on the clock (fixed-depth matches, learning runs, node counts) stays valid only if
the new code computes the same things, so these tests compare it with the reference implementations and with
the frozen agent in src/abalone_a4, on positions from random games that include pushes and push-offs.
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abalone import engine as E, features as F
from abalone.search import Searcher
from abalone_a4.search import Searcher as SearcherA4


def random_positions(n_games=60, seed=5, max_plies=160):
    """Positions of random games from both start layouts, to the end or the ply cap: openings, middle games,
    positions with pushes and push-offs, and positions near the end of a game."""
    rng = random.Random(seed)
    out = []
    for g in range(n_games):
        P, O = E.START["standard" if g % 2 else "belgian-daisy"]
        for _ in range(rng.randrange(20, max_plies)):
            out.append((P, O))
            moves = list(E.successors_plain(P, O))
            if not moves or P.bit_count() <= E.WIN_COUNT or O.bit_count() <= E.WIN_COUNT:
                break
            pushes = [m for m in moves if m[1] == "sumito"]
            _, _, P2, O2, _ = rng.choice(pushes if pushes and rng.random() < 0.4 else moves)
            P, O = O2, P2
    return out


POSITIONS = random_positions()


class SameMoves(unittest.TestCase):
    def test_successors_same_sequence(self):
        for P, O in POSITIONS:
            self.assertEqual(list(E.successors(P, O)), list(E.successors_plain(P, O)))

    def test_pushoffs_same_subsequence(self):
        seen = 0
        for P, O in POSITIONS:
            want = [(mv, P2, O2) for mv, _, P2, O2, off in E.successors_plain(P, O) if off]
            self.assertEqual(list(E.pushoffs(P, O)), want)
            seen += bool(want)
        self.assertGreater(seen, 100)                    # the positions do contain push-offs

    def test_perft(self):
        P, O = E.START["standard"]
        self.assertEqual([E.perft(P, O, d) for d in (1, 2, 3)], [44, 1936, 98912])


class SameValues(unittest.TestCase):
    WEIGHTS = [F.weights_list(F.HAND), [22.137332891842753, 16.967696535466096, 10.766235332659814,
                                        -6.547396229074165, -3.868431892086487, 2.4334679054101187],
               [1.5, -2.25, 0.125, 3.0, -7.5, 0.33]]

    def test_features_same_integers(self):
        for P, O in POSITIONS:
            self.assertEqual(F.raw(P, O), F.raw_plain(P, O))

    def test_evaluation_same_float(self):
        for wl in self.WEIGHTS:
            ev = F.make_evaluator(wl)
            for P, O in POSITIONS:
                r = F.raw_plain(P, O)
                old = sum(a * b for a, b in zip(wl, [r[k] / F.SCALE[k] for k in F.FEATURES]))
                self.assertEqual(ev(P, O), old)


class SameSearch(unittest.TestCase):
    def test_same_decision_and_nodes_at_fixed_depth(self):
        w = dict(zip(F.FEATURES, SameValues.WEIGHTS[1]))
        rng = random.Random(9)
        for P, O in rng.sample([p for p in POSITIONS if p[0].bit_count() > 9 and p[1].bit_count() > 9], 25):
            for depth in (2, 3):
                new = Searcher(weights=w, depth=depth).choose(P, O)
                old = SearcherA4(weights=w, depth=depth).choose(P, O)
                self.assertEqual(new[0], old[0])
                for k in ("value", "depth"):
                    self.assertEqual(new[1][k], old[1][k])
                for k in ("nodes", "qnodes", "cutoffs"):
                    self.assertEqual(new[1]["stats"][k], old[1]["stats"][k])


if __name__ == "__main__":
    unittest.main()

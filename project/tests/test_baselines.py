"""The two baselines the final agent is compared with: minimax without pruning and Monte Carlo tree search."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from abalone import engine as E, features as F
from abalone.mcts import MCTS
from abalone.search import Searcher, MinimaxSearcher, minimax
from test_speed import POSITIONS

W = dict(zip(F.FEATURES, [22.137332891842753, 16.967696535466096, 10.766235332659814,
                          -6.547396229074165, -3.868431892086487, 2.4334679054101187]))
MIDGAME = [p for p in POSITIONS if p[0].bit_count() > 9 and p[1].bit_count() > 9]


class Minimax(unittest.TestCase):
    def test_same_value_as_alpha_beta(self):
        """Pruning never changes the value: minimax and alpha-beta (no TT) agree, with and without quiescence."""
        for P, O in random.Random(3).sample(MIDGAME, 12):
            for q in (False, True):
                mm = MinimaxSearcher(weights=W, depth=2, quiescence=q).choose(P, O)[1]
                ab = Searcher(weights=W, depth=2, quiescence=q, tt=False).choose(P, O)[1]
                self.assertEqual(mm["value"], ab["value"])
                self.assertGreaterEqual(mm["stats"]["nodes"], ab["stats"]["nodes"])
            self.assertAlmostEqual(MinimaxSearcher(weights=W, depth=2, quiescence=False).choose(P, O)[1]["value"],
                                   round(minimax(P, O, 2, F.weights_list(W)), 4))


class MonteCarlo(unittest.TestCase):
    def test_legal_move(self):
        for P, O in random.Random(4).sample(MIDGAME, 8):
            mv, t = MCTS(weights=W, time_limit=None, iterations=200).choose(P, O)
            self.assertIn(mv, [m for m, *_ in E.successors(P, O)])
            self.assertEqual(t["stats"]["iterations"], 200)

    def test_takes_a_winning_push_off(self):
        """With the opponent at 9 marbles, a push-off wins the game at once; MCTS must find one."""
        wins = [(P, O) for P, O in POSITIONS if O.bit_count() == 9 and P.bit_count() > 9 and any(E.pushoffs(P, O))]
        self.assertTrue(wins)
        for P, O in wins[:6]:
            mv, _ = MCTS(weights=W, time_limit=None, iterations=300).choose(P, O)
            self.assertIn(mv, [m for m, _, _ in E.pushoffs(P, O)])


if __name__ == "__main__":
    unittest.main()

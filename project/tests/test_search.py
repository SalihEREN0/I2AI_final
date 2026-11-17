"""Correctness checks for the final agent. Run from the repository root: python -m unittest discover tests"""
import itertools
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abalone import engine as E, features as F
from abalone.search import Searcher, minimax
from abalone.game import play, RandomAgent, GreedyU


def random_position(rng, plies):
    P, O = E.START["standard"]
    for _ in range(plies):
        moves = list(E.successors(P, O))
        if not moves:
            break
        _, _, P2, O2, _ = rng.choice(moves)
        if P2.bit_count() <= E.WIN_COUNT or O2.bit_count() <= E.WIN_COUNT:
            break
        P, O = O2, P2
    return P, O


_POOL = []


def game_positions():
    """Positions from depth-2 self-play games with random openings: they contain contact, pushes and push-offs."""
    if not _POOL:
        rng = random.Random(2026)
        for _ in range(4):
            P, O = random_position(rng, 4)
            a, b = Searcher(depth=2, name="a"), Searcher(depth=2, name="b")
            seen = set()
            for ply in range(160):
                _POOL.append((P, O))
                mv, _ = (a, b)[ply % 2].choose(P, O, ply, seen)
                P2, O2, _ = E.apply(P, O, mv)
                if P2.bit_count() <= E.WIN_COUNT or O2.bit_count() <= E.WIN_COUNT + 1:
                    break
                seen.add((O2, P2))
                P, O = O2, P2
    return _POOL


def pushy_position(rng):
    return rng.choice(game_positions())


def with_push_off():
    return [(P, O) for P, O in game_positions() if any(off for *_, off in E.successors(P, O))]


class Engine(unittest.TestCase):
    def test_perft(self):
        P, O = E.START["standard"]
        self.assertEqual([E.perft(P, O, d) for d in (1, 2, 3)], [44, 1936, 98912])

    def test_generators_agree(self):
        rng = random.Random(7)
        for _ in range(200):
            P, O = pushy_position(rng)
            fast = Counter((O2, P2) for _, _, P2, O2, _ in E.successors(P, O))
            slow = Counter({(o, p): n for (p, o), n in E.naive_successors(P, O).items()})
            self.assertEqual(fast, slow)

    def test_push_off_prefilter_has_no_false_negatives(self):
        rng = random.Random(3)
        for _ in range(300):
            P, O = pushy_position(rng)
            if any(off for *_, off in E.successors(P, O)):
                self.assertTrue(E.can_push_off(P, O))

    def test_belgian_daisy(self):
        B, W = E.START["belgian-daisy"]
        self.assertEqual((B.bit_count(), W.bit_count(), B & W), (14, 14, 0))

    def test_features_zero_sum(self):
        rng = random.Random(5)
        for _ in range(100):
            P, O = pushy_position(rng)
            a, b = F.raw(P, O), F.raw(O, P)
            self.assertEqual({k: -v for k, v in a.items()}, b)


class AlphaBeta(unittest.TestCase):
    def test_same_value_as_minimax_for_every_switch_combination(self):
        """Pruning, ordering, killers and the TT must never change the value at a fixed depth."""
        rng = random.Random(11)
        wl = F.weights_list(F.HAND)
        positions = [pushy_position(rng) for _ in range(6)]
        for P, O in positions:
            want = minimax(P, O, 2, wl)
            for tt, ordering, killers in itertools.product((False, True), repeat=3):
                s = Searcher(depth=2, tt=tt, ordering=ordering, killers=killers, quiescence=False)
                _, tr = s.choose(P, O)
                self.assertAlmostEqual(tr["iterations"][-1]["value"], want, places=6,
                                       msg="tt=%s ordering=%s killers=%s" % (tt, ordering, killers))

    def test_depth3_matches_minimax(self):
        rng = random.Random(12)
        wl = F.weights_list(F.HAND)
        for _ in range(2):
            P, O = pushy_position(rng)
            _, tr = Searcher(depth=3, quiescence=False).choose(P, O)
            self.assertAlmostEqual(tr["iterations"][-1]["value"], minimax(P, O, 3, wl), places=6)

    def test_root_exact_gives_exact_values_for_all_moves(self):
        rng = random.Random(13)
        P, O = pushy_position(rng)
        wl = F.weights_list(F.HAND)
        _, tr = Searcher(depth=2, quiescence=False, root_exact=True, top_k=100).choose(P, O)
        for c in tr["candidates"]:
            self.assertEqual(c["bound"], "=")
            P2, O2, _ = E.apply(P, O, E.parse_move(c["move"]))
            self.assertAlmostEqual(c["value"], -minimax(O2, P2, 1, wl), places=6)

    def test_exact_values_after_the_decision(self):
        """The demo's multi-PV pass: every listed move gets its minimax value, the chosen move the value it was chosen
        with, and the agent's own tables are left as the decision left them."""
        rng = random.Random(15)
        wl = F.weights_list(F.HAND)
        for _ in range(3):
            P, O = pushy_position(rng)
            s = Searcher(depth=3, quiescence=False)
            mv, tr = s.choose(P, O)
            tt, hist = dict(s.tt), dict(s.history)
            moves = [E.parse_move(c["move"]) for c in tr["candidates"]]
            vals, nodes, complete, searched = s.exact_values(P, O, moves, tr["depth"])
            self.assertEqual(searched, len(moves))
            self.assertTrue(complete)
            self.assertEqual(set(vals), set(moves))
            self.assertGreater(nodes, 0)
            self.assertAlmostEqual(vals[mv][0], tr["value"], places=6)
            for m in moves:
                P2, O2, _ = E.apply(P, O, m)
                self.assertAlmostEqual(vals[m][0], -minimax(O2, P2, 2, wl), places=6)
                self.assertLessEqual(vals[m][0], tr["value"] + 1e-9)
            self.assertEqual((s.tt, s.history), (tt, hist))

    def test_exact_values_keep_finds_the_best_k_of_all_moves(self):
        """keep=k searches each move only far enough to rule it out, yet returns the true k best with exact values."""
        rng = random.Random(16)
        wl = F.weights_list(F.HAND)
        for _ in range(3):
            P, O = pushy_position(rng)
            s = Searcher(depth=3, quiescence=False, top_k=1000)
            mv, tr = s.choose(P, O)
            moves = [E.parse_move(c["move"]) for c in tr["candidates"]]
            true = sorted((-minimax(O2, P2, 2, wl) for _, _, P2, O2, _ in E.successors(P, O)), reverse=True)
            vals, _, complete, searched = s.exact_values(P, O, moves, tr["depth"], keep=5)
            self.assertEqual(searched, len(moves))
            self.assertTrue(complete)
            self.assertIn(mv, vals)
            self.assertEqual(sorted((round(v, 6) for v, _ in vals.values()), reverse=True), [round(v, 6) for v in true[:5]])

    def test_search_tree_is_true_and_changes_nothing(self):
        """The demo's alpha-beta tree: recording it leaves the search as it was, every reply's value is its minimax
        value or a correct bound, each refuted root move ends on its refutation, and the best move's replies were all
        searched, with the move's value their minimum."""
        rng = random.Random(17)
        wl = F.weights_list(F.HAND)
        for _ in range(3):
            P, O = pushy_position(rng)
            plain = Searcher(depth=3, tt=False, quiescence=False)
            s = Searcher(depth=3, tt=False, quiescence=False, tree=True)
            (m0, t0), (m1, t1) = plain.choose(P, O), s.choose(P, O)
            self.assertEqual((m0, t0["value"], t0["stats"]["nodes"], t0["stats"]["cutoffs"]),
                             (m1, t1["value"], t1["stats"]["nodes"], t1["stats"]["cutoffs"]))
            self.assertNotIn("tree", t0)
            tree = t1["tree"]
            self.assertEqual(len(tree), t1["legal_moves"])
            for node in tree:
                if node["kind"] != "searched":
                    continue
                for r in node["replies"]:
                    a, b, _ = E.apply(P, O, E.parse_move(r["pv"][0]))      # our move, then the opponent's reply
                    b, a, _ = E.apply(b, a, E.parse_move(r["pv"][1]))
                    true = minimax(a, b, 1, wl)                             # back to us, one ply left
                    if r["bound"] == "=":
                        self.assertAlmostEqual(r["value"], true, places=6)
                    elif r["bound"] == "<=":
                        self.assertLessEqual(true, r["value"] + 1e-9)
                    else:
                        self.assertGreaterEqual(true, r["value"] - 1e-9)
                if node["move"] == t1["move"]:
                    self.assertFalse(node["cut"])
                    self.assertEqual(len(node["replies"]), node["legal"])
                    self.assertAlmostEqual(min(r["value"] for r in node["replies"]), node["value"], places=6)
                elif node["bound"] == "<=":
                    self.assertTrue(node["cut"])
                    self.assertEqual(node["replies"][-1]["bound"], "<=")
                    self.assertLessEqual(node["replies"][-1]["value"], node["alpha"] + 1e-9)

    def test_pv_leaf_reproduces_value(self):
        """The value of the search equals the static value of the PV's last position (what TDLeaf trains on)."""
        rng = random.Random(14)
        for _ in range(8):
            P, O = pushy_position(rng)
            s = Searcher(depth=3, tt=False)
            mv, tr = s.choose(P, O)
            if isinstance(tr["value"], str):
                continue
            a, b = s.leaf(P, O, [E.parse_move(m) for m in tr["pv"]])
            self.assertAlmostEqual(F.evaluate(a, b, s.wl), tr["value"], places=6)

    def test_pool_covers_push_offs(self):
        self.assertGreater(len(with_push_off()), 5)

    def test_takes_a_free_marble(self):
        """Black 3-in-line pushes White's 2 marbles on row A to the west: one goes off. Depth 1 must see it."""
        P = E.mask("A3 A4 A5 B1 B2 B3 B4 B5 B6 C3 C4 C5".split())
        O = E.mask("A1 A2 I5 I6 I7 I8 I9 H4 H5 H6 H7 H8 H9".split())
        mv, tr = Searcher(depth=1).choose(P, O)
        self.assertEqual(tr["move"], "A3 A4 A5 W")
        self.assertTrue(tr["candidates"][0]["pushes_off"])

    def test_generators_agree_on_push_off_positions(self):
        for P, O in with_push_off()[:100]:
            fast = Counter((O2, P2) for _, _, P2, O2, _ in E.successors(P, O))
            slow = Counter({(o, p): n for (p, o), n in E.naive_successors(P, O).items()})
            self.assertEqual(fast, slow)


class TDLeaf(unittest.TestCase):
    def test_self_play_update(self):
        from abalone import tdleaf
        update, g = tdleaf.self_play((F.HAND, 1, 0.7, "marbles", 4, 3, 60))
        self.assertEqual(len(update), len(F.FEATURES))
        self.assertTrue(all(abs(x) < 10 for x in update))
        self.assertIn(g["reason"], ("six off", "repetition", "ply cap"))

    def test_reward(self):
        from abalone.tdleaf import reward
        self.assertEqual(reward("black", "six off", 14, 8, "marbles"), 1.0)
        self.assertEqual(reward("draw", "repetition", 12, 12, "result"), 0.0)
        self.assertLess(reward("draw", "repetition", 11, 13, "marbles"), 0)


class Supervised(unittest.TestCase):
    def test_logistic_fit_recovers_known_weights(self):
        from abalone.supervised import fit, sigmoid
        rng = random.Random(4)
        true = [1.5, -0.8, 0.3]
        X = [[rng.gauss(0, 1) for _ in true] for _ in range(4000)]
        y = [1.0 if rng.random() < sigmoid(sum(b * x for b, x in zip(true, row))) else 0.0 for row in X]
        beta = fit(X, y, ridge=0.0)
        for b, t in zip(beta, true):
            self.assertAlmostEqual(b, t, delta=0.15)

    def test_quiet_filter_keeps_exposed_positions(self):
        """Positions with a non-zero 'exposed' feature but no legal push-off must count as quiet."""
        from abalone.supervised import quiet
        kept = [(P, O) for P, O in game_positions() if quiet(P, O) and F.raw(P, O)["exposed"] != 0]
        self.assertGreater(len(kept), 0)
        for P, O in game_positions():
            if quiet(P, O):
                self.assertFalse(any(off for *_, off in E.successors(P, O)))


class ValueNetwork(unittest.TestCase):
    def test_symmetries_are_board_symmetries(self):
        from abalone.valuenet import SYMMETRIES, transform
        self.assertEqual(len({tuple(p) for p in SYMMETRIES}), 12)
        for P, O in game_positions()[::60]:
            n = E.perft(P, O, 2)
            for p in SYMMETRIES:
                self.assertEqual(sorted(p), list(range(61)))
                self.assertEqual(E.perft(transform(P, p), transform(O, p), 2), n)

    def test_gradient_step_matches_finite_difference(self):
        import math
        from abalone.valuenet import ValueNet
        P, O = game_positions()[10]
        act, t, eps = ValueNet.active(P, O), 1.0, 1e-6

        def loss(n):
            p = n.prob(P, O)
            return -(t * math.log(p) + (1 - t) * math.log(1 - p))
        net = ValueNet(8, seed=1)
        i, j = act[0], 3
        base = loss(net)
        net.W1[i][j] += eps
        numeric = (loss(net) - base) / eps
        net.W1[i][j] -= eps
        before = net.W1[i][j]
        net.step(act, t, lr=1.0)
        self.assertAlmostEqual(before - net.W1[i][j], numeric, places=4)


class AndOrSearch(unittest.TestCase):
    def test_reproduces_assignment_3(self):
        """A3's test board: of 73 first moves only E5 E6 E7 → E forces two marbles off within two moves."""
        from abalone.andor import forced
        B = E.mask("A1 A2 A3 A4 A5 B2 B3 B4 B5 E5 E6 E7 D6 D7".split())
        W = E.mask("I5 I6 I7 I8 I9 H4 H5 H6 H7 H8 H9 G7 E8 E9".split())
        r = forced(B, W, need=2, moves=2)
        self.assertEqual(r["first_moves"], 73)
        self.assertEqual(r["forcing"], ["E5 E6 E7 E"])
        self.assertEqual(sum(len(a["replies"]) for a in r["plan"]["answers"]), r["plan"]["replies"])

    def test_assignment_3_agent_takes_the_forced_capture(self):
        """A3's greedy trap: U alone prefers a quiet move to the capture; the A3 agent (greedy + AND–OR) captures."""
        from abalone.andor import forced
        from abalone.game import GreedyAndOr
        B = E.mask("A1 A2 A3 A4 A5 B2 B3 B4 B5 E5 E6 E7 D6 D7".split())
        W = E.mask("I5 I6 I7 I8 I9 H4 H5 H6 H7 H8 H9 G7 E8 E9".split())
        self.assertEqual(forced(B, W, need=1, moves=2)["forcing"], ["E5 E6 E7 E"])
        self.assertNotEqual(GreedyU().choose(B, W)[1]["move"], "E5 E6 E7 E")
        self.assertEqual(GreedyAndOr().choose(B, W)[1]["move"], "E5 E6 E7 E")

    def test_agrees_with_the_rulebook_generator(self):
        """'Is a push-off forced within two moves?' answered again with the literal generator and no plan object."""
        from abalone.andor import forced

        def naive(P, O, target, depth):
            for (P2, O2) in E.naive_successors(P, O):
                if O2.bit_count() <= target:
                    return True
                if depth > 1:
                    replies = list(E.naive_successors(O2, P2))
                    if replies and all(P3.bit_count() > E.WIN_COUNT and naive(P3, O3, target, depth - 1)
                                       for (O3, P3) in replies):
                        return True
            return False
        checked = 0
        for P, O in game_positions()[::25]:
            r = forced(P, O, need=1, moves=2)
            self.assertEqual(bool(r["forcing"]), naive(P, O, O.bit_count() - 1, 2))
            checked += bool(r["forcing"])
        self.assertGreater(checked, 0)


class Games(unittest.TestCase):
    def test_game_runs_and_ends(self):
        r = play(Searcher(depth=1, name="ab1"), RandomAgent(1), max_plies=60)
        self.assertIn(r["winner"], ("black", "white", "draw"))
        self.assertLessEqual(r["plies"], 60)


if __name__ == "__main__":
    unittest.main()

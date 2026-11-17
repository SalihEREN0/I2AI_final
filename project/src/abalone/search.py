"""Adversarial search (Chapter 6): negamax alpha-beta with iterative deepening and the classic enhancements.

Enhancements, each with its own switch so the ablations in docs/a4-research.md §6 can turn it off:
  tt          transposition table keyed by the position (P, O); also gives the best move from the last iteration
  ordering    push-offs first, then pushes, then by the history heuristic (Schaeffer 1989); TT move first
  killers     two killer moves per ply (Akl & Newborn 1977)
  quiescence  at the depth limit, keep searching moves that push a marble off, so a position is never judged
              in the middle of an exchange (the horizon effect)
Chorus (2009) measured each of these on Abalone; Lemmens (2005) used ordering and a TT.

Every call to choose() returns the move and a trace: a JSON-ready record of what the search did. The developer
view, the matches and the slides all read this record, so every number shown comes from a logged calculation.
"""
import math
import time

from . import features as F
from .engine import successors, pushoffs, apply, can_push_off, move_name, names, WIN_COUNT

WIN = 1_000_000.0
INF = float("inf")
EXACT, LOWER, UPPER = 0, 1, 2
BOUND = {EXACT: "=", LOWER: ">=", UPPER: "<="}


class Timeout(Exception):
    pass


class Searcher:
    def __init__(self, weights=None, depth=4, time_limit=None, tt=True, ordering=True, killers=True,
                 quiescence=True, qdepth=6, root_exact=False, top_k=8, tree=False, name="alpha-beta"):
        self.w = dict(weights or F.HAND)
        self.wl = F.weights_list(self.w)
        self.max_depth = depth
        self.time_limit = time_limit
        self.use_tt, self.use_ordering, self.use_killers = tt, ordering, killers
        self.use_q, self.qdepth = quiescence, qdepth
        self.root_exact = root_exact          # search every root move with a full window: exact values, slower
        self.top_k = top_k
        self.tree = tree                      # record the top two levels of each pass for the trace (display only)
        self.name = name
        self.tt = {}
        self.history = {}
        self._ev = F.make_evaluator(self.wl)
        self.ecache = {}                      # position -> static value; iterative deepening revisits most leaves

    def config(self):
        return {"name": self.name, "depth": self.max_depth, "time_limit": self.time_limit, "tt": self.use_tt,
                "ordering": self.use_ordering, "killers": self.use_killers, "quiescence": self.use_q,
                "qdepth": self.qdepth, "root_exact": self.root_exact, "weights": self.w}

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, P, O):
        v = self.ecache.get((P, O))
        if v is None:
            v = self.ecache[(P, O)] = self._ev(P, O)
        return v

    # ------------------------------------------------------------------ the search proper
    def _tick(self):
        self.nodes += 1
        if self.deadline and self.nodes & 1023 == 0 and time.perf_counter() > self.deadline:
            raise Timeout

    def _order(self, moves, tt_move, ply):
        """TT move first, then push-offs, killers, other pushes, and the rest by history score.

        The keys are built in one pass and the indices sorted, instead of calling a key function per move: the
        same keys and the same stable sort give the same order as before, at a fraction of the cost.
        """
        if not self.use_ordering and not tt_move:
            return moves
        mvs = [m[0] for m in moves]
        if not self.use_ordering:                # only the TT move goes first; the rest keep their order
            if tt_move in mvs:
                i = mvs.index(tt_move)
                return [moves[i]] + moves[:i] + moves[i + 1:]
            return moves
        hget = self.history.get
        keys = [(3e12 if off else 1e12 if kind == "sumito" else 0.0) + hget(mv, 0) for mv, kind, _, _, off in moves]
        if self.use_killers and ply < len(self.killers):
            for k in self.killers[ply]:
                if k in mvs:
                    i = mvs.index(k)
                    if not moves[i][4]:
                        keys[i] += 2e12
        if tt_move in mvs:
            keys[mvs.index(tt_move)] = 4e12
        return [moves[i] for i in sorted(range(len(moves)), key=keys.__getitem__, reverse=True)]

    def _negamax(self, P, O, depth, alpha, beta, ply, rec=None):
        """rec: a dict to record this node's replies in, for the tree in the trace; None everywhere but at ply 1."""
        self.nodes += 1                           # _tick(), inlined: this runs once per node
        if self.deadline and self.nodes & 1023 == 0 and time.perf_counter() > self.deadline:
            raise Timeout
        if P.bit_count() <= WIN_COUNT:
            return -(WIN - ply), []
        if depth <= 0:
            return self._quiesce(P, O, alpha, beta, ply, self.qdepth) if self.use_q else (self.evaluate(P, O), [])

        alpha0, key, tt_move = alpha, (P, O), None
        if self.use_tt:
            self.tt_probes += 1
            e = self.tt.get(key)
            if e is not None:
                e_depth, e_val, e_flag, e_move, e_pv = e
                tt_move = e_move
                if e_depth >= depth:
                    self.tt_hits += 1
                    if e_flag == EXACT:
                        self.tt_cuts += 1
                        if rec is not None:
                            rec["tt"] = True
                        return e_val, list(e_pv)
                    if e_flag == LOWER and e_val > alpha:
                        alpha = e_val
                    elif e_flag == UPPER and e_val < beta:
                        beta = e_val
                    if alpha >= beta:
                        self.tt_cuts += 1
                        if rec is not None:
                            rec["tt"] = True
                        return e_val, [e_move] if e_move else []

        moves = list(successors(P, O))
        if not moves:
            return 0.0, []                       # no legal move: treated as a draw (Lemmens' "dead end")
        moves = self._order(moves, tt_move, ply)
        if rec is not None:
            rec["legal"], rec["beta"] = len(moves), beta

        best, best_pv, best_move = -INF, [], None
        for i, (mv, kind, P2, O2, off) in enumerate(moves):
            a0 = alpha
            v, pv = self._negamax(O2, P2, depth - 1, -beta, -alpha, ply + 1)
            v = -v
            if rec is not None:
                rec["replies"].append((mv, v, a0, pv))
            if v > best:
                best, best_move, best_pv = v, mv, [mv] + pv
            if v > alpha:
                alpha = v
            if alpha >= beta:
                self.cutoffs += 1
                if rec is not None:
                    rec["cut"] = True
                if i == 0:
                    self.first_cutoffs += 1
                if not off and self.use_killers:
                    k = self.killers[ply]
                    if mv not in k:
                        k.insert(0, mv)
                        del k[2:]
                if self.use_ordering:
                    self.history[mv] = self.history.get(mv, 0) + depth * depth
                break

        if self.use_tt:
            flag = UPPER if best <= alpha0 else LOWER if best >= beta else EXACT
            self.tt[key] = (depth, best, flag, best_move, tuple(best_pv) if flag == EXACT else ())
        return best, best_pv

    def _quiesce(self, P, O, alpha, beta, ply, qd):
        self.nodes += 1
        if self.deadline and self.nodes & 1023 == 0 and time.perf_counter() > self.deadline:
            raise Timeout
        self.qnodes += 1
        if P.bit_count() <= WIN_COUNT:
            return -(WIN - ply), []
        stand = self.evaluate(P, O)
        if stand >= beta or qd == 0 or not can_push_off(P, O):
            return stand, []
        best, best_pv = stand, []
        if stand > alpha:
            alpha = stand
        for mv, P2, O2 in pushoffs(P, O):
            v, pv = self._quiesce(O2, P2, -beta, -alpha, ply + 1, qd - 1)
            v = -v
            if v > best:
                best, best_pv = v, [mv] + pv
            if v > alpha:
                alpha = v
            if alpha >= beta:
                break
        return best, best_pv

    # ------------------------------------------------------------------ root: iterative deepening + trace
    def _root(self, P, O, depth, order):
        """One iteration at the root. Returns (best value, best pv, per-move results) in search order."""
        alpha, results = -INF, []
        best, best_pv = -INF, []
        for mv, kind, P2, O2, off in order:
            n0 = self.nodes
            if (O2, P2) in self.avoid:          # repeats an earlier position of this game: scored as a draw
                v, pv, kind = 0.0, [], "repeat"
            else:
                lo = -INF if self.root_exact else alpha
                rec = {"replies": [], "alpha": lo} if self.tree else None
                v, pv = self._negamax(O2, P2, depth - 1, -INF, -lo, 1, rec)
                v = -v
            exact = self.root_exact or v > alpha or not results
            results.append({"move": mv, "kind": kind, "off": off, "value": v, "bound": "=" if exact else "<=",
                            "nodes": self.nodes - n0, "pv": [mv] + pv})
            if self.tree and kind != "repeat":
                results[-1]["rec"] = rec
            if v > best:
                best, best_pv = v, [mv] + pv
            if v > alpha:
                alpha = v
        return best, best_pv, results

    def choose(self, P, O, ply=0, avoid=()):
        """Pick a move for the side to move P. Returns (move, trace).

        avoid = positions (opponent to move, opponent's marbles first) already seen in this game; a root move
        that leads back to one is scored 0, a draw. Ahead, the agent then avoids repeating; behind, it may not.
        """
        t0 = time.perf_counter()
        self.avoid = avoid
        self.deadline = t0 + self.time_limit if self.time_limit else None
        self.nodes = self.qnodes = self.cutoffs = self.first_cutoffs = 0
        self.tt_probes = self.tt_hits = self.tt_cuts = 0
        self.killers = [[] for _ in range(self.max_depth + self.qdepth + 4)]
        self.history = {k: v // 2 for k, v in self.history.items()}     # age old history
        if len(self.tt) > 1_000_000:
            self.tt.clear()
        if len(self.ecache) > 300_000:
            self.ecache.clear()

        root = list(successors(P, O))
        if not root:
            return None, {"error": "no legal move"}
        order = self._order(root, None, 0)
        iterations, done = [], None
        for d in range(1, self.max_depth + 1):
            n0, q0, t1 = self.nodes, self.qnodes, time.perf_counter()
            c0, f0 = self.cutoffs, self.first_cutoffs
            try:
                if d == 1:
                    saved, self.deadline = self.deadline, None   # depth 1 always completes
                best, pv, results = self._root(P, O, d, order)
                if d == 1:
                    self.deadline = saved
            except Timeout:
                break
            done = (d, best, pv, results)
            cut = self.cutoffs - c0
            iterations.append({"depth": d, "value": best, "best": move_name(pv[0]), "nodes": self.nodes - n0,
                               "qnodes": self.qnodes - q0, "ms": round(1000 * (time.perf_counter() - t1), 1),
                               "cutoffs": cut,
                               "first_move_cutoff_rate": round((self.first_cutoffs - f0) / cut, 3) if cut else None})
            # next iteration: best move first, the rest by this iteration's values (exact ones ahead of bounds)
            rank = {r["move"]: (r["move"] == pv[0], r["value"]) for r in results}
            order = sorted(order, key=lambda x: rank[x[0]], reverse=True)
            if abs(best) > WIN / 2:
                break                                                # a forced win or loss was found
        d, best, pv, results = done
        self.last_value, self.last_pv = best, pv              # raw numbers for TDLeaf; the trace rounds them
        elapsed = time.perf_counter() - t0
        return pv[0], self._trace(P, O, ply, d, best, pv, results, iterations, elapsed)

    # ------------------------------------------------------------------ exact values for the developer view
    def exact_values(self, P, O, moves, depth, avoid=(), time_limit=None, keep=None):
        """Exact values of root moves at the given depth, as multi-PV analysis gives them. During the decision
        alpha-beta only proves the other moves "no better" (an upper bound); this says how good each one is. A fresh
        copy with this searcher's settings and an empty TT does it, so the agent's own tables, history and timing are
        untouched, and the chosen move gets back the value it was chosen with.

        keep=None: every move gets a full window. keep=k: only the best k need exact values, so each move is searched
        with alpha = the k-th best value so far; a move that cannot beat it fails low cheaply and is left out. Pass the
        moves best-first (the decision's order) and the k kept are the k best of all of them.
        Returns {move: (value, pv)} for the moves valued before the time limit, the nodes searched, whether every move
        was searched in time, and how many of the moves (from the front of the list) were."""
        x = self.__class__.__new__(self.__class__)
        x.__dict__.update(self.__dict__)
        x.tt, x.history, x.avoid = {}, {}, avoid
        x.killers = [[] for _ in range(depth + x.qdepth + 4)]
        x.nodes = x.qnodes = x.cutoffs = x.first_cutoffs = x.tt_probes = x.tt_hits = x.tt_cuts = 0
        x.deadline = time.perf_counter() + time_limit if time_limit else None
        by_move = {mv: (P2, O2) for mv, _, P2, O2, _ in successors(P, O)}
        out, complete, searched = {}, False, 0
        try:
            for mv in moves:
                P2, O2 = by_move[mv]
                if (O2, P2) in avoid:
                    v, pv = 0.0, []
                else:
                    lo = sorted((r[0] for r in out.values()), reverse=True)[keep - 1] \
                        if keep and len(out) >= keep else -INF
                    v, pv = x._negamax(O2, P2, depth - 1, -INF, -lo, 1)
                    v = -v
                    if v <= lo:                     # failed low: no better than the k-th best, value only a bound
                        searched += 1
                        continue
                out[mv] = (v, [mv] + pv)
                if keep and len(out) > keep:        # the k-th best rose: drop whatever fell below it
                    del out[min(reversed(list(out)), key=lambda m: out[m][0])]     # on a tie, the latest goes
                searched += 1
            complete = True
        except Timeout:
            pass
        return out, x.nodes, complete, searched

    # ------------------------------------------------------------------ trace for the developer view
    def leaf(self, P, O, pv):
        """The position at the end of the principal variation, as (root mover's marbles, opponent's marbles)."""
        a, b = P, O
        for i, mv in enumerate(pv):
            if i % 2 == 0:
                a, b, _ = apply(a, b, mv)
            else:
                b, a, _ = apply(b, a, mv)
        return a, b

    def _trace(self, P, O, ply, depth, best, pv, results, iterations, elapsed):
        ranked = sorted(results, key=lambda r: (r["bound"] == "=", r["value"]), reverse=True)
        cands = []
        for r in ranked[: self.top_k]:
            P2, O2, _ = apply(P, O, r["move"])
            cands.append({"move": move_name(r["move"]), "kind": r["kind"], "pushes_off": bool(r["off"]),
                          "value": _num(r["value"]), "bound": r["bound"], "nodes": r["nodes"],
                          "pv": [move_name(m) for m in r["pv"]],
                          "static": {k: v["contribution"] for k, v in F.breakdown(P2, O2, self.w).items()}})
        la, lb = self.leaf(P, O, pv)
        why = F.breakdown(la, lb, self.w)
        n = self.nodes
        return {
            "ply": ply, "agent": self.name, "to_move": names(P), "opponent": names(O),
            "move": move_name(pv[0]), "value": _num(best), "depth": depth,
            "pv": [move_name(m) for m in pv], "pv_length": len(pv),
            "legal_moves": len(results), "candidates": cands,
            "why": {"leaf_value": _num(sum(v["contribution"] for v in why.values())), "features": why},
            "iterations": iterations,
            "stats": {"nodes": n, "qnodes": self.qnodes, "ms": round(1000 * elapsed, 1),
                      "nodes_per_s": round(n / elapsed) if elapsed else None,
                      "ebf": ebf(iterations[-1]["nodes"] - iterations[-1]["qnodes"], depth) if iterations else None,
                      "cutoffs": self.cutoffs, "tt_probes": self.tt_probes, "tt_hits": self.tt_hits,
                      "tt_cutoffs": self.tt_cuts, "tt_size": len(self.tt)},
            "config": {k: v for k, v in self.config().items() if k != "weights"},
            **({"tree": self._tree(results)} if self.tree else {}),
        }

    @staticmethod
    def _tree(results):
        """The deepest pass's top two levels, for the developer view: every root move in search order and, under it,
        the opponent's replies tried before alpha-beta stopped, all from the root mover's side. A root move is
        searched to beat alpha, the best value so far; a reply that holds it to alpha or less refutes it (a cutoff)
        and the replies after it are never searched. The search is fail-soft, so a value outside the window is a
        bound: "<=" for a refutation, ">=" for a reply the opponent already had better than."""
        out = []
        for r in results:
            rec = r.get("rec")
            node = {"move": move_name(r["move"]), "value": _num(r["value"]), "bound": r["bound"], "nodes": r["nodes"],
                    "kind": "repeat" if rec is None else "tt" if rec.get("tt") else "leaf" if "legal" not in rec
                    else "searched"}
            if rec is not None and rec["alpha"] > -INF:
                node["alpha"] = _num(rec["alpha"])
            if node["kind"] == "searched":
                b = rec["beta"]
                node.update(legal=rec["legal"], cut=bool(rec.get("cut")), replies=[
                    {"move": move_name(m), "value": _num(-v), "bound": "=" if a0 < v < b else "<=" if v >= b else ">=",
                     "pv": [move_name(x) for x in [r["move"], m] + pv]}
                    for m, v, a0, pv in rec["replies"]])
            out.append(node)
        return out


def ebf(n, depth):
    """Effective branching factor b* (Ch. 3): the b for which a uniform tree of the given depth, b + b² + … + b^d,
    has the n positions a pass searched."""
    if not depth or n < 1:
        return None
    lo, hi = 1.0, max(2.0, float(n))
    for _ in range(80):
        b = (lo + hi) / 2
        if sum(b ** i for i in range(1, depth + 1)) < n:
            lo = b
        else:
            hi = b
    return round(lo, 2)


def _num(v):
    if v >= WIN / 2:
        return "win in %d" % math.ceil(WIN - v)
    if v <= -WIN / 2:
        return "loss in %d" % math.ceil(WIN + v)
    return round(v, 4)


class MinimaxSearcher(Searcher):
    """Minimax without alpha-beta (Chapter 6), as a baseline: every reply of every node is searched.

    Everything else is the final agent's: the evaluation, quiescence search, iterative deepening and the clock.
    Without pruning a transposition table, move ordering and killer moves have nothing to work with, so they
    are off; the root is searched with a full window, so every root value is exact.
    """

    def __init__(self, **kw):
        kw.update(tt=False, ordering=False, killers=False, root_exact=True)
        super().__init__(**kw)

    def _negamax(self, P, O, depth, alpha, beta, ply, rec=None):
        self.nodes += 1
        if self.deadline and self.nodes & 1023 == 0 and time.perf_counter() > self.deadline:
            raise Timeout
        if P.bit_count() <= WIN_COUNT:
            return -(WIN - ply), []
        if depth <= 0:
            return self._quiesce(P, O, -INF, INF, ply, self.qdepth) if self.use_q else (self.evaluate(P, O), [])
        best, best_pv = -INF, []
        for mv, kind, P2, O2, off in successors(P, O):
            v, pv = self._negamax(O2, P2, depth - 1, -INF, INF, ply + 1)
            if -v > best:
                best, best_pv = -v, [mv] + pv
        return (best, best_pv) if best_pv else (0.0, [])

    def _quiesce(self, P, O, alpha, beta, ply, qd):
        self.nodes += 1
        if self.deadline and self.nodes & 1023 == 0 and time.perf_counter() > self.deadline:
            raise Timeout
        self.qnodes += 1
        if P.bit_count() <= WIN_COUNT:
            return -(WIN - ply), []
        best, best_pv = self.evaluate(P, O), []          # standing pat: the side to move need not push
        if qd == 0 or not can_push_off(P, O):
            return best, best_pv
        for mv, P2, O2 in pushoffs(P, O):
            v, pv = self._quiesce(O2, P2, -INF, INF, ply + 1, qd - 1)
            if -v > best:
                best, best_pv = -v, [mv] + pv
        return best, best_pv


def minimax(P, O, depth, wl, ply=0):
    """Plain minimax (negamax, no pruning) for testing: alpha-beta must return the same value."""
    if P.bit_count() <= WIN_COUNT:
        return -(WIN - ply)
    if depth == 0:
        return F.evaluate(P, O, wl)
    vals = [-minimax(O2, P2, depth - 1, wl, ply + 1) for _, _, P2, O2, _ in successors(P, O)]
    return max(vals) if vals else 0.0

"""The demo app: a local web server for user mode and developer mode.

  PYTHONPATH=src python -m abalone.server [--port 8024]     then open http://localhost:8024

Standard library only, and the page has no external files, so the demo runs offline. The server owns one game
at a time. Agents are the same Searcher objects the matches use, and every move's trace is the record the
developer view draws, so what the audience sees is what the agent computed. Every finished ply is also saved
to experiments/demo-games/<game>.json, which the page can replay if a live demo has to be abandoned.

While an agent thinks, /api/progress reports each finished iteration of its search (depth, value, best moves so
far) and the live node count, so the page can show the search deepening instead of a spinner. It only reads what
the search has already computed, so the agent's decisions are unchanged. Its timing is barely affected: the
recording wrapper runs inside the time budget, once per finished depth.
"""
import argparse
import json
import math
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from . import features as F
from .engine import START, CELLS, NAME, DIRS, successors, move_name, parse_move, names, WIN_COUNT
from .match import make_agent
from .andor import forced
from .search import Searcher, WIN, _num
from .tdleaf import K

ROOT = Path(__file__).resolve().parents[2]
UI = Path(__file__).resolve().parent / "ui" / "index.html"
SAVES = ROOT / "experiments" / "demo-games"
WEIGHT_FILES = {
    "tdleaf-marbles": ROOT / "experiments" / "tdleaf-d2-marbles" / "weights-final.json",
    "tdleaf-result": ROOT / "experiments" / "tdleaf-d2-result" / "weights-final.json",
    "logistic": ROOT / "experiments" / "ch19-logistic" / "weights-final.json",
}
EXACT_TIME = 3.0         # seconds at most for the exact values after a decision (0.4-2.2 s on an M3, 1.3 on average)
SHOWN = 5                # moves the page lists: after a decision, the best SHOWN of all legal moves (8 costs ~60% more)

# How a search value reads as an expected score (win 1, draw ½, loss 0) for the side to move: σ(c·v), with c fitted
# by logistic regression on how 1,200 of TDLeaf's self-play games ended, for the final agent's weights (Ch. 19).
# TDLeaf's own K (tdleaf.py) is a training setting, never checked against results: it reads values twice as steeply.
SCORE_FIT = ROOT / "experiments" / "refit-logistic-on-tdleaf-games" / "report.json"


def _score_scale():
    r = json.loads(SCORE_FIT.read_text())
    return {"c": r["test"]["tdleaf-marbles"]["fitted_scale"], "source": str(SCORE_FIT.relative_to(ROOT)),
            "games": r["games"], "positions": r["train_positions"], "draws": r["results"]["repetition"],
            "log_loss": r["test"]["tdleaf-marbles"]["log_loss"], "coin_flip": r.get("baseline_coin_flip_log_loss")}


SCORE = _score_scale()
KS = 2 / SCORE["c"]      # the same curve written as TDLeaf writes it: (1 + tanh(v / KS)) / 2 = σ(c·v)


def weight_sets():
    out = {"hand": dict(F.HAND)}
    for k, p in WEIGHT_FILES.items():
        if p.exists():
            out[k] = json.loads(p.read_text())
    return out


def presets():
    w = WEIGHT_FILES["tdleaf-marbles"].relative_to(ROOT) if WEIGHT_FILES["tdleaf-marbles"].exists() else None
    learned = (",weights=%s" % w) if w else ""
    # label: the full name in menus; short: the name on the player chips
    return [
        {"id": "human", "label": "Human", "short": "Human", "spec": "human"},
        {"id": "final", "label": "Final agent: alpha-beta + learned weights, 1 s", "short": "Final agent",
         "spec": "ab:time=1%s,name=final" % learned},
        {"id": "final-fast", "label": "Final agent, 0.3 s", "short": "Final agent (fast)",
         "spec": "ab:time=0.3%s,name=final-fast" % learned},
        {"id": "hand", "label": "Alpha-beta + hand-set U, 1 s", "short": "Hand-set U", "spec": "ab:time=1,name=hand-U"},
        {"id": "logistic", "label": "Alpha-beta + logistic-regression weights, 1 s", "short": "Logistic (Ch. 19)",
         "spec": "ab:time=1,weights=experiments/ch19-logistic/weights-final.json,name=logistic"},
        {"id": "valuenet", "label": "Alpha-beta + value network (Ch. 22), 1 s", "short": "Value net (Ch. 22)",
         "spec": "net:time=1,net=experiments/ch22-valuenet/net.json,name=value-network"},
        {"id": "a3", "label": "Assignment 3 agent (greedy on U + AND–OR)", "short": "A3 agent", "spec": "greedy-andor"},
        {"id": "random", "label": "Random mover", "short": "Random", "spec": "random"},
    ]


def _summary(name):
    """One match summary from experiments/, with the file it came from (every number on screen has a source)."""
    p = ROOT / "experiments" / name / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"source": str(p.relative_to(ROOT)), "w": d["a_wins"], "d": d["draws"], "l": d["a_losses"],
            "score": d["a_score"], "ci": d["a_score_ci95"], "reasons": d["reasons"]}


def evidence(log):
    """The measured results the developer view shows next to the live game."""
    games = {r["iteration"]: r["games"] for r in log}
    curve = []
    for p in sorted((ROOT / "experiments" / "e4-learning-curve").glob("it*/summary.json")):
        it = int(p.parent.name[2:])
        s = _summary("e4-learning-curve/" + p.parent.name)
        s["games"] = games.get(it, it * 8)
        curve.append(s)
    out = {"vs_a3": _summary("e1-final-vs-a3-andor"),
           "ablation": [dict(_summary("e2-ablate-" + k) or {}, part=k) for k in
                        ("learning", "quiescence", "ordering", "tt", "killers")],
           "curve": curve,
           "models": [dict(_summary(f) or {}, a=a, b=b) for f, a, b in (
               ("match-tdleaf-vs-hand-d2", "TDLeaf (Ch. 23)", "hand-set U"),
               ("e5-result-vs-hand", "TDLeaf, result-only reward", "hand-set U"),
               ("ch19-logistic-vs-hand", "logistic regression (Ch. 19)", "hand-set U"),
               ("ch19-logistic-vs-tdleaf", "logistic regression (Ch. 19)", "TDLeaf"),
               ("refit-logistic-vs-tdleaf", "logistic, refit on TDLeaf's games", "TDLeaf"),
               ("ch22-net-vs-hand-d2", "value network (Ch. 22)", "hand-set U"),
               ("ch22-net-vs-tdleaf-d2", "value network (Ch. 22)", "TDLeaf"))]}
    p = ROOT / "experiments" / "convergence" / "summary.json"
    if p.exists():                                   # E6: why edge and exposed are still moving at the end
        c = json.loads(p.read_text())
        out["convergence"] = {"corr_edge_exposed": c["corr"]["edge"]["exposed"], "source": str(p.relative_to(ROOT)),
                              "vs150_d2": _summary("convergence-400-vs-150-d2"),
                              "vs150_time": _summary("convergence-400-vs-150-time"), "seeds": len(c["seeds"])}
    for key, f in (("search_cost", "search-cost"), ("andor", "ch4-andor-vs-alphabeta")):
        p = ROOT / "experiments" / f / "summary.json"
        if p.exists():
            out[key] = dict(json.loads(p.read_text()), source=str(p.relative_to(ROOT)))
    return out


def _raw(v):
    """A trace value as a number for sorting: a forced win above every evaluation, a forced loss below."""
    if isinstance(v, str):
        n = int(v.split()[-1])
        return WIN - n if v.startswith("win") else -WIN + n
    return v


def win_prob_black(value, side):
    """The search value as Black's expected score (win 1, draw ½), on the scale fitted to game results: σ(c·v)."""
    if not isinstance(value, (int, float)):
        if isinstance(value, str):
            mover_wins = value.startswith("win")
            return 1.0 if mover_wins == (side == "black") else 0.0
        return None
    v = value if side == "black" else -value
    return (1 + math.tanh(v / KS)) / 2


class _Stoppable(Searcher):
    """A searcher another request can stop: once `stop` is set, any deadline it has reads as passed. Depth 1 runs
    without one, so a stopped search still ends with a move, the way a search that runs out of time does."""
    def __init__(self, stop, **kw):
        self.stop, self._deadline = stop, None
        super().__init__(**kw)

    @property
    def deadline(self):
        return 1e-9 if self._deadline is not None and self.stop.is_set() else self._deadline

    @deadline.setter
    def deadline(self, v):
        self._deadline = v


class Game:
    def __init__(self, start="standard", black="human", white="final"):
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.start = start
        ids = {p["id"]: p["spec"] for p in presets()}
        self.specs = {"black": ids.get(black, black), "white": ids.get(white, white)}
        self.agents = {s: (None if spec == "human" else make_agent(spec)) for s, spec in self.specs.items()}
        self.B, self.W = START[start]
        self.side = "black"
        self.seen = {}
        self.history = []          # one entry per ply: move, trace, position after
        self.result = None
        self.progress = None       # the search in progress, read by /api/progress
        self.pending = None        # the latest agent decision still waiting for its exact values (/api/exact)
        for a in self.agents.values():
            if isinstance(a, Searcher):
                a.top_k = 1000     # every legal move in the trace, in the decision's order, for the exact pass
                a.tree = True      # the top two levels of the search, for the alpha-beta tree in the Search tab
        self._count()

    # ------------------------------------------------------------------ rules
    def pov(self):
        return (self.B, self.W) if self.side == "black" else (self.W, self.B)

    def _count(self):
        P, O = self.pov()
        key = (P, O, self.side)
        self.seen[key] = self.seen.get(key, 0) + 1
        if self.seen[key] >= 3 and not self.result:
            self.result = {"winner": "draw", "reason": "repetition"}
        return self.seen[key]

    def _apply(self, mv, trace, agent_name):
        P, O = self.pov()
        for m, kind, P2, O2, off in successors(P, O):
            if m == mv:
                break
        else:
            raise ValueError("illegal move " + move_name(mv))
        if self.side == "black":
            self.B, self.W = P2, O2
        else:
            self.W, self.B = P2, O2
        value = trace.get("value") if trace else None
        self.history.append({
            "ply": len(self.history), "side": self.side, "agent": agent_name, "move": move_name(mv),
            "cells": [NAME[c] for c in mv[0]], "dir": DIRS[mv[1]][0], "kind": kind, "pushed_off": off,
            "value": value,
            "win_prob_black": win_prob_black(value, self.side) if trace and "depth" in trace else None,
            "black": names(self.B), "white": names(self.W),
            "off": {"black": 14 - self.B.bit_count(), "white": 14 - self.W.bit_count()},
            "trace": trace,
        })
        if self.W.bit_count() <= WIN_COUNT:
            self.result = {"winner": "black", "reason": "six off"}
        elif self.B.bit_count() <= WIN_COUNT:
            self.result = {"winner": "white", "reason": "six off"}
        self.side = "white" if self.side == "black" else "black"
        if not self.result and len(self.history) >= 200:
            nb, nw = self.B.bit_count(), self.W.bit_count()
            self.result = {"winner": "black" if nb > nw else "white" if nw > nb else "draw", "reason": "ply cap"}
        self.history[-1]["repeat"] = self._count()   # how often this position has now occurred (3 = draw)
        self.save()

    def agent_step(self):
        agent = self.agents[self.side]
        if agent is None or self.result:
            return
        P, O = self.pov()
        other = "white" if self.side == "black" else "black"
        avoid = {(a, b) for (a, b, s) in self.seen if s == other}
        watched = isinstance(agent, Searcher)
        self.pending = None        # not asked for (user mode, or the page moved on): the move keeps its bounds
        self.progress = {"ply": len(self.history), "side": self.side, "t0": time.perf_counter(),
                         "agent": agent, "iterations": []}
        if watched:
            agent._root = self._watch(agent, self.progress)
        try:
            mv, trace = agent.choose(P, O, len(self.history), avoid)
        finally:
            if watched:
                del agent._root
            self.progress = None
        trace["agent_label"] = self.label(self.side)
        pool = []
        if watched and trace.get("depth"):
            # the trace lists SHOWN moves; the rest wait for the exact pass, which finds the best SHOWN of them all
            trace["candidates"], pool = trace["candidates"][:SHOWN], trace["candidates"][SHOWN:]
        if watched and trace.get("why"):
            self._leaves(agent, P, O, trace["candidates"])
        self._apply(mv, trace, agent.name)
        if watched and trace.get("depth"):
            self.pending = (len(self.history) - 1, agent, P, O, avoid, pool)

    @staticmethod
    def _leaves(agent, P, O, cands):
        """U's breakdown at the end of each candidate's line, so the page can compare any two candidates."""
        for c in cands:
            pv = c["exact"]["pv"] if "exact" in c else c["pv"]
            la, lb = agent.leaf(P, O, [parse_move(m) for m in pv])
            c["leaf"] = {k: v["contribution"] for k, v in F.breakdown(la, lb, agent.w).items()}

    def exact(self):
        """Exact values for the latest agent decision (/api/exact). The page asks for them while it reveals the move,
        so the work overlaps the animation; the server lock keeps it from running alongside the next search.

        During the decision alpha-beta only proves the other moves no better than the best. Here every legal move is
        searched again at the same depth, multi-PV style: each only far enough to show whether it is among the best
        SHOWN, which then get exact values and are listed. Display only: the move is already played, and the trace
        keeps this pass's cost apart from the decision's."""
        job, self.pending = self.pending, None
        if not job:
            return None
        i, agent, P, O, avoid, pool = job
        trace = self.history[i]["trace"]
        t0 = time.perf_counter()
        cands = trace["candidates"] + pool
        vals, nodes, complete, searched = agent.exact_values(P, O, [parse_move(c["move"]) for c in cands], trace["depth"], avoid,
                                         time_limit=EXACT_TIME, keep=SHOWN)
        for c in cands:
            r = vals.get(parse_move(c["move"]))
            if r:
                c["exact"] = {"value": _num(r[0]), "pv": [move_name(m) for m in r[1]]}
        # best first by exact value (the chosen move first among equals); moves not finished in time keep their order
        done = sorted((c for c in cands if "exact" in c), reverse=True,
                      key=lambda c: (_raw(c["exact"]["value"]), c["move"] == trace["move"]))
        rest = [c for c in cands if "exact" not in c]
        keep = (done + rest)[:SHOWN]
        if not any(c["move"] == trace["move"] for c in keep):
            keep[-1] = next(c for c in cands if c["move"] == trace["move"])
        trace["candidates"] = keep
        if trace.get("why"):
            self._leaves(agent, P, O, [c for c in keep if "exact" in c])
        # every move left out failed low against the SHOWN-th best: that value bounds them all
        below = min((_raw(c["exact"]["value"]) for c in done), default=None) if complete and len(cands) > SHOWN else None
        trace["exact"] = {"depth": trace["depth"], "pool": len(cands), "searched": searched, "scored": len(done),
                          "complete": complete, "keep": SHOWN,
                          "below": _num(below) if below is not None else None,
                          "nodes": nodes,
                          "ms": round(1000 * (time.perf_counter() - t0), 1), "time_limit": EXACT_TIME}
        self.save()
        return {"ply": i, "trace": trace}

    @staticmethod
    def _watch(agent, prog):
        """Wrap the searcher's root iteration: each finished depth is recorded for /api/progress."""
        root = agent._root

        def watched(P, O, depth, order):
            best, pv, results = root(P, O, depth, order)
            ranked = sorted(results, key=lambda r: (r["bound"] == "=", r["value"]), reverse=True)[:6]
            prog["iterations"].append({
                "depth": depth, "value": _num(best), "pv": [move_name(m) for m in pv[:4]],
                "ms": round(1000 * (time.perf_counter() - prog["t0"])), "nodes": agent.nodes,
                "top": [{"move": move_name(r["move"]), "value": _num(r["value"]), "bound": r["bound"]} for r in ranked]})
            return best, pv, results
        return watched

    def human_move(self, text):
        if self.agents[self.side] is not None or self.result:
            raise ValueError("not a human's turn")
        self._apply(parse_move(text), None, "human")

    def undo(self, plies):
        moves = [h["move"] for h in self.history[:-plies]] if plies else []
        traces = [(h["trace"], h["agent"]) for h in self.history[:-plies]]
        old = self.history
        self.B, self.W = START[self.start]
        self.side, self.seen, self.history, self.result, self.pending = "black", {}, [], None, None
        self._count()
        for m, (t, a) in zip(moves, traces):
            self._apply(parse_move(m), t, a)
        return len(old) - len(self.history)

    # ------------------------------------------------------------------ views
    def label(self, side, key="label"):
        spec = self.specs[side]
        for p in presets():
            if p["spec"] == spec:
                return p[key]
        return spec

    def legal(self):
        P, O = self.pov()
        return [{"move": move_name(m), "cells": [NAME[c] for c in m[0]], "dir": DIRS[m[1]][0], "kind": kind,
                 "pushes_off": bool(off)} for m, kind, _, _, off in successors(P, O)]

    def state(self, full=True):
        human = self.agents[self.side] is None and not self.result
        last = self.history[-1] if self.history else None
        return {
            "id": self.id, "start": self.start, "side": self.side, "ply": len(self.history),
            "black": names(self.B), "white": names(self.W),
            "off": {"black": 14 - self.B.bit_count(), "white": 14 - self.W.bit_count()},
            "players": {s: {"spec": self.specs[s], "label": self.label(s), "short": self.label(s, "short"),
                            "human": self.agents[s] is None}
                        for s in ("black", "white")},
            "result": self.result, "human_to_move": human,
            "repeat": self.seen.get((*self.pov(), self.side), 1),
            "legal": self.legal() if human else [],
            "history": [{k: v for k, v in h.items() if k != "trace"} for h in self.history] if full else None,
            "last": last,
        }

    def save(self):
        SAVES.mkdir(parents=True, exist_ok=True)
        rec = {"id": self.id, "start": self.start, "players": {s: self.label(s) for s in ("black", "white")},
               "result": self.result, "history": self.history}
        (SAVES / (self.id + ".json")).write_text(json.dumps(rec))


class App:
    def __init__(self):
        self.lock = threading.Lock()
        self.game = Game()
        self.stop = threading.Event()   # set by the next move, undo or new game: stops the /api/options search
        self.opts = None                # ((game id, ply), the last /api/options answer)

    def info(self):
        log = []
        p = ROOT / "experiments" / "tdleaf-d2-marbles" / "log.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                log.append({"iteration": r["iteration"], "games": r["games"], "weights": r["weights"]})
        return {"cells": [{"name": NAME[i], "r": r, "n": n} for i, (r, n) in enumerate(CELLS)],
                "dirs": [{"name": d, "dr": dr, "dn": dn} for d, dr, dn in DIRS],
                "starts": {k: {"black": names(b), "white": names(w)} for k, (b, w) in START.items()},
                "presets": presets(), "weights": weight_sets(),
                "features": F.FEATURES, "scale": F.SCALE, "K": KS, "K_train": K, "score": SCORE, "training_log": log,
                "training_source": str(p.relative_to(ROOT)) if p.exists() else None,
                "evidence": evidence(log)}

    def analyse(self, body):
        """Search the current position again with other settings, without moving: the live ablation."""
        g = self.game
        P, O = g.pov()
        ws = weight_sets()
        s = Searcher(weights=ws.get(body.get("weights", "tdleaf-marbles"), F.HAND),
                     depth=int(body.get("depth", 64)), time_limit=float(body.get("time", 1.0)),
                     tt=bool(body.get("tt", True)), ordering=bool(body.get("ordering", True)),
                     killers=bool(body.get("killers", True)), quiescence=bool(body.get("quiescence", True)),
                     root_exact=bool(body.get("root_exact", False)), top_k=int(body.get("top_k", 8)),
                     name="analysis")
        other = "white" if g.side == "black" else "black"
        avoid = {(a, b) for (a, b, s2) in g.seen if s2 == other}
        _, trace = s.choose(P, O, len(g.history), avoid)
        trace["settings"] = body
        trace["win_prob_black"] = win_prob_black(trace["value"], g.side)
        return trace

    def options(self):
        """Values for a human player's moves (/api/options), for the developer view: the final agent's search from
        the human's side (0.5 s, as the hint), then the exact pass of /api/exact, so the board can label the best
        SHOWN moves with their rank and expected score. Display only. It runs outside the lock on a copy of the
        position, and the next move, undo or new game stops it, so it never holds up a move or takes time from an
        agent's search."""
        with self.lock:
            g = self.game
            ply = len(g.history)
            if g.agents[g.side] is not None or g.result:
                return {"id": g.id, "ply": ply, "moves": []}
            if self.opts and self.opts[0] == (g.id, ply):
                return self.opts[1]
            P, O = g.pov()
            other = "white" if g.side == "black" else "black"
            avoid = {(a, b) for (a, b, s) in g.seen if s == other}
            self.stop.set()                          # an older request still running is for a position now gone
            stop = self.stop = threading.Event()
            key = (g.id, ply)
        t0 = time.perf_counter()
        s = _Stoppable(stop, weights=weight_sets().get("tdleaf-marbles", F.HAND), depth=64, time_limit=0.5,
                       top_k=1000, name="options")
        mv, trace = s.choose(P, O, ply, avoid)
        cands = [parse_move(c["move"]) for c in trace["candidates"]]
        vals, nodes, complete, searched = s.exact_values(P, O, cands, trace["depth"], avoid,
                                                         time_limit=EXACT_TIME, keep=SHOWN)
        if stop.is_set():
            return {"id": key[0], "ply": ply, "moves": [], "stopped": True}
        # best first, the agent's own pick first among equals (as /api/exact ranks them)
        ranked = sorted(vals.items(), reverse=True, key=lambda kv: (_raw(_num(kv[1][0])), kv[0] == mv))
        out = {"id": key[0], "ply": ply, "depth": trace["depth"], "pick": move_name(mv),
               "moves": [{"move": move_name(m), "value": _num(v), "pv": [move_name(x) for x in pv]}
                         for m, (v, pv) in ranked],
               "pool": len(cands), "searched": searched, "complete": complete,
               "nodes": trace["stats"]["nodes"] + nodes, "ms": round(1000 * (time.perf_counter() - t0), 1)}
        with self.lock:
            self.opts = (key, out)
        return out


APP = App()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html"):
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/info":
                return self._send(200, APP.info())
            if path == "/api/progress":                    # no lock: the search holds it while it runs
                p = APP.game.progress
                if not p:
                    return self._send(200, {"active": False})
                return self._send(200, {"active": True, "ply": p["ply"], "side": p["side"],
                                        "ms": round(1000 * (time.perf_counter() - p["t0"])),
                                        "nodes": getattr(p["agent"], "nodes", None),
                                        "iterations": list(p["iterations"])})
            if path == "/api/state":
                with APP.lock:
                    return self._send(200, APP.game.state())
            if path == "/api/saved":
                files = sorted(SAVES.glob("*.json"), reverse=True) if SAVES.exists() else []
                out = []
                for f in files[:50]:
                    rec = json.loads(f.read_text())
                    out.append({"id": rec["id"], "players": rec["players"], "plies": len(rec["history"]),
                                "result": rec["result"]})
                return self._send(200, out)
            if path.startswith("/api/saved/"):
                name = Path(path.rsplit("/", 1)[1]).name
                f = SAVES / (name + ".json")
                if not f.exists():
                    return self._send(404, {"error": "no such game"})
                rec = json.loads(f.read_text())
                for h in rec["history"]:                 # games saved before the fitted scale replay on it too
                    if isinstance(h.get("win_prob_black"), (int, float)):
                        h["win_prob_black"] = win_prob_black(h["value"], h["side"])
                return self._send(200, rec)
            return self._send(404, {"error": "not found"})
        except Exception as e:                       # the demo must not die on a bad request
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            body = self._body()
            if path == "/api/options":                   # no lock: it takes it only to copy the position
                return self._send(200, APP.options())
            if path in ("/api/new", "/api/step", "/api/move", "/api/undo"):
                APP.stop.set()                           # the position changes: its options are no longer wanted
            with APP.lock:
                g = APP.game
                if path == "/api/new":
                    APP.game = Game(body.get("start", "standard"), body.get("black", "human"), body.get("white", "final"))
                    return self._send(200, APP.game.state())
                if path == "/api/step":
                    g.agent_step()
                    return self._send(200, g.state())
                if path == "/api/exact":
                    return self._send(200, g.exact() or {})
                if path == "/api/move":
                    g.human_move(body["move"])
                    return self._send(200, g.state())
                if path == "/api/undo":
                    g.undo(int(body.get("plies", 1)))
                    return self._send(200, g.state())
                if path == "/api/analyse":
                    return self._send(200, APP.analyse(body))
                if path == "/api/andor":
                    P, O = g.pov()
                    if g.result:
                        return self._send(200, {"side": g.side, "ply": len(g.history), "results": []})
                    moves = min(int(body.get("moves", 2)), 2)          # 3+ of our moves is too slow to run live
                    return self._send(200, {"side": g.side, "ply": len(g.history),
                                            "results": [forced(P, O, n, moves) for n in (1, 2)]})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(400, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser(prog="abalone.server")
    ap.add_argument("--port", type=int, default=8024)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print("Abalone demo on http://localhost:%d  (Ctrl+C to stop)" % a.port, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

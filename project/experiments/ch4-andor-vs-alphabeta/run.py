"""Where a push-off can be forced within two moves (but not at once), does the final agent choose a forcing move?

Run from the repository root: python experiments/ch4-andor-vs-alphabeta/run.py
Positions: every position of the Chapter 19 test games (240 held-out games of the hand-set agent).
Agent: alpha-beta with the TDLeaf weights at fixed depth 3, so the run is deterministic.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from abalone import supervised as S
from abalone.andor import forced
from abalone.engine import START, apply, parse_move, successors
from abalone.match import make_agent

recs = S.load("experiments/ch19-logistic")
test = sorted(S.test_games(len(recs)))
agent = make_agent("ab:depth=3,weights=experiments/tdleaf-d2-marbles/weights-final.json,name=final")
rows = []
for gi in test[:60]:
    rec = recs[gi]
    P, O = START[rec["start"]]
    for ply, text in enumerate(rec["moves"]):
        if ply >= rec["opening_plies"] and not any(off for *_, off in successors(P, O)):
            r = forced(P, O, 1, 2)
            if r["forcing"]:
                _, t = agent.choose(P, O)
                rows.append({"game": gi, "ply": ply, "forcing": r["forcing"], "first_moves": r["first_moves"],
                             "agent_move": t["move"], "agent_forces": t["move"] in r["forcing"], "agent_value": t["value"]})
        P2, O2, _ = apply(P, O, parse_move(text))
        P, O = O2, P2
summary = {"games": 60, "positions_with_forced_push_off": len(rows),
           "agent_chose_forcing_move": sum(r["agent_forces"] for r in rows), "agent": "alpha-beta, TDLeaf weights, depth 3"}
out = Path("experiments/ch4-andor-vs-alphabeta")
(out / "summary.json").write_text(json.dumps(summary, indent=1))
(out / "positions.json").write_text(json.dumps(rows, indent=1))
print(summary)

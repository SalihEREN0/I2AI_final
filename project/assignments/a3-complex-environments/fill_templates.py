"""Fills the {{path}} placeholders in templates/*.md from results/a3data.json; output goes next to this file under the same name.

Path examples: {{ls.algos.steepest.solved}} · {{ex.test.belief.0.beliefSize}} · {{ls.climbU.stops.local-max}}
Numbers are written as 1,234.5; a trailing "|..." format suffix (e.g. {{path|tr}}) is accepted and ignored.
Derived values: ls.climbU.never (climbs that never pushed a marble off), ratio.<method> (moves relative to A*).

Usage:
  python fill_templates.py                          # every templates/*.md -> this folder
  python fill_templates.py --out DIR T1.md T2.md    # other templates, written to DIR
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent

D = json.loads((HERE / "results" / "a3data.json").read_text(encoding="utf-8"))
CU, CM = D["ls"]["climbU"], D["ls"]["commonSolved"]["meanMoves"]
CU["never"] = CU["n"] - CU["anyCapture"]
D["ratio"] = {k: round(v / CM["astar"]) if v / CM["astar"] >= 10 else round(v / CM["astar"], 1)
              for k, v in CM.items() if k != "astar"}
# the talk and the templates also use these numbers in words; stop here if the data changes
assert sum(a["replies"] for a in D["ex"]["test"]["andor"]["answers"]) == 39 and D["ex"]["test"]["andor"]["lost"] == 0


def num(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v):,}"
    return str(v)


def fmt(v):
    if isinstance(v, dict):
        return " · ".join(f"{k} {num(x)}" for k, x in v.items())
    if isinstance(v, list):
        return ", ".join(fmt(x) for x in v)
    return num(v)


def get(path):
    v = D
    for part in path.split("."):
        v = v[int(part)] if isinstance(v, list) else v[part]
    return v


def fill(text):
    # a "|format" suffix from the old bilingual templates is ignored
    out = re.sub(r"\{\{([^}|]+)(\|[^}]*)?\}\}", lambda m: fmt(get(m.group(1).strip())), text)
    assert "{{" not in out
    return out


if __name__ == "__main__":
    args, out = sys.argv[1:], HERE
    if "--out" in args:
        i = args.index("--out")
        out = pathlib.Path(args[i + 1])
        del args[i:i + 2]
    for t in [pathlib.Path(a) for a in args] or sorted((HERE / "templates").glob("*.md")):
        (out / t.name).write_text(fill(t.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
        print("ok", out / t.name)

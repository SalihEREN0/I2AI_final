"""Command line for the final agent. Run from the repository root with the package on the path:

  PYTHONPATH=src python -m abalone search [--depth 4] [--start standard] [--weights w.json]
        one decision from a start position, printed as the developer-view trace (JSON)
  PYTHONPATH=src python -m abalone match A B [--openings 10] [--out DIR]
        a match under fixed conditions; agent specs are described in abalone/match.py
  PYTHONPATH=src python -m abalone train --out DIR [--iterations 50] [--games 8] [--depth 2] [--alpha 1] [--alpha-decay 0]
        [--lam 0.7] [--reward marbles|result]
        TDLeaf(λ) self-play training; writes weights-NNNN.json checkpoints and log.jsonl
  PYTHONPATH=src python -m abalone fit --out DIR [--games 1200] [--compare name=weights.json ...]
        logistic regression of game outcomes on the features (Chapter 19); writes weights and report.json
  PYTHONPATH=src python -m abalone net --out DIR [--hidden 32] [--lr 0.01] [--decay 0.9]
        a value network on the raw board, trained on the Chapter 19 positions (Chapter 22)
"""
import argparse
import json
from pathlib import Path

from .engine import START, ascii_board
from .search import Searcher
from . import match, tdleaf, supervised, valuenet
from . import features as F


def main():
    ap = argparse.ArgumentParser(prog="abalone")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--depth", type=int, default=4)
    s.add_argument("--start", default="standard", choices=list(START))
    s.add_argument("--weights")
    s.add_argument("--exact", action="store_true", help="exact values for every root move (slower)")
    m = sub.add_parser("match")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--openings", type=int, default=10)
    m.add_argument("--opening-plies", type=int, default=2)
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--out")
    m.add_argument("--trace", action="store_true", help="keep every move's trace in games.jsonl")
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--iterations", type=int, default=50)
    t.add_argument("--games", type=int, default=8)
    t.add_argument("--depth", type=int, default=2)
    t.add_argument("--alpha", type=float, default=1.0)
    t.add_argument("--alpha-decay", type=float, default=0, help="T: step size alpha / (1 + (t-1)/T); 0 keeps it constant")
    t.add_argument("--lam", type=float, default=0.7)
    t.add_argument("--reward", default="marbles", choices=["marbles", "result"])
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--init", help="start from these weights instead of the hand-set U")
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True)
    f.add_argument("--games", type=int, default=1200)
    f.add_argument("--depth", type=int, default=2)
    f.add_argument("--seed", type=int, default=19)
    f.add_argument("--compare", nargs="*", default=[], help="name=path/to/weights.json, judged on the same test games")
    f.add_argument("--data", help="learn from this folder's games.jsonl instead of generating games")
    n = sub.add_parser("net")
    n.add_argument("--out", required=True)
    n.add_argument("--hidden", type=int, default=32)
    n.add_argument("--lr", type=float, default=0.01)
    n.add_argument("--decay", type=float, default=0.9)
    n.add_argument("--epochs", type=int, default=40)
    n.add_argument("--patience", type=int, default=6)
    n.add_argument("--data", default="experiments/ch19-logistic", help="folder whose games.jsonl to learn from")
    a = ap.parse_args()

    if a.cmd == "search":
        w = json.loads(Path(a.weights).read_text()) if a.weights else None
        B, W = START[a.start]
        print(ascii_board(B, W))
        _, tr = Searcher(weights=w, depth=a.depth, root_exact=a.exact).choose(B, W)
        print(json.dumps(tr, indent=1))
    elif a.cmd == "match":
        print(json.dumps(match.run(a.a, a.b, a.openings, a.opening_plies, a.seed, a.out, keep_trace=a.trace), indent=1))
    elif a.cmd == "train":
        init = json.loads(Path(a.init).read_text()) if a.init else None
        tdleaf.train(a.out, a.iterations, a.games, a.depth, a.alpha, a.lam, a.reward, seed=a.seed, init=init,
                     alpha_decay=a.alpha_decay)
    elif a.cmd == "fit":
        compare = {"hand": F.HAND}
        for kv in a.compare:
            name, path = kv.split("=", 1)
            compare[name] = json.loads(Path(path).read_text())
        print(json.dumps(supervised.run(a.out, a.games, a.depth, a.seed, compare=compare, data=a.data), indent=1))
    elif a.cmd == "net":
        r = valuenet.train(a.out, data_dir=a.data, hidden=a.hidden, epochs=a.epochs, lr=a.lr, decay=a.decay,
                           patience=a.patience)
        print(json.dumps({k: v for k, v in r.items() if k != "log"}, indent=1))


if __name__ == "__main__":
    main()

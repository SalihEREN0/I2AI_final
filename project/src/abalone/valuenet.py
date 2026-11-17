"""A small value network on the raw board (Chapter 22), and an alpha-beta agent that uses it.

The linear U sees the board through six features we chose. This network sees the board itself: 122 inputs
(61 cells × "marble of the side to move" / "marble of the opponent"), one hidden layer of ReLU units and one
output, the probability that the side to move wins:

    h = ReLU(b₁ + Σ_{active inputs i} W₁[i])        p = σ(b₂ + w₂·h)

At most 28 of the 122 inputs are on, so the first layer is a sum of ≤ 28 rows of W₁, never a full matrix
product. That keeps both training and play feasible in pure Python. It is the same idea that makes NNUE
cheap in chess engines (Nasu 2018), without NNUE's incremental update.

Data: exactly the quiet positions and the train/test split of the Chapter 19 logistic regression
(experiments/ch19-logistic), so both models are judged on the same 240 held-out games. 10% of the training
games are held back again to pick the epoch (early stopping). During training every position is shown under
a random one of the board's 12 symmetries (6 rotations × mirror), which multiplies the data without changing
any label.

In search the network's logit z is used as the value, scaled by K/2 like the logistic weights (z·K/2), so all
evaluations share one scale and the repetition draw (0) means the same to each.
"""
import json
import math
import random
import time
from pathlib import Path

from .engine import CELLS, IDX, bits
from .search import Searcher
from . import supervised as S
from .tdleaf import K

N = len(CELLS)


# ------------------------------------------------------------------ the 12 symmetries of the hexagonal board
def _cube(r, n):
    a, b = n - 5, r - 4                      # axial: E = (a+1, b), NE = (a+1, b+1), NW = (a, b+1)
    return a, b - a, -b                      # cube (x, y, z), x + y + z = 0


def _cell(x, y, z):
    a, b = x, -z
    return IDX.get((b + 4, a + 5), -1)


def _symmetries():
    out = []
    for mirror in (False, True):
        for rot in range(6):
            perm = []
            for r, n in CELLS:
                x, y, z = _cube(r, n)
                if mirror:
                    y, z = z, y
                for _ in range(rot):
                    x, y, z = -z, -x, -y        # 60° rotation
                perm.append(_cell(x, y, z))
            out.append(perm)
    return out


SYMMETRIES = _symmetries()


def transform(mask, perm):
    m = 0
    for c in bits(mask):
        m |= 1 << perm[c]
    return m


# ------------------------------------------------------------------ the network
class ValueNet:
    def __init__(self, hidden=32, seed=22):
        rng = random.Random(seed)
        self.hidden = hidden
        a = 1 / math.sqrt(28)                                   # ≈ 1/√(active inputs)
        self.W1 = [[rng.uniform(-a, a) for _ in range(hidden)] for _ in range(2 * N)]
        self.b1 = [0.0] * hidden
        self.w2 = [rng.uniform(-1 / math.sqrt(hidden), 1 / math.sqrt(hidden)) for _ in range(hidden)]
        self.b2 = 0.0

    @staticmethod
    def active(P, O):
        return [c for c in bits(P)] + [N + c for c in bits(O)]

    def hidden_layer(self, act):
        h = self.b1[:]
        W1 = self.W1
        for i in act:
            h = [x + y for x, y in zip(h, W1[i])]
        return h

    def logit(self, P, O):
        h = self.hidden_layer(self.active(P, O))
        return self.b2 + sum(w * x for w, x in zip(self.w2, h) if x > 0)

    def prob(self, P, O):
        return S.sigmoid(self.logit(P, O))

    def step(self, act, t, lr):
        """One SGD step on the log-loss of one example. Returns the example's loss before the step."""
        pre = self.hidden_layer(act)
        h = [x if x > 0 else 0.0 for x in pre]
        z = self.b2 + sum(w * x for w, x in zip(self.w2, h))
        p = S.sigmoid(z)
        g = p - t                                               # d loss / d z
        dh = [g * w if x > 0 else 0.0 for w, x in zip(self.w2, pre)]
        self.w2 = [w - lr * g * x for w, x in zip(self.w2, h)]
        self.b2 -= lr * g
        self.b1 = [b - lr * d for b, d in zip(self.b1, dh)]
        for i in act:
            row = self.W1[i]
            self.W1[i] = [w - lr * d for w, d in zip(row, dh)]
        p = min(max(p, 1e-12), 1 - 1e-12)
        return -(t * math.log(p) + (1 - t) * math.log(1 - p))

    def to_json(self):
        return {"hidden": self.hidden, "W1": self.W1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    @classmethod
    def from_json(cls, d):
        net = cls(d["hidden"])
        net.W1, net.b1, net.w2, net.b2 = d["W1"], d["b1"], d["w2"], d["b2"]
        return net


def evaluate_set(net, data):
    ll, right, decisive = 0.0, 0, 0
    for P, O, t in data:
        p = min(max(net.prob(P, O), 1e-12), 1 - 1e-12)
        ll -= t * math.log(p) + (1 - t) * math.log(1 - p)
        if t != 0.5:
            decisive += 1
            right += (p > 0.5) == (t == 1.0)
    return {"log_loss": round(ll / len(data), 4), "accuracy_decisive": round(right / decisive, 4),
            "positions": len(data), "decisive_positions": decisive}


def train(out, data_dir="experiments/ch19-logistic", hidden=32, epochs=30, lr=0.01, decay=0.9, patience=4,
          seed=22, augment=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    recs = S.load(data_dir)
    test_ids = S.test_games(len(recs))
    rng = random.Random(seed)
    train_ids = [i for i in range(len(recs)) if i not in test_ids]
    rng.shuffle(train_ids)
    val_ids = set(train_ids[:len(train_ids) // 10])
    tr, va, te = [], [], []
    for i, rec in enumerate(recs):
        bucket = te if i in test_ids else va if i in val_ids else tr
        bucket.extend(S.positions(rec))
    net = ValueNet(hidden, seed)
    log, best, best_epoch, bad = [], None, 0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        rng.shuffle(tr)
        loss = 0.0
        for P, O, t in tr:
            if augment:
                perm = SYMMETRIES[rng.randrange(12)]
                P, O = transform(P, perm), transform(O, perm)
            loss += net.step(ValueNet.active(P, O), t, lr)
        v = evaluate_set(net, va)
        row = {"epoch": ep, "lr": round(lr, 5), "train_log_loss": round(loss / len(tr), 4),
               "val_log_loss": v["log_loss"], "val_accuracy": v["accuracy_decisive"],
               "seconds": round(time.time() - t0, 1)}
        log.append(row)
        print(row, flush=True)
        if best is None or v["log_loss"] < best:
            best, best_epoch, bad = v["log_loss"], ep, 0
            (out / "net.json").write_text(json.dumps(net.to_json()))
        else:
            bad += 1
            if bad >= patience:
                break
        lr *= decay
    net = ValueNet.from_json(json.loads((out / "net.json").read_text()))
    report = {"data": data_dir, "hidden": hidden, "parameters": 2 * N * hidden + hidden + hidden + 1,
              "augment_symmetries": augment, "train_positions": len(tr), "val_positions": len(va),
              "test_positions": len(te), "best_epoch": best_epoch, "log": log,
              "test": evaluate_set(net, te), "train_seconds": round(time.time() - t0, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


# ------------------------------------------------------------------ alpha-beta with the network as its evaluation
class NetSearcher(Searcher):
    """The same alpha-beta (Chapter 6), with the value network in place of the linear U."""

    def __init__(self, net, **kw):
        super().__init__(**kw)
        self.net = net
        self.scale = K / 2

    def evaluate(self, P, O):
        return self.net.logit(P, O) * self.scale

    def _trace(self, *args):
        t = super()._trace(*args)
        t["why"] = None                    # the feature breakdown explains a linear U, not the network
        t["evaluation"] = "value network (%d hidden units)" % self.net.hidden
        for c in t["candidates"]:
            c.pop("static", None)
        return t


def load_net(path):
    return ValueNet.from_json(json.loads(Path(path).read_text()))

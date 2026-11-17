# experiments

One folder per training run or match, with the settings it ran with (`config.json`, or the player specs in
`summary.json`) and its results. Nothing here is edited by hand.

| Folders | What | Made by |
|---|---|---|
| `e1-*`, `e2-ablate-*`, `e4-learning-curve/`, `e5-*` | Assignment 4 evidence with the presented engine (`src/abalone_a4`) | `sh experiments/run-evidence.sh` (its output: `evidence.log`) |
| `match-tdleaf-*`, `ch4-andor-vs-alphabeta/` | TDLeaf against the hand-set weights at depths 2 and 3; AND-OR against alpha-beta on forced push-offs | `python -m abalone match A B --openings 25 --out experiments/<match>`, players as in `summary.json`; `python experiments/ch4-andor-vs-alphabeta/run.py` |
| `tdleaf-d2-marbles/`, `tdleaf-d2-result/` | the two 150-iteration TDLeaf runs, marble reward and result reward | `python -m abalone train`; settings in each `config.json` |
| `ch19-*`, `ch22-*`, `refit-*` | logistic regression and the value network (Chapters 19 and 22), and their matches | `python -m abalone fit` / `net` (options and the learning rate of each epoch in `report.json`); their matches: `python -m abalone match A B --openings 25 --out experiments/<match>`, players as in `summary.json` |
| `tdleaf-*-400*/`, `e4-learning-curve-400/`, `convergence*/` | convergence check: 400 iterations with a decaying step size, three seeds | last section of `run-evidence.sh` |
| `search-cost/` | nodes and time at fixed depth, presented engine | `python experiments/search-cost/run.py` |
| `engine-speed/` | the faster engine against the presented one: the same search on 386 positions, and the depth reached in 0.5 s | `python experiments/engine-speed/run.py` |
| `final/` | every run with the faster engine (November 2026): matches, ablations, MCTS, minimax, search cost, depth in 0.5 s | `sh experiments/run-final.sh` (its output: `final-run-stdout.log`); see [final/README.md](final/README.md) |

Timed results (`engine-speed/`, `final/search-cost/`, `final/baselines-depth/`) come from runs on AC power with the
process pinned to one fast core (`src/abalone/steady.py`, `engine-speed/cores.log`). The earlier unpinned runs are
kept as `*-unpinned.*` for the record and are not used. Games played in the demo app are saved to `demo-games/`,
which is not published.

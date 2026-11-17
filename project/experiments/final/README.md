# Experiments with the faster engine (November 2026)

The engine in [src/abalone](../../src/abalone) searches exactly like the agent presented in Assignment 4
([src/abalone_a4](../../src/abalone_a4)) at a fixed depth (same moves, values and node counts;
[tests/test_speed.py](../../tests/test_speed.py)) but 5.5 times faster at depth 3 and 4.5 times at depth 4
([experiments/engine-speed](../engine-speed)). With 0.5 s per move it completes 4.1 plies on average instead of 3.3.
These runs repeat the time-limited experiments of Assignment 4 with it and add the two alternatives of Chapter 6.

Run from the repository root: `sh experiments/run-final.sh` (sequential; each match uses all CPU cores; a finished
match is skipped, so an interrupted run continues where it stopped).

| Folder | What it measures |
|---|---|
| `final-vs-a4` | the current engine against the Assignment 4 engine: same weights, same clock, 100 games |
| `e1-final-vs-a3-andor`, `e1-hand-vs-a3-andor` | progress over the Assignment 3 agent (greedy on U, AND–OR for a forced push-off) |
| `e2-ablate-*` | the final agent against itself with one part switched off, 50 games; `-200`: the same with 200 games |
| `mcts-vs-final` | Monte Carlo tree search ([src/abalone/mcts.py](../../src/abalone/mcts.py)) with the final agent's evaluation |
| `minimax-vs-final` | minimax without alpha-beta (`MinimaxSearcher`), same evaluation, quiescence and clock |
| `search-cost` | nodes and time per move at a fixed depth, with minimax for reference |
| `baselines-depth` | how far each searcher gets in 0.5 s (one process, nothing else running): completed depth of alpha-beta on both engines and of minimax, iterations of MCTS |

Every match: 0.5 s per move, 25 (or 50, 100) random two-ply openings from both start layouts, each played with both
colours; third repetition is a draw; 200-ply cap decided by marble count. Each folder holds the games
(`games.jsonl`) and `summary.json`. The results are tabulated in the supplementary material (Tables S11–S13).

Timing (`search-cost`, `baselines-depth` and [experiments/engine-speed](../engine-speed)): one process, pinned to one
logical CPU with high priority and Windows power throttling switched off ([src/abalone/steady.py](../../src/abalone/steady.py)),
on AC power, nothing else running; each `summary.json` records these conditions. A first, unpinned run gave the same
node counts but times that disagreed by up to a factor of 1.8 for the same search: on this laptop the same searches
take 35–50% longer on the slower Zen 5c cores and 2.2 times as long on battery
([engine-speed/cores.log](../engine-speed/cores.log)). That run is kept as `summary-unpinned.json` and `run-unpinned.log` and not
used.

Machine: AMD Ryzen AI 9 HX 370 (12 cores, 24 threads), Windows 11, Python 3.14, 24 games in parallel. The
Assignment 4 results in `experiments/` came from a faster computer: the same depth-3 searches took 99 ms there
(`experiments/search-cost`) and 122 ms here on the fastest core with the same engine (`experiments/engine-speed`). Engines are therefore
compared only within one run; `final-vs-a4` plays both on the same machine under the same conditions.

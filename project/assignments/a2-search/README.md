# Assignment 2 · Abalone as a search problem (Chapter 3)

Presented on 14 October 2026. Format: at most 5 minutes per pair including Q&A; the talk itself was 3:45–4:00.

## Files

| File | Purpose |
|---|---|
| [deck.html](deck.html) | **The deck** (8 main slides + 3 backups). Fonts are embedded, so it works offline. → / space: next · ←: back · N: speaker notes and timer · F: fullscreen · R: replay the step |
| [abalone_search.py](abalone_search.py) | Rules engine (two independently written move generators), BFS, UCS, A*, IDA*, weighted A*, greedy search, the heuristics, and every measurement |
| [search-comparison.md](search-comparison.md) | Search algorithms and heuristics compared: the ones on the slides, advanced methods from the textbook, and recent literature. IDA*, IDA* + TT and weighted A* measured on the same 60 puzzles |
| `results/` | Measurement output (JSON + logs). The deck reads these from its embedded `DATA` block |

## Reproducing the numbers

Run from this folder. Python 3, standard library only.

```bash
python abalone_search.py count          # legal moves at the start and in the Week 03 mid-game position (44 / 73)
python abalone_search.py states         # state-space size; our evaluation of Lemmens's formula
python abalone_search.py verify 2000    # fast vs naive generator on 2,000 positions
python abalone_search.py tree 4         # perft(1–4), distinct boards, speed, IDS overhead (~15 s)
python abalone_search.py bench 60 314 1000000   # 60 puzzles, five searches (~4 min)
python abalone_search.py demo           # the plan the opponent breaks, and the slide 6 puzzle (~1 min)
python abalone_search.py more 60        # IDA*, IDA* + transposition table, weighted A* (w = 2) on the same puzzles (~1.5 min)
python abalone_search.py inject         # results/*.json -> the DATA block in deck.html
```

## Main findings

- **Formulation:** a state is 123 bits; ACTIONS(start) = 44; RESULT is deterministic (sumito included); ACTION-COST = 1 (moves are counted, not marbles); the goal is 6 marbles.
- **Abstraction:** the rules are exact (two generators agree on 2,000 positions); what is lost is the opponent's choice: the plan breaks on 6 of White's 75 replies.
- **Uninformed search:** 5,045,110 paths and 1,358,966 distinct boards at ply 4; IDS overhead 2.0%; BFS at ply 10 ≈ 10 EB.
- **Heuristics:** hR (looks at the opponent) is admissible but blind; hF (formation, from a relaxed problem) is consistent. On 60 puzzles A* with hF generates 31× fewer nodes than BFS at d = 3, b* 42.6 → 13.4, always the shortest solution; 0 consistency violations on 103,125 edges.

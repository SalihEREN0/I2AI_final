# Assignment 3 · Abalone through Chapter 4 (search in complex environments)

Presented on 21 October 2026. Format: at most 7 minutes per pair including Q&A; the talk itself was ≈ 6:00.

## Files

| File | Purpose |
|---|---|
| [deck.html](deck.html) | **The deck** (9 main slides + 4 backups). Fonts are embedded, so it works offline. → / space: next · ←: back · **N**: speaker notes and timer · **S**: Q&A and timing page · F: fullscreen · R: replay the step |
| [abalone_ch4.py](abalone_ch4.py) | **This assignment's lab:** test board, AND–OR search, observation and reversibility counts, the opening puzzle, seven local search methods on 200 puzzles |
| [ch4-comparison.md](ch4-comparison.md) | Chapter 4 methods compared: the ones on the slides, the others in the textbook, and recent literature; discussion. Generated from [templates/ch4-comparison.md](templates/ch4-comparison.md) |
| [fill_templates.py](fill_templates.py) | Fills the `{{path}}` placeholders in `templates/*.md` from `results/a3data.json`, so no number in the documents is typed by hand |
| `results/` | Measurement output (JSON + logs). The deck reads these from its embedded `window.A3` block |

The rules engine, both move generators, hF and A* are imported unchanged from Assignment 2: [../a2-search/abalone_search.py](../a2-search/abalone_search.py).

## Reproducing the numbers

Run from this folder. Python 3, standard library only.

```bash
python abalone_ch4.py recheck        # Assignment 2's numbers again with the same engine: are all four identical?
python abalone_ch4.py test           # the test board (slides 1–4), ~1 min
python abalone_ch4.py opening        # the opening puzzle (slides 5–6)
python abalone_ch4.py local 200 2027 # 200 puzzles, seven local search methods (~20 min)
python abalone_ch4.py inject         # results/*.json -> window.A3 in deck.html (and results/a3data.json)
python fill_templates.py             # templates/*.md -> ch4-comparison.md
```

- `recheck`: Assignment 2's numbers with the same engine (44/73/81 moves, perft 1–3, 75 replies / 6 breaking, A* 439 nodes).
- `test`: 73 moves, 39 replies, 37 / 2, the AND–OR plan and its independent check, U's greedy choice, the counterfactual, belief-state size, reversibility, 1,936 paths back to the opening.
- `opening`: steepest descent with hF and its plateau, two-move exits, climbing U and its flat top.
- `local`: steepest, sideways, stochastic, first choice, annealing, LRTA*, 20 restarts and climbing U; stuck runs classified as shoulder / ridge / flat optimum.

## Main findings

- **Who decides (4.3):** our move is deterministic (OR), White's reply is a set (AND). The "push, push again" plan that ignores White works on 37 of 39 replies; the AND–OR conditional plan handles all 39 (2 marbles in 2 moves, no marble lost). Of 73 first moves, only the capture guarantees this.
- **Decision:** U's greedy choice ignores White (it ranks 3 quiet moves above the capture; after its favourite, 7 of 43 replies end the guarantee) → **AND–OR**.
- **Observation (4.4):** fully observable, PERCEPT(s) = s, a single possible board. If we only saw next to our own marbles, the belief state would hold C(30, 13) = 119,759,850 boards.
- **Reversibility (4.5):** 72 of 73 moves can be undone by the reverse move; a capture never. Since the rules are known, no online exploration is needed.
- **Local search (4.1):** steepest descent on hF gets stuck on plateaus; the plateaus have exits (shoulders), not ridges. Climbing U stops at a non-attacking "fortress". Sideways moves and annealing escape but spend many times more moves than A* → **A* and AND–OR for moves**; local search only for tuning U's weights (4.2, proposed).

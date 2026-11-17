# Assignment 4 · the final agent (Chapters 1–4, 6, 19, 22, 23)

To be presented on 4 November 2026. Format: at most 11 minutes per pair, including the live demo and questions. The slides take about 5 minutes; the demo about 4.

## Files

| File | Purpose |
|---|---|
| [deck.html](deck.html) | **The deck** (7 main slides + 3 backups). It opens with our agent on one page. Fonts are embedded, so it works offline. → / space: next · ←: back · **N**: speaker notes and timer · **S**: script, demo plan and Q&A · F: fullscreen · R: replay the step |
| [collect.py](collect.py) | Reads every number the deck shows from `experiments/` into `results/a4data.json` and writes it into the deck's `window.A4` block, so no number on a slide is typed by hand |
| `results/a4data.json` | The collected numbers, one file |

The agent, its training and the demo app are in [src/abalone](../../src/abalone); every experiment, with the command that produced it, is in [experiments/](../../experiments) and described in [src/abalone/README.md](../../src/abalone/README.md). The research and decisions behind it: [docs/a4-research.md](../../docs/a4-research.md).

## Reproducing

From the repository root:

```bash
python -m unittest discover tests               # correctness checks
sh experiments/run-evidence.sh                  # E1, E2, E4–E6 (about two and a half hours on 8 cores)
python experiments/search-cost/run.py           # nodes per move with each part switched off
python assignments/a4-final-agent/collect.py    # experiments/ -> results/a4data.json -> deck.html
PYTHONPATH=src python -m abalone.server         # the demo, at http://localhost:8024
python tools/export_deck.py assignments/a4-final-agent/deck.html   # PNG frames and a PPTX backup
```

## Main findings

- **Progress since Assignment 3:** against the Assignment 3 agent as presented there (greedy on U, AND–OR when a push-off can be forced), the final agent won 48 of 50 games and lost none under the same conditions (50–0–0 against plain greedy U). Alpha-beta does most of that: with the hand-set U it already scores 0.99 against the A3 agent.
- **What each part contributes (equal time):** the learned weights matter most, then quiescence search and move ordering; the transposition table and killer moves show no effect in play at 50 games, but they still save nodes at a fixed depth.
- **Reinforcement learning:** TDLeaf(λ) self-play learned U's weights in minutes; most of the gain came in the first 120 games, and a result-only reward learned as well as the marble-difference reward. A 3,200-game run with three seeds plays even with the 1,200-game one: strength levels off by ~1,300 games, and the weights still moving (edge, exposed; correlated 0.71) barely change play (E6).
- **Learning from examples:** logistic regression predicts results best and a small value network as well as TDLeaf, but both lose to TDLeaf in play. Refit on TDLeaf's own games, logistic regression plays level: what RL added is that it makes its own training games.
- **Literature:** no RL method has been shown to solve Abalone or beat strong alpha-beta; we integrated the two methods known to work at our scale and reviewed the rest.

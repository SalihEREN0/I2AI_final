# abalone-ai

An AI agent for the board game **Abalone**, built step by step through the chapters of Russell & Norvig's *Artificial Intelligence: A Modern Approach* (4th ed.). Each stage looks at the same game through a new chapter, measures what that chapter's methods actually do on it, and feeds the result into the next agent.

Built for the *Introduction to Artificial Intelligence* course at Eskişehir Osmangazi University (Fall 2026) by **Team Eight Bits**: Eren Terakye and Salih Eren.

## Stages

| Stage | Chapter | What it shows | Folder |
|---|---|---|---|
| 1 · Preliminary study | 1–2 Intelligent agents | the agent, its task environment (PEAS), and a utility-based design with utility U | [assignments/a1-preliminary-study](assignments/a1-preliminary-study) |
| 2 · Search problem | 3 Solving problems by searching | a bitboard rules engine checked by a second, independent generator; BFS, UCS, IDS, A*, IDA*, weighted A*, greedy; two heuristics and a proof of consistency | [assignments/a2-search](assignments/a2-search) |
| 3 · Complex environments | 4 Search in complex environments | AND–OR planning against the opponent's replies, observability and reversibility, seven local search methods on 200 puzzles | [assignments/a3-complex-environments](assignments/a3-complex-environments) |
| 4 · Final agent | 6 Adversarial search, 19/22/23 learning | alpha-beta with a per-move trace, evaluation weights learned by TDLeaf(λ) self-play, supervised and deep-learning comparisons, live demo with a developer view | [assignments/a4-final-agent](assignments/a4-final-agent), [src/abalone](src/abalone) |

Each stage folder has a self-contained HTML deck (open `deck.html` in a browser), the code that produced every number on it, and the raw results.

## Selected results

- **Rules you can trust.** Two independently written move generators produce the same resulting boards on 2,000 positions, including pushes and marbles pushed off. perft from the start: 44 · 1,936 · 98,912 · 5,045,110.
- **A heuristic beats hardware.** On 60 frozen-opponent puzzles, A* with our formation heuristic hF generates 31× fewer nodes than BFS at depth 3 (effective branching factor 42.6 → 13.4) and always finds the shortest plan; hF has 0 consistency violations on 103,125 edges. IDA* uses ≈ 1,300× less memory at depth 4.
- **The opponent matters.** A plan that ignores White works on 37 of 39 replies; an AND–OR conditional plan handles all 39, and of 73 first moves only one guarantees it.
- **Local search is the wrong tool for moves.** Hill climbing gets stuck on plateaus (mostly shoulders, not ridges); sideways moves and annealing escape but spend many times more moves than A*, and every extra move gives the opponent a turn.
- **Search does most of the work, learning adds the rest.** Against the Assignment 3 agent as presented (greedy on U, AND–OR when a push-off can be forced) the final agent won 48 of 50 games and lost none under the same conditions; alpha-beta with the hand-set U already scores 0.99. At equal time the learned weights are the largest single part (0.83 against hand-set), then quiescence and move ordering.
- **The reinforcement learning that works at this scale.** TDLeaf(λ) self-play learned U's weights in minutes, most of the gain in the first 120 games. Logistic regression (Ch. 19) predicts results best and a small value network (Ch. 22) as well as TDLeaf, but both lose to it in play: what RL added is that it makes its own training games. No published RL agent has beaten strong alpha-beta on Abalone.

## After Assignment 4: a faster engine (November 2026)

The agent presented in Assignment 4 is kept, unchanged, in [src/abalone_a4](src/abalone_a4). The current engine in [src/abalone](src/abalone) searches exactly like it, with the same moves in the same order, the same features and values, and the same nodes at a fixed depth ([tests/test_speed.py](tests/test_speed.py)), but faster: the features are computed for all marbles at once from six shifted bitboards, quiescence generates only the moves that push a marble off, and leaf values are cached.

- **5.5× faster at depth 3, 4.5× at depth 4** on the 386 positions of the search-cost study, with identical moves, values and node counts ([experiments/engine-speed](experiments/engine-speed); timed on one pinned core, on AC power). With 0.5 s per move it completes 4.1 plies on average instead of 3.3.
- **The extra ply wins games.** Same weights, same clock: the current engine beat the Assignment 4 engine 82–6–12 in 100 games, and the Assignment 3 agent 50–0–0.
- **The alternatives of Chapter 6**, minimax without pruning and Monte Carlo tree search ([src/abalone/mcts.py](src/abalone/mcts.py)), play with the final agent's own evaluation, so their matches compare only the ways of searching: against the final agent, minimax scored 2–3–45 and MCTS 0–0–50. All runs: [experiments/final](experiments/final), `sh experiments/run-final.sh`.
- **The midterm paper** and its supplementary material report all four stages; both are in the repository root.

Details and literature comparisons: [search-comparison.md](assignments/a2-search/search-comparison.md), [ch4-comparison.md](assignments/a3-complex-environments/ch4-comparison.md), the final agent's research and decisions in [docs/a4-research.md](docs/a4-research.md), and a research plan for deep RL on Abalone in [docs/rl-literature-and-plan.md](docs/rl-literature-and-plan.md).

## The final agent: where to start

- **The deck:** [assignments/a4-final-agent/deck.html](assignments/a4-final-agent/deck.html), opening with the agent on one page. It works offline; every number on it is read from `experiments/` and its source is on backup slide 8.
- **The live demo:** `PYTHONPATH=src python3 -m abalone.server`, then http://localhost:8024. User mode shows the game; developer mode shows the moves the agent considers, their values and why it plays one, the alpha-beta tree, the AND–OR push-off plans, the learned weights and a live ablation. Python 3 standard library only.
- **If the demo cannot run:** the recording in [demo-video/](demo-video), and its games replayable from the app's replay menu.
- **The evidence:** [experiments/](experiments), one folder per match or training run with its settings and results; [assignments/a4-final-agent/README.md](assignments/a4-final-agent/README.md) lists the commands that reproduce them. Correctness checks: `python3 -m unittest discover tests`.

## Reproducing

The experiments need only Python 3 (standard library). From a stage folder:

```bash
cd assignments/a2-search
python abalone_search.py verify 2000   # cross-check the two move generators
python abalone_search.py bench         # the 60-puzzle search benchmark (~4 min)

cd ../a3-complex-environments
python abalone_ch4.py recheck          # re-derive Assignment 2's numbers with the same engine
python abalone_ch4.py local 200 2027   # local search on 200 puzzles (~20 min)
```

Each stage README lists every command and what it produces. Exporting decks to PPTX/PDF uses the scripts in [tools/](tools) (`pip install -r requirements.txt`).

## Layout

```
assignments/   one folder per stage: deck, code, results (frozen once presented)
src/abalone/   the agent package for the final stage: engine, alpha-beta with traces, TDLeaf(λ) training, MCTS
src/abalone_a4/ the agent exactly as presented in Assignment 4, frozen for comparison
tests/         correctness checks (python -m unittest discover -s tests), incl. the equivalence of both engines
experiments/   training logs, learned weights and match results, each with the command's settings;
               experiments/final/ holds the runs with the faster engine
docs/          research notes
demo-video/    a recording of the demo, the fallback if it cannot run live
paper/         the IEEE-format paper reporting the project
tools/         deck → PPTX export, Markdown → PDF
assets/        team logo
```

## Licence

Code under the MIT License; documents and slides under CC BY 4.0. See [LICENSE](LICENSE).

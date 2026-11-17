# abalone: the final agent (Assignment 4)

Plan and literature: [docs/a4-research.md](../../docs/a4-research.md). Python 3, standard library only.

| Module | Chapter | Contents |
|---|---|---|
| [engine.py](engine.py) | 3 | rules engine ported from Assignment 2 (two generators, perft), Belgian Daisy start, push-off prefilter |
| [features.py](features.py) | 2, 19 | zero-sum features and the linear evaluation U = Σ wᵢ·fᵢ; `HAND` = Assignment 1's U |
| [search.py](search.py) | 6 | alpha-beta with iterative deepening, TT, move ordering, killers, quiescence (each switchable) and the per-move trace |
| [game.py](game.py) | — | game loop and match rules (repetition, 200-ply cap), the A3 agent (greedy on U + AND–OR) and plain greedy U, random agent |
| [match.py](match.py) | — | matches with fixed openings, both colours, score with a 95% CI |
| [tdleaf.py](tdleaf.py) | 23 | TDLeaf(λ) self-play training of the weights |
| [supervised.py](supervised.py) | 19 | logistic regression of game outcomes on the same features (Texel/Buro), from saved self-play games |
| [valuenet.py](valuenet.py) | 22 | a value network on the raw board (122 inputs, one hidden layer), its 12 board symmetries, and alpha-beta using it |
| [andor.py](andor.py) | 4 | AND–OR search for forced push-offs, carried over from Assignment 3 (reproduces its test-board result) |
| [server.py](server.py), [ui/index.html](ui/index.html) | all | the demo app: user mode and developer mode, human or agent players, live ablation, replay of saved games |

```bash
python -m unittest discover tests                       # 25 checks: perft, generators, alpha-beta = minimax, TDLeaf
PYTHONPATH=src python -m abalone search --depth 4        # one decision with its full trace (JSON)
PYTHONPATH=src python -m abalone train --out experiments/tdleaf-d2-marbles --iterations 150 --alpha 10
PYTHONPATH=src python -m abalone match "ab:depth=2,weights=experiments/tdleaf-d2-marbles/weights-final.json" ab:depth=2 --openings 25
PYTHONPATH=src python -m abalone fit --out experiments/ch19-logistic --compare tdleaf-marbles=experiments/tdleaf-d2-marbles/weights-final.json
sh experiments/run-evidence.sh                         # all evidence runs (vs A3, ablations, learning curve, reward), ~1.5 h
PYTHONPATH=src python -m abalone.server                # the demo app on http://localhost:8024 (works offline)
```

## The demo app

**User mode** shows the board, the score (marbles pushed off) and whose turn it is. **Developer mode** (press <kbd>D</kbd>) puts the developer view next to the board, on one screen. Everything it draws comes from the trace the search returned:

- **Agent loop (Ch. 2):** percept → state → decide → act, along the top. Decide lights up while the agent searches and Act when it plays; percept and state take no measurable time, so they are not staged.
- **The move, step by step.** While the agent thinks, its best moves after each finished depth appear on the board and in the list, and change as it looks deeper (`/api/progress`). The moment the search returns, the move is drawn as played: the decision is made, and the page does not stage a comparison it is not making. The other moves appear on the board only once the check after the move (below) has ranked them, captioned as such; then the marbles slide. Every move on the board is drawn the same way: an outline round the marbles that move, an arrow one step the way they go, a ring where a marble lands, dashed rings on the opposing marbles a push shifts and an "off" mark where one falls off. Candidates that share marbles get nested outlines. After an agent's move, its **predicted reply** (move 2 of the line it searched) is drawn dashed. *Reveal* sets the pace (slow / normal / off).
- **Decision tab (Ch. 2, 6, 19):** the move in two stages, each named on screen. *What it knew when it chose:* the decision's own numbers, the chosen move's value and an upper bound for the others. *Checked after the move:* how good the other moves really were, ranked, with value, expected score and the positions the decision spent on each. During the decision alpha-beta only proves the other moves *no better* than the best (an upper bound, which in most positions is the same number for all of them), and its search order is a poor guide to which alternatives are best: the true top eight came from as deep as 44th place. So once the move is chosen the page asks `/api/exact`, and the server searches every legal move again at the same depth and weights, multi-PV style (`Searcher.exact_values(keep=5)`): each move only until it is clear whether it beats the fifth best so far. The five best get exact values (checked against brute force and minimax in the tests), the chosen one exactly the value it was chosen with, and one row bounds all the others. It runs after the move is chosen, outside the agent's time budget and without touching its tables (on an M3, 30 plies of final vs final: 0.4–0.7 s in the opening, 1.5–2.2 s in the middlegame, 1.3 s on average; at normal reveal speed the marbles move about 1.6 s after the decision, longer when the check is slow; the next search waits for the check, so its thinking time is never time spent waiting). It stops after 3 s: the list then says how many moves it reached, and the rows it did not reach keep their bounds. Until the values arrive, and in games played in user mode, the list is headed as what it is, the first moves in search order. When the values arrive the rows slide to their ranks and each dot to its value. Each row shows the value as a dot and, as a ring, where alpha-beta stopped during the decision, with a dashed line at the chosen move's value: the pruning made visible. One sentence, marked as checked after the move, says how clear the choice was (the gap to the next best, in value and expected score), and a bar shows where the deepest pass spent its positions (most on the chosen move, the rest cut off with a few hundred each). The board labels the top four with their ranks and expected scores. While the agent thinks, the live list says that only its leader's value is exact. Hover a move to draw its best line on the board; click it to compare it with the chosen move in *Why*: U's feature contributions for both, at the end of their lines or right after the move, with the difference said in words. In user mode the values are not computed; games saved without them replay with the bounds.
- **Search tab (Ch. 3, 6):** depth, positions, positions/s, b* (the textbook effective branching factor: the b with b + b² + … + b^d equal to the deepest pass's positions, quiescence left out), cutoffs, TT hits; each iterative-deepening pass, live while it runs, on a log scale with how many times the pass before it cost. Under the numbers, **the alpha-beta tree**: the top two levels of the deepest pass as the search ran them (`trace.tree`, recorded only by the demo's agents). Every move (MAX) in search order with its value, or ≤ where it was only proved no better than α, the best value so far; open one to see the opponent's replies (MIN) tried before the **β-cutoff**, and how many were pruned, never searched. In the opening, most moves are refuted by their first reply. The chosen move is always shown, open, wherever the search found it; hover any node to draw its line. A test checks that recording the tree changes nothing in the search and that every value in it is the minimax value or a correct bound.
- **Push-offs tab (Ch. 4):** AND–OR search, after every move, for 1 or 2 marbles forced off within 2 moves: the forcing first moves and the plan as an OR/AND tree (hover a row to draw it), and whether alpha-beta played a forcing move.
- **Learning tab (Ch. 19, 22, 23):** hand-set, TDLeaf and logistic-regression weights, each weight's path during self-play, the E4 learning curve and the learned models' match results.
- **Ablation tab (Ch. 6, 23):** re-search the position with parts switched off next to the full agent (both moves drawn on the board), with the measured E1 and E2 results and the fixed-depth search cost.
- **Game timeline and eval bar:** Black's expected score after every move (win 1, draw ½, loss 0), one line per agent. Each agent estimates from the end of its own searched line, so the two disagree (shaded); one line through both would zigzag. The axis zooms to the game; push-offs, repeated positions and the repetition count (third time is a draw) are marked. While an agent thinks, its current estimate pulses at the next ply. Clicking a move opens that decision with its full trace.
- **Expected score, and where it comes from.** A search value v becomes an expected score σ(c·v), with c = 0.0304 fitted by logistic regression on how 1,200 of TDLeaf's self-play games ended, for the final agent's weights (`experiments/refit-logistic-on-tdleaf-games/report.json`, Ch. 19); the server reads c from that file. It is an expected score, not a win chance: half of those games were draws, which count ½. TDLeaf's own K = 30 is a training setting and was never checked against results: it would read +10 as 66% where the fit says 58%. The fit is on positions scored by U, and the demo applies it to search values, which are on the same scale.
- **Playing it yourself.** Drag one of your marbles: it moves one step that way together with any of your marbles in front of it, so a flick from the back of a line pushes. While you drag, ghosts show where every marble lands (and which one goes off); an illegal drag stops at a bar, rings what blocks it and says why (for example "2 vs 2" or "edge"). Drag back onto the marble or press <kbd>Esc</kbd> to cancel. For a side step, click 2–3 marbles in a line (clicking both ends of a three fills in the middle), then drag one or click an arrow. One clicked marble gets an arrow for each line it can push. **Hint** (<kbd>H</kbd>) asks the final agent for its move from your side, drawn on the board. In developer mode your own moves get values too: on your turn the final agent searches from your side (0.5 s, as the hint) and the exact pass ranks your best moves (`/api/options`). The board labels the top four with their rank and your expected score, the first in the hint's colour, in place of the predicted reply. It runs outside the server's lock and your next move stops it, so playing quickly is never held up and the agent's own search never shares the processor with it.

Every number from `experiments/` is shown with its source file. Every game is saved to `experiments/demo-games/` (not committed) and can be replayed move by move from the replay menu, as a backup if a live game goes wrong. New game offers ready matchups (you vs the final agent, final vs the Assignment 3 agent, learned vs hand-set weights, final vs the value network). Keys: <kbd>Space</kbd> agent move, <kbd>A</kbd> autoplay, <kbd>U</kbd> undo, <kbd>H</kbd> hint, <kbd>1</kbd>–<kbd>5</kbd> tabs, <kbd>N</kbd> new game, <kbd>?</kbd> help.

## First results (2026-10-27, Apple M3, 8 processes)

- **Correctness.** perft 44 · 1,936 · 98,912; both generators agree on 300 positions from self-play, 100 of them with push-offs; alpha-beta returns the plain-minimax value under all 8 combinations of TT / ordering / killers.
- **Search cost** (10 mid-game positions, average per move): depth 3 ≈ 8,200 nodes / 0.09 s; without move ordering ≈ 13,800 nodes. Depth 4 ≈ 30,700 nodes / 0.5 s; without the TT ≈ 37,300. Superseded by `experiments/search-cost` (386 positions), the numbers the deck shows: depth 3 ≈ 7,800 nodes, ≈ 12,800 without ordering.
- **TDLeaf(λ)**, depth 2, λ = 0.7, α = 10, marble-difference reward, 1,200 self-play games in 6.4 minutes (games recorded in `experiments/tdleaf-d2-marbles/games.jsonl`). Weights went from Assignment 1's U to: marbles 10 → 22.1, centre 20 → 17.0, cohesion 10 → 10.8, edge −15 → −6.5, exposed 0 → −3.9, wedge 0 → +2.4. Self-play games ending in six off rose from 1 in 8 to about 4 in 8.
- **Learned vs hand-set weights**, 25 random openings × both colours, both start positions:
  - Depth 2: 47 wins, 2 draws, 1 loss; score 0.96 (95% CI 0.91–1.00).
  - Depth 3: 44 wins, 5 draws, 1 loss; score 0.93 (95% CI 0.87–0.99).
  - Most wins are 200-ply adjudications by marble margin (30 of 50 at depth 2, 41 of 50 at depth 3); the mean margin is about +3 marbles.

## Evidence runs (2026-10-27, `sh experiments/run-evidence.sh`)

Every match: 25 random 2-ply openings × both colours × both start positions = 50 games; 200-ply cap scored by marble difference; third repetition = draw. Score = (wins + ½ draws) / games, with a 95% confidence interval. Raw games and summaries are in `experiments/e*/`.

**E1 · progress since Assignment 3.** The final agent (alpha-beta, learned weights, 0.5 s per move) against the Assignment 3 agent as presented there (greedy on U, and AND–OR whenever a push-off can be forced within two of its moves: `greedy-andor`): **48 wins, 2 draws, 0 losses**, score 0.98 (0.95–1.00). 45 of the wins were by pushing six marbles off, the 2 draws by repetition; mean margin +4.5 marbles, games averaged 75 plies. The steps in between (2026-11-03, Windows PC, 24 processes; the other runs: Apple M3, 8 processes):
- Against plain greedy U without AND–OR: 50–0–0 (43 by six off).
- Alpha-beta with the hand-set U against the A3 agent: 49–1–0, score 0.99. Search does almost all of the work against this opponent; the learned weights show against a stronger one: the full agent beats this hand-set alpha-beta 0.83 at equal time (E2).
- The A3 agent against plain greedy U: 1–49–0. Two one-ply greedy agents repeat moves until the third repetition, so a forceable push-off almost never comes up and AND–OR rarely fires.

**E2 · what each part contributes,** at equal time (0.5 s per move): the full agent against itself with one part switched off.

| Switched off | Full agent W–D–L | Score (95% CI) | Six off / cap | Reading |
|---|---|---|---|---|
| learned weights (hand-set U instead) | 37–9–4 | **0.83** (0.74–0.92) | 14 / 34 | the largest contribution; mostly won on marbles at the cap |
| quiescence search | 32–6–12 | **0.70** (0.58–0.82) | 38 / 10 | clear: without it the agent misjudges positions in the middle of a push exchange |
| move ordering | 30–4–16 | **0.64** (0.51–0.77) | 37 / 13 | just significant: ordering buys depth in the same time |
| transposition table | 26–6–18 | 0.58 (0.45–0.71) | 33 / 17 | not significant at 50 games |
| killer moves | 24–5–21 | 0.53 (0.40–0.66) | 36 / 12 | no measurable effect on top of history-based ordering |

**E4 · learning curve.** TDLeaf checkpoints against the hand-set weights at depth 2. Most of the gain comes early: after 120 self-play games (15 iterations) the score is already 0.93; later checkpoints score between 0.88 and 0.96, overlapping confidence intervals. The weights keep moving after that (see `experiments/tdleaf-d2-marbles/log.jsonl`), but the match score does not show further improvement.

| Self-play games | 120 | 240 | 360 | 480 | 600 | 720 | 840 | 960 | 1,080 | 1,200 |
|---|---|---|---|---|---|---|---|---|---|---|
| Score vs hand-set | 0.93 | 0.88 | 0.90 | 0.91 | 0.94 | 0.93 | 0.94 | 0.95 | 0.96 | 0.96 |

**E5 · reward design.** The same training with the result-only reward (any draw = 0) also beats the hand-set weights, 41–7–2 (score 0.89, CI 0.82–0.96). Head to head, marble-difference reward vs result-only reward is **24–1–25** (score 0.49, CI 0.35–0.63): no difference. Abalearn's warning that self-play stalls without a progress reward did not show up here, probably because the search already avoids repetitions and the random openings vary the games. Both runs learned the same direction: more weight on marbles, less on the edge penalty, a negative weight on exposed marbles and a positive one on wedges (result-only final weights: marbles 14.0, centre 10.1, cohesion 7.8, edge −5.5, exposed −2.9, wedge 2.0).

**E6 · does the training converge?** The weight plot of the 1,200-game run still slopes at the end for exposed and edge, so we ran the same training for 400 iterations (3,200 games) with a step size that decays from 10 to 3.3 (`--alpha-decay 200`), three times with different seeds (`experiments/tdleaf-d2-marbles-400*`), and kept the 1,200-game run as the final agent:
- **Strength levels off by ~1,300 games.** Checkpoints every 40 iterations score 0.87, 0.93, 0.93, 0.95, 0.97, 0.96, 0.93, 0.89, 0.93, 0.92 against the hand-set weights (`experiments/e4-learning-curve-400/`). The 3,200-game weights against the 1,200-game ones: 20–7–23 (score 0.47, CI 0.34–0.60) at depth 2 and 24–5–21 (0.53, CI 0.40–0.66) at 0.5 s per move.
- **Marbles, centre and wedge settle**; the three seeds end within 1.2 of each other on every weight (marbles 22.6–23.9) and most much closer (exposed −6.1 to −6.3).
- **Edge and exposed overlap.** Over 302,000 self-play positions they are correlated 0.71. The covariance of the six features has its smallest eigenvalue (0.012, against 1.63 for the largest) along 0.79·exposed − 0.59·edge, so the games barely tell those two weights apart. Along that direction every seed moved the same way, from about +2 at iteration 150 to about −2 at 400, while the well-determined combination stayed put. The drift is slow, systematic convergence along a direction that hardly changes the evaluation, not noise, and it does not change how the agent plays (`experiments/convergence/summary.json`).
- The result-only reward shows the same pattern over 400 iterations (`experiments/tdleaf-d2-result-400/`): edge −5.5 → −3.9 and exposed −2.9 → −4.2, with the other weights within 1.1 of the 150-iteration run.

**Caveats.**
- Many wins in the weight comparisons are adjudicated at 200 plies (34 of 50 in E2's learning ablation), not six-off wins.
- The time-limited matches (E1, E2) are sensitive to other load on the machine; the demo app was tested lightly while E1 was running.
- 50 games resolve differences of about ±0.12 in score. The TT and killer results are "no effect detected", not "no effect".

## Chapter 19: learning the weights from examples (2026-10-27)

Logistic regression of the game result on the same six features, P(win) = σ(Σ βᵢ·fᵢ), turned into evaluation weights with w = β·K/2 so they share TDLeaf's scale. The training data was generated for this purpose: 1,200 self-play games of the hand-set agent at depth 2 with 4 random opening plies (8.3 minutes on 8 cores; 315 ended six off, 286 by repetition, 599 at the cap). Only quiet positions (no legal push-off for either side) are used, and train and test are split by game: 54,153 training positions from 960 games, 13,006 test positions from 240 games. The fit takes under a second (Newton's method). `experiments/ch19-logistic/report.json` has every number below.

**Fitted weights:** marbles 16.3, centre 13.7, cohesion −0.7, edge +2.8, exposed −5.6, wedge −0.7. Like TDLeaf it raises marbles and puts a clear negative weight on exposed marbles; unlike TDLeaf it gives cohesion almost nothing and the edge a small positive weight.

**How well each weight set predicts the result on the 240 test games** (each fixed set gets one fitted scale, so only the direction of its weights is judged):

| Weights | Log-loss (coin flip = 0.693) | Accuracy on decided games |
|---|---|---|
| logistic regression (Ch. 19) | **0.511** | **85.5%** |
| TDLeaf, marble reward (Ch. 23) | 0.523 | 83.1% |
| TDLeaf, result reward | 0.524 | 82.8% |
| hand-set U | 0.574 | 79.1% |

**How well each plays** (depth 2, 50 games):

| Match | W–D–L | Score (95% CI) |
|---|---|---|
| logistic vs hand-set U | 38–12–0 | 0.88 (0.82–0.94) |
| logistic vs TDLeaf | 9–10–31 | **0.28** (0.17–0.39) |

**Reading.** Supervised learning beats the hand-set weights, and it predicts outcomes better than any other weight set. But it loses clearly to TDLeaf over the board. Logistic regression learns which positions *the hand-set agent* went on to win; TDLeaf learns from the positions its own search reaches and keeps adjusting as its play changes. Predicting results from someone else's games is not the same as choosing good moves, which is the case for reinforcement learning on this game.

A first fit had "exposed" at exactly 0: the quick push-off prefilter used to pick quiet positions also flagged every edge marble touching an opponent, so all positions where the feature is non-zero were dropped. The fit above uses the exact check, and a test now guards it.

## Chapter 22: a value network on the raw board (2026-10-27)

The linear U sees the board through six features we chose; this network sees the cells themselves. Input: 122 units (61 cells × side to move / opponent, at most 28 on), one hidden ReLU layer, one sigmoid output: the probability that the side to move wins. Pure Python: the first layer only adds the rows of the marbles present, so one evaluation costs ≈ 30 µs against ≈ 16 µs for the linear U.

**Training.** The same quiet positions and the same split by game as Chapter 19, with 10% of the training games held back to stop training early. Each position is shown under a random one of the board's 12 symmetries (tests check that all 12 keep move generation identical). A small grid (hidden 16 / 32 / 64 × two learning-rate schedules), chosen on validation loss only, put all six settings within 0.008 of each other; every run peaked at epoch 2 and then began to memorise the ~860 training games. Chosen: 16 hidden units, 1,985 parameters, 14 s of training. Files: `experiments/ch22-valuenet/`.

**Prediction on the 240 test games:**

| Model | Log-loss | Accuracy on decided games |
|---|---|---|
| logistic regression on 6 features (Ch. 19) | **0.511** | **85.5%** |
| value network on the raw board (Ch. 22) | 0.523 | 83.3% |
| TDLeaf weights (Ch. 23) | 0.523 | 83.1% |
| hand-set U | 0.574 | 79.1% |

**Play** (50 games each):

| Match | W–D–L | Score (95% CI) |
|---|---|---|
| network vs hand-set U, depth 2 | 18–11–21 | 0.47 (0.35–0.59) |
| network vs TDLeaf, depth 2 | 0–3–47 | **0.03** (0.00–0.06) |
| network vs TDLeaf, 0.5 s per move | 1–0–49 | **0.02** (0.00–0.06) |

**Reading.** With about 1,000 games of data, learning features from the raw board did not beat six designed features: the network predicts as well as TDLeaf's weights but plays only as well as the hand-set U, and loses almost every game to TDLeaf, at equal depth and at equal time alike. Its slower evaluation is not the reason (the equal-depth result is just as one-sided). A first guess was that it learned from the wrong games (the hand-set agent's); the test below rules that out as the main cause. More likely ~1,000 games are far too few to learn Abalone from raw cells with ~2,000 parameters: deep-RL Abalone work (Claußen 2022; newmbewb) trains over many generations of the network's own games and far more positions, the long-term plan in `docs/rl-literature-and-plan.md`. **Decision: the value network is built, measured and kept in the demo as a comparison, not integrated into the final agent.**

Note: two quick sanity matches (network vs the A3 agent and vs hand-set U, 20 games each, not saved) ran alongside the first equal-time attempt; that attempt was discarded and rerun on an idle machine. The match against hand-set U above was then rerun in full and saved.

## Testing the explanation: supervised learning on the learner's own games (2026-10-27)

Chapter 19 and 22 learned from the hand-set agent's games. Is that why they lose to TDLeaf? TDLeaf's run now records its games: re-running it with recording reproduced the committed run exactly (all 150 iterations: the same weights, updates and results), so `experiments/tdleaf-d2-marbles/games.jsonl` holds the very 1,200 games the final weights learned from. Both supervised models were refitted on them with unchanged settings (same split rule, same network size and schedule).

These games are harder to predict: TDLeaf plays an equal opponent, half of them end in a repetition draw, and every weight set scores ≈ 0.65 log-loss on their test games (coin flip 0.693; logistic 0.655, network 0.653, TDLeaf 0.655, hand-set 0.658). The fitted logistic weights: marbles 9.2, centre 11.8, cohesion 3.2, edge 1.3, exposed −3.4, wedge −0.9.

| Match (depth 2, 50 games) | Learned from the hand-set agent's games | Learned from TDLeaf's own games |
|---|---|---|
| logistic regression vs TDLeaf | 0.28 (0.17–0.39) | **0.60** (0.48–0.72) |
| logistic regression vs hand-set U | 0.88 (0.82–0.94) | 0.87 (0.79–0.95) |
| value network vs TDLeaf | 0.03 (0.00–0.06) | 0.07 (0.01–0.13) |
| value network vs hand-set U | 0.47 (0.35–0.59) | 0.54 (0.43–0.65) |

**Reading.**
- **Logistic regression:** the explanation holds. Fitted on the learner's own games it goes from losing clearly to TDLeaf to level with it (the interval includes 0.5). Which games a model learns from mattered more than the learning method.
- **Value network:** the explanation does not hold. On the right games it still loses almost every game to TDLeaf, so its weakness is the representation and the amount of data, not the source of the games.
- **What RL added, precisely:** TDLeaf's advantage over supervised learning here is that it generates its own training games as it learns. Given those games, plain logistic regression on the same six features plays about as well.

Files: `experiments/refit-logistic-on-tdleaf-games/`, `experiments/refit-net-on-tdleaf-games/`, `experiments/refit-*-vs-*/`.

## Chapter 4 in the final agent: forced push-offs (2026-10-27)

Assignment 3's AND–OR planner now runs inside the demo for any position and either side ([andor.py](andor.py)). Tests check that it still gives Assignment 3's answer on its test board (exactly one of 73 first moves, E5 E6 E7 → E, forces two marbles off within two moves) and that a second, rulebook-literal implementation agrees on game positions.

**AND–OR vs alpha-beta.** In 60 of the Chapter 19 test games, 529 positions had a push-off that the side to move could force within two moves but not at once. The final agent's alpha-beta (TDLeaf weights, depth 3) chose one of the forcing first moves in 286 of them (54%). AND–OR only asks whether a capture can be *guaranteed*; alpha-beta weighs that capture against everything else in its evaluation and often prefers another move. The panel shows this difference live during a game. Reproduce with `python experiments/ch4-andor-vs-alphabeta/run.py` (2 minutes; every position is listed in `positions.json`).

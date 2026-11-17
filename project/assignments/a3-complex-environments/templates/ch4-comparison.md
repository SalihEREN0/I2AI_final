# Chapter 4 methods and our measurements on Abalone

**Abalone Through Chapter 4** · Introduction to AI, Assignment 3 · Team Eight Bits, Pair 24

**What this document is for:** it puts the methods of Chapter 4 (Search in Complex Environments) side by side: the ones we showed on the slides, the others in the textbook, and recent literature. Each row answers three questions: **what it does, what we measured in our game, and why we chose it or not.** Every number comes from [abalone_ch4.py](abalone_ch4.py) (`results/a3data.json`); method definitions follow the course slides (weeks 06–07) and AIMA 4th edition, Chapter 4.

## 0. Symbols and terms

| Symbol / term | Meaning |
|---|---|
| s, s′ | state (board + side to move), next state |
| a | action (a move: 1–3 marbles + a direction) |
| RESULT(s, a) | deterministic outcome: a single board (Assignment 2) |
| RESULTS(s, a) | nondeterministic outcome: a **set** of possible boards (4.3). For us: our move + each White reply |
| OR node | we choose: **one** move is enough |
| AND node | we don't choose: a sub-plan is needed for **every** outcome |
| PERCEPT(s) | what the agent perceives in s. For us PERCEPT(s) = s (full observability) |
| belief state | the set of states the agent thinks it might be in (4.4) |
| hF | Assignment 2's heuristic: estimated moves left to push one white marble off. **Lower is better**, 0 = goal |
| U | Assignment 1's utility: U = 10·pushed-off + 2·centre + 1·cohesion − 3·edge. **Higher is better** |
| ΔE, T | improvement and temperature in annealing: a worse move is accepted with probability e^(ΔE/T); T = 2 · 0.995^t |
| H(s) | the cost estimate LRTA* learns; starts as hF |
| d | length of the shortest solution A* finds (moves) |
| "solved" | the method, by its own rules, pushed one white marble off |
| "boards evaluated" | number of boards whose score (hF or U) the method computed |

## 1. The map of Chapter 4: where does Abalone sit?

| Section | Assumption it relaxes | In Abalone | What we did in our agent |
|---|---|---|---|
| 4.1 Local search | "the path matters" | the path matters: every extra move gives White a turn | measured seven methods; **not chosen** for moves |
| 4.2 Continuous spaces | discrete states | the board is discrete; only U's weight vector is continuous | **proposed** for tuning U's weights |
| 4.3 Nondeterministic actions | deterministic outcome | the outcome is deterministic but **White chooses**: strategic | White = AND node; we built **AND–OR** |
| 4.4 Partial observation | full observation | **fully observable** | no belief state needed |
| 4.5 Online search | known RESULT | the rules are known; a capture can't be undone | no exploration: replan from the real board every turn |

## 2. The purple words on slide 4/28 and our measurements

The course slide explains with three words why hill climbing gets stuck, and with three more how to escape. We defined each one by **measuring** it in our own game.

| Term | Course definition | How we measured it | Result |
|---|---|---|---|
| **Local maximum** | a peak higher than its surroundings; the climber is drawn to it and then has nowhere to go | climbing U, every neighbour strictly lower | **{{ls.climbU.stops.local-max}}** of 200 climbs end on a strict local maximum |
| **Ridge** | a sequence of local maxima; single moves go down, their combinations go up (AIMA Figure 4.4) | in a stuck hF run, a two-move exit exists **only starting with a worse move** | **{{ls.stuck.escapeWorseFirstOnly}}** |
| **Plateau** | a flat region: a **flat local maximum** (no exit) or a **shoulder** (progress possible) | best neighbour is equal; we walked up to 500 boards over equal moves looking for a better one | **{{ls.stuck.plateau}}** of the {{ls.stuck.n}} stuck hF runs are on a plateau: {{ls.stuck.shoulder}} shoulders, {{ls.stuck.flatMin}} flat, {{ls.stuck.unknown}} undecided |
| **Stochastic hill climbing** | random among uphill moves | uniformly among better neighbours | {{ls.algos.stochastic.solved}} / 200 |
| **First-choice hill climbing** | generate successors randomly, take the first better one | shuffle successors, take the first better one | {{ls.algos.firstChoice.solved}} / 200, {{ls.algos.firstChoice.meanEvals}} boards on average (steepest: {{ls.algos.steepest.meanEvals}}) |
| **Random restart** | restart from random initial states | from the same board, with a new seed, up to 20 times | {{ls.restart.solved}} / 200 ({{ls.restart.meanTries}} tries on average when solved) |

**Open observations:**

1. **Our plateaus are mostly shoulders, not ridges.** At the opening, steepest descent drops hF 3 → 2 in one move, then 52 of 55 neighbours are equal: a plateau. 5 of 3,231 move pairs go down and **all start sideways** (hF 2 → 2 → 1). That is a plateau with an exit, i.e. a shoulder. A ridge would make the first move worse.
2. **Stochastic and first-choice don't help on a plateau.** Both choose "among uphill moves"; on a plateau there are none. The result is almost the same as steepest ascent ({{ls.algos.steepest.solved}} / {{ls.algos.stochastic.solved}} / {{ls.algos.firstChoice.solved}}). First-choice only gains in cost: it evaluates fewer boards.
3. **Random restart doesn't fit our problem.** AIMA's definition generates random **initial states**; our initial state is the real board in the game. Starting from another board would mean solving another problem. Our honest equivalent (same board, new seed) gets {{ls.restart.solved}} / 200.
4. **U's peak does not serve the goal.** Climbing U from the opening goes 26 → 100 in 17 moves; all of the gain comes from centre (+52), edge (+21) and cohesion (+1), and **0** from pushing marbles off. The top is flat (2 equal neighbours) and has an exit: after one sideways move there is a 101, so this is also a shoulder. Across 200 puzzles, **{{ls.climbU.never}}** climbs never push a marble off.

## 3. Local search methods: measurements

Setup: 200 puzzles, seed {{ls.seed}}; each puzzle is built from the opening with 12–28 random plies (both sides, no marble pushed off), Black to move, White frozen, goal: one white marble. We kept those whose shortest A* solution is 2–4 moves (200 out of {{ls.tries}} draws; d = 2: {{ls.dHist.2}}, d = 3: {{ls.dHist.3}}, d = 4: {{ls.dHist.4}}). Objective function hF (lower is better). Each method ran once per puzzle with its own seed.

| Method | Rule (course) | Solved / 200 | Mean moves (when solved) | Shortest found | Boards evaluated / run | Why it stopped |
|---|---|---|---|---|---|---|
| Steepest ascent (Figure 4.2) | best neighbour; stop if none is better | {{ls.algos.steepest.solved}} | {{ls.algos.steepest.meanMoves}} | {{ls.algos.steepest.optimal}} | {{ls.algos.steepest.meanEvals}} | {{ls.algos.steepest.stops}} |
| Stochastic | random among better ones | {{ls.algos.stochastic.solved}} | {{ls.algos.stochastic.meanMoves}} | {{ls.algos.stochastic.optimal}} | {{ls.algos.stochastic.meanEvals}} | {{ls.algos.stochastic.stops}} |
| First choice | shuffle, take the first better one | {{ls.algos.firstChoice.solved}} | {{ls.algos.firstChoice.meanMoves}} | {{ls.algos.firstChoice.optimal}} | {{ls.algos.firstChoice.meanEvals}} | {{ls.algos.firstChoice.stops}} |
| Restart ×20 | stochastic, same board, new seed | {{ls.restart.solved}} | — | — | — | — |
| Sideways moves | allow equal (≤ 100 in a row), ≤ 1,000 moves | {{ls.algos.sideways.solved}} | {{ls.algos.sideways.meanMoves}} | {{ls.algos.sideways.optimal}} | {{ls.algos.sideways.meanEvals}} | {{ls.algos.sideways.stops}} |
| Simulated annealing (Figure 4.5) | random successor; accept worse with e^(ΔE/T), T = 2 · 0.995^t, ≤ 1,000 steps | {{ls.algos.annealing.solved}} | {{ls.algos.annealing.meanMoves}} | {{ls.algos.annealing.optimal}} | {{ls.algos.annealing.meanEvals}} | {{ls.algos.annealing.stops}} |
| LRTA* (Figure 4.24) | H(s) ← min (1 + H(s′)); go to the cheapest; ≤ 1,000 moves | {{ls.algos.lrta.solved}} | {{ls.algos.lrta.meanMoves}} | {{ls.algos.lrta.optimal}} | {{ls.algos.lrta.meanEvals}} | {{ls.algos.lrta.stops}} |
| **A\* + hF** (Assignment 2) | f = g + hF | **200** | **{{ls.astar.meanMoves}}** | **200** | {{ls.astar.meanGen}} (generated) | goal |

**Moves on the same puzzles.** Each method solves a different set, so their own means average over different sets. On the {{ls.commonSolved.n}} puzzles all four (A*, sideways, LRTA*, annealing) solve, the mean number of moves is:

| A\* | Sideways | LRTA* | Annealing |
|---|---|---|---|
| {{ls.commonSolved.meanMoves.astar}} | {{ls.commonSolved.meanMoves.sideways}} | {{ls.commonSolved.meanMoves.lrta}} | {{ls.commonSolved.meanMoves.annealing}} |

**Reading:**

- **Methods that accept only better boards** (steepest, stochastic, first choice) get stuck on plateaus: all they solve are easy puzzles that never touch a plateau.
- **Methods that accept equal or worse boards** (sideways, annealing) escape. But in a game every move gives White a turn: sideways spends ~{{ratio.sideways}}× and annealing ~{{ratio.annealing}}× as many moves as A*.
- **LRTA*** only uses what it learned when it returns to the same board. Its mean number of revisits per run is {{ls.algos.lrta.meanRevisits}}: our plateaus are so wide that the 1,000-move cap fills up before learning helps.

## 4. Other local search methods in the textbook

| Method | What it does | In our game | Decision |
|---|---|---|---|
| Local beam search | keeps k states together, picks the k best successors | a game has **one** real board; a population of k boards is not ours to choose | possible for U's weights |
| Stochastic beam | picks k successors with probability proportional to score | same problem | same |
| Genetic algorithm | population + crossover + mutation | "crossing" two boards doesn't give a legal board | meaningful for U's weight vectors (crossover is natural on a vector) |
| Empirical gradient / Newton (4.2) | measures the slope in a continuous space and moves along it | the board is discrete; U's weights are continuous | **proposed**: tuning U |
| Online DFS (4.5) | exploration that can backtrack in an unknown environment | backtracking is not guaranteed (captures) | unnecessary: RESULT is known |

## 5. Advanced methods in the literature

### 5.1 Escaping plateaus (discrete local search)

| Method | Idea | Why it does or doesn't fit our plateaus |
|---|---|---|
| Tabu search (Glover 1989) | forbid recently visited states; go to the best allowed neighbour (even if worse) | prevents sideways walks from cycling on shoulders; the move-count problem remains |
| Late acceptance hill climbing (Burke and Bykov 2017) | compare the new state with the score L steps ago | one parameter, simpler than annealing; same "too many moves" cost |
| Iterated local search (Lourenço et al. 2003) | "kick" the local optimum, climb again | in a real game a kick = a wasted move |
| WalkSAT-style noise (Selman et al. 1994) | random step with probability p, greedy with 1 − p | a random walk on the plateau; cost similar to annealing |

**Common conclusion:** all of these improve the **final state**. Because the path matters in a game, they suit tuning the objective function (U), not choosing moves.

### 5.2 Tuning U's weights (continuous local search, 4.2)

U = 10·pushed-off + 2·centre + 1·cohesion − 3·edge. The weight vector (10, 2, 1, −3) is a point in a continuous space; here **only the point reached matters**, not the path. That is exactly where local search belongs.

| Method | Idea | Who uses it |
|---|---|---|
| Texel tuning (Österlund 2014) | tune the weights by local search on how well the evaluation predicts the game result (logistic loss) | chess engines (Chess Programming Wiki) |
| SPSA (Spall 1992) | perturb all weights at once by random ±, estimate the gradient from two measurements | Stockfish's parameter tuning |
| CMA-ES (Hansen and Ostermeier 2001) | a population of weight vectors; learn the covariance of the distribution | noisy, derivative-free problems |
| Random restart | climb from different starting vectors | meaningful here: every weight vector is a valid start |

**The criterion problem:** what do we tune U against? Our proposal: in self-play games or A*'s sub-goal puzzles, "how often is U's chosen move the same as the move AND–OR guarantees". The trap on slide 3 (U ranks three quiet moves above the capture) is a direct tuning signal.

## 6. Nondeterministic outcomes: AND–OR and alternatives

Test board: we have 73 moves, only the capture pushes a marble off; then White has {{ex.test.afterCapture.replies}} replies. Goal: 2 white marbles within at most 2 of our moves.

| Plan type | How it sees White | On the test board | Cost | Decision |
|---|---|---|---|---|
| Fixed plan (Assignment 2's A*) | frozen | works for {{ex.test.afterCapture.keep}} of {{ex.test.afterCapture.replies}} replies | cheapest | fragile |
| Greedy U (one move) | not at all | favourite {{ex.test.greedy.best}}; then {{ex.test.greedy.refuting}} of {{ex.test.greedy.replies}} replies end the guarantee | 73 boards | rejected |
| **AND–OR (Figure 4.11)** | any reply may happen (AND) | **39 / 39**, marbles lost {{ex.test.andor.lost}}; of 73 first moves only the capture | {{ex.test.andor.search.boards}} boards, {{ex.test.andor.search.ms}} ms | **chosen** |
| Minimax + α–β (Chapter 6) | the opponent chooses by its own goal | — | deeper tree; ~b^(d/2) with α–β | next |
| Expectimax | White is random | — | — | doesn't fit: White doesn't roll dice |
| Proof-number search, df-pn (Allis 1994; Nagai 2002) | answers "is it a forced win?" on an AND/OR tree with the least proof | — | far more efficient than AND–OR on deep forced lines | candidate for "force a push-off" questions deeper than 2 |

**Independent check:** the AND–OR result was recomputed with Assignment 2's second generator, which follows the rulebook literally, by a separate routine with no plan object that only asks "is it forced?": the same single first move ({{ex.test.recheck.firstMovesForcing}}).

**Counterfactual:** had D6 stayed on its opening cell (B6), the reply E9 → SW would have no answer and no first move would guarantee the goal: a single supporting marble decides the guarantee.

## 7. Observation and exploration

| Question | Abalone | If it weren't so | Cost |
|---|---|---|---|
| Fully observable? (4.4) | yes: PERCEPT(s) = s | if we only saw next to our marbles | C(30, 13) = {{ex.test.belief.0.beliefSize}} possible boards; seeing two cells out, {{ex.test.belief.1.formula}} = {{ex.test.belief.1.beliefSize}} |
| Sensorless (conformant) plan | not needed | if we saw nothing | every move must work on every possible board |
| Reversibility (4.5) | {{ex.test.moves.quietReversible}} of 73 moves are undone by the reverse move; a capture never | — | a capture is a one-way door |
| Repetition | {{ex.test.backToOpening}} paths back to the opening after 4 plies | — | search must remember what it has seen |
| Safely explorable? | no: losing 6 marbles is a dead end | — | learning by trial is risky |
| Is online search needed? | no: RESULT is known | if the rules were unknown | LRTA*, online DFS; but one-way doors |

## 8. Discussion: what would unlimited compute change, and why did we choose these?

**With unlimited compute:**

- We could run **AND–OR to the end of the game**: "is it a forced win?" would be answered exactly (solving the game). Then even Chapter 6's minimax would be unnecessary; the AND–OR tree would be a complete strategy.
- **Belief states** would be no problem even under partial observation.
- **Local search** could be used for move choice too; but compute doesn't fix the "extra moves" problem. Sideways moves and annealing would still spend many times more **game moves** than A*, and each one gives White a turn. That is a cost of the **model**, not of compute.

**What we actually chose and why:**

| Decision | Why | What we rejected, and why |
|---|---|---|
| **A\* + hF** for move sequences (sub-goals while White is frozen) | guarantees the shortest path; in a game the path matters | local search: stuck on plateaus or spends too many moves |
| **AND–OR** when the opponent replies (shallow, 2 moves) | handles every reply; 39/39 on the test board | fixed plan (37/39), U's greedy choice (guarantee ends on 7/43 replies) |
| **No belief state** | full observability: one possible board | — |
| **No exploration**, replan every turn | RESULT is known; captures are irreversible | online search, LRTA* |
| Local search **only for U's weights** (proposed) | there only the point reached matters; the vector is continuous | using it to choose moves |
| Next: **minimax + α–β** (Chapter 6) | the AND node treats White as "anything may happen"; minimax accounts for White's goal | expectimax: White doesn't roll dice |

## References

1. S. Russell, P. Norvig. *Artificial Intelligence: A Modern Approach*, 4th ed. (Global), Chapter 4: Figures 4.1–4.5 (local search, annealing), 4.10–4.11 (AND–OR), 4.24 (LRTA*).
2. S. Okyay. I2AI weeks 06–07: Search in Complex Environments (slides 1–28; purple words on slide 4/28). Week 03: Intelligent Agents (slide 10/25, "strategic").
3. Pair 24. Assignment 1 (U, task environment) and Assignment 2 (engine, hF, A*, puzzle model) decks; [abalone_search.py](../a2-search/abalone_search.py).
4. S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi. Optimization by simulated annealing. *Science* 220, 1983.
5. R. E. Korf. Real-time heuristic search. *Artificial Intelligence* 42, 1990.
6. F. Glover. Tabu search, part I. *ORSA Journal on Computing* 1(3), 1989.
7. E. K. Burke, Y. Bykov. The late acceptance hill-climbing heuristic. *European Journal of Operational Research* 258(1), 2017.
8. H. R. Lourenço, O. C. Martin, T. Stützle. Iterated local search. *Handbook of Metaheuristics*, 2003.
9. B. Selman, H. Kautz, B. Cohen. Noise strategies for improving local search. *AAAI* 1994.
10. J. C. Spall. Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. *IEEE Transactions on Automatic Control* 37(3), 1992.
11. N. Hansen, A. Ostermeier. Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation* 9(2), 2001.
12. P. Österlund. Texel's tuning method. Chess Programming Wiki, 2014.
13. L. V. Allis. *Searching for Solutions in Games and Artificial Intelligence* (proof-number search). PhD thesis, Maastricht, 1994.
14. A. Nagai. *Df-pn Algorithm for Searching AND/OR Trees and Its Applications*. PhD thesis, University of Tokyo, 2002.

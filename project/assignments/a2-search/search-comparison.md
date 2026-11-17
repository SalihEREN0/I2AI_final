# Search algorithms and heuristics: the ones on our slides and current methods

**Abalone as a Search Problem** · Introduction to AI, Assignment 2 · Team Eight Bits, Pair 24

**What this document is for:**
- It compares the search algorithms and heuristics on our slides with the advanced methods in Chapter 3 of the course and with recent literature, specifically for Abalone.
- Each row follows the pattern *what it does → why it works or doesn't → what it gives in Abalone*.
- Symbols such as g, h, f, b, b*, d are explained first, in §0, each with an example from our game.
- The closing discussion (§7) asks how the choice would change with different resources (unlimited compute, unlimited memory, tight memory, many cores) and why we chose what we did.
- Every number comes from our own engine ([abalone_search.py](abalone_search.py)) or from the cited source.
- IDA* and weighted A* were measured separately for this comparison: `python abalone_search.py more`, on the same 60 puzzles.

**Scope note:** game-tree search (alpha-beta, MCTS) belongs to Chapter 6; here it only appears as the "next step", with the deck's own Abalone sources.

---

## 0. Symbols and terms (read this first)

The tables use the symbols of AIMA (the textbook, Chapter 3). Each comes with an example from our game.

### State, move, cost

| Symbol | Meaning | Example in Abalone |
|---|---|---|
| **s**, **s′** | state and next state: the whole board + whose turn it is | s′ = RESULT(s, a): the board after a push |
| **a** | action, i.e. a move | "E5 E6 E7 → E" |
| **c(n, a, n′)** | cost of going from n to n′ with move a (ACTION-COST) | 1 per move |
| **ε** | smallest action cost | 1 |
| **C\*** | total cost of the cheapest solution | since every move costs 1, C* = d |
| **ply** | half-move: a single move by one player | in the slide 5 tree both sides alternate; in the puzzles only Black moves, so ply = move |

### Search tree

| Symbol | Meaning | Example in Abalone |
|---|---|---|
| **n** | node: a record in the tree (state + parent + move + path cost). The same board can appear in several nodes | 3.7 nodes per board on average at ply 4 |
| **b** | branching factor: number of moves in a state | 44 at the start, ≈ 60 on average |
| **d** | depth of the shortest solution, i.e. number of moves in the shortest plan. In the memory formula b·d: the depth the search reaches | slide 6 puzzle d = 3; "d = 3 · 37" in the tables = 37 puzzles of 3 moves |
| **m** | maximum depth of the tree; may be infinite | unbounded in Abalone because moves can be undone; DFS time is O(bᵐ) |
| **l** | depth limit (in DLS, and in each IDS iteration) | IDS iterations l = 1, 2, 3, … |
| **N** | number of nodes generated ("nodes generated" in the tables) | slide 6 puzzle: BFS 89,909, A* + hF 439 |
| **b\*** | effective branching factor: the branching of a uniform tree that generates N nodes at depth d; N + 1 = 1 + b* + (b*)² + … + (b*)ᵈ. **Smaller is better** | at d = 3, BFS 42.6, A* + hF 13.4 |
| **perft(d)** | number of all move sequences of d plies from the start (used to validate the engine) | 44 · 1,936 · 98,912 · 5,045,110 |
| **O(·)** | order of growth: O(bᵈ) means "grows in proportion to b to the d" | BFS time and memory are O(bᵈ) |

### How a search works inside

| Term | Meaning | Example in Abalone |
|---|---|---|
| **generated** | a node created as a child | the numbers in the tables |
| **expanded** | a node whose children have been generated | A* + hF generates 73 children for every node it expands and never expands 98.6% of them |
| **frontier** | nodes waiting to be expanded: a FIFO queue in BFS, a LIFO stack in DFS, a priority queue in A* | the yellow dots on slide 5 |
| **reached table** | table of boards already seen (graph search), so the same board isn't expanded again | essential because boards repeat |
| **TT (transposition table)** | the same idea, used in IDA* and alpha-beta | IDA* + TT: 17% fewer nodes at d = 4 |
| **memory** (in the tables) | number of boards held at the same time; × 16 B = bytes | at d = 4, A* holds 6,554 boards, IDA* 5 |

### Evaluation and heuristics

| Symbol | Meaning | Example in Abalone |
|---|---|---|
| **g(n)** | cost of the path **travelled** from the start to n | after 2 moves, g = 2 |
| **h(n)** | estimate of the cost **remaining** from n to the goal (heuristic) | hF = 3 at the start of the slide 6 puzzle |
| **h\*(n)** | true remaining cost; unknown, used to measure the estimate's accuracy | h* = 3 on slide 6, so hF is exact |
| **f(n)** | evaluation function: decides which node is expanded first | A*: f = g + h · greedy: f = h · UCS: f = g |
| **w** | weight of h in weighted A*: f = g + w·h; the plan is at most w times longer | w = 2: 5 of 60 plans are 1 move longer |
| **k** | number of best nodes kept in beam search | — |
| **h0** | 0 everywhere; no heuristic (equals UCS) | — |
| **h1** | number of opponent marbles still to push off (one move pushes off at most 1) | 1 in the puzzles |
| **h2** | sum of the edge distances of the nearest opponent marbles / 2 (one move pushes at most 2 marbles) | — |
| **hR** | max(h1, h2): looks only at the opponent | 1 on slide 6 (true 3): blind |
| **hF** | also adds the distance of two of our marbles to the push point, plus the push | 3 on slide 6 (true 3) |

*Note:* the **n** in the hR formula means "number of opponent marbles to push off", not a node.

### Properties of heuristics

| Term | Definition | What it buys | For us |
|---|---|---|---|
| **admissible** | h(n) ≤ h*(n) for every n: never overestimates | A* finds the shortest plan | hR and hF: hF ≤ true cost on 60 of 60 puzzles |
| **consistent** | h(n) ≤ c(n, a, n′) + h(n′): the estimate never drops faster than the cost in one move | also implies admissibility; A* never expands a board twice | hF: 0 violations on 103,125 edges |
| **dominant** | both admissible and h₂(n) ≥ h₁(n) for every n | the dominant one expands fewer nodes | hF ≥ hR everywhere |
| **relaxed problem** | a problem with relaxed rules that is easy to solve; its solution cost gives an admissible h | a way to construct heuristics | hF: "our marbles may pass through each other" |

### Units

| Unit | Bytes | For us |
|---|---|---|
| KB, MB, GB | 10³, 10⁶, 10⁹ | DFS 9.3 KB; BFS 23 MB at ply 4; 16 GB at ply 6 |
| TB, PB, EB | 10¹², 10¹⁵, 10¹⁸ | BFS 7.6 PB at ply 10 |
| YB (yottabyte) | 10²⁴ | the table needed to solve Abalone completely ≈ 10 YB |

---

## 1. What do we compare on?

The four criteria of AIMA 3.3.4:
- **Complete?** Does it find a solution if one exists?
- **Optimal?** Does it find the shortest (cheapest) solution?
- **Time:** how many nodes does it generate?
- **Memory:** how many nodes does it hold at once?

Four properties make Abalone hard and decide which algorithm fits:

| Property | Our measurement | Consequence |
|---|---|---|
| **Large branching** | 44 moves at the start, ≈ 60 on average (Lemmens); work × 51 per ply | memory and nodes "generated but never expanded" are the real cost |
| **Boards repeat** | every move can be undone; 3.7 paths per board at ply 4 | memory (reached table, TT) is needed to avoid re-expanding boards |
| **The goal is not a single board** | every board with "6 white marbles off" is a goal | backward search is impractical |
| **There is an opponent** | 6 of White's 75 replies break the plan | Chapter 3 holds for sub-problems (White frozen); the full game is Chapters 4 and 6 |

---

## 2. Uninformed search on the slides

| Algorithm | What it does | Complete | Shortest | Cost in Abalone (measured) | Decision and reason |
|---|---|---|---|---|---|
| **BFS** | shallow to deep, layer by layer (FIFO) | ✓ | ✓ (equal costs) | keeps every board: 23 MB at ply 4, fills 16 GB at ply 6; ply 6 ≈ 9 hours | correct answer but **not enough memory** |
| **UCS** | expands the lowest g | ✓ | ✓ | with equal costs it follows BFS's order but tests the goal late: **47×** the nodes at d = 2 | **unnecessary** with equal costs |
| **DFS** | goes deepest first (LIFO) | ✗ | ✗ | memory ≈ 9.3 KB; **loops** because moves can be undone | good memory, **unreliable** |
| **DLS** | DFS + depth limit | ✗ (if limit < d) | ✗ | needs the right limit in advance | not enough on its own |
| **IDS** | repeats DLS with limits 1, 2, 3, … | ✓ | ✓ | DFS's memory, only **+2.0%** nodes over BFS; can stop at any time | **chosen**: BFS's answer + DFS's memory |
| **Bidirectional** | from the start and the goal at once | — | — | no single goal board; a backward step means "pushing a marble back" | **not applicable** |

**Why IDS?**
- The depth limit cuts cycles, so IDS is complete.
- Going from shallow to deep, it finds the shortest solution first.
- The repeated work is small: ≈ b / (b − 1). For b = 44 that is 2.3%; we measured 2.0%.
- "Anytime": in a timed game it returns the answer of the last completed depth.

---

## 3. Informed search on the slides

Measurement: the same 60 puzzles (random mid-game, White frozen, push one white marble off), mean nodes generated (b*):

| Search | f(n) | d = 2 (19) | d = 3 (37) | d = 4 (4) | Shortest plan | Decision |
|---|---|---|---|---|---|---|
| BFS | order | 2,977 (54.1) | 79,189 (42.6) | > 792,241 | 58/58 | correct, expensive |
| UCS | g | 141,071 | > 912,543 | > 1,000,028 | 28/28 | late goal test |
| A* + hR | g + hR | 2,993 | 79,207 | > 792,252 | 58/58 | hR is blind: no better than BFS |
| **A\* + hF** | g + hF | **870 (29.0)** | **2,582 (13.4)** | **13,196 (10.4)** | **60/60** | **chosen** |
| Greedy + hF | hF | 11,520 | 21,972 | 30,116 | 8/60 (mean 252 moves) | fast but long plans |

(> : some runs were cut at the 1 million node limit; the true value is larger.)

**Why A* + hF and not greedy?**
- A* with f = g + h counts both the path travelled and what remains, so with an admissible h it finds the shortest plan.
- Greedy search with f = h forgets g. It wanders among boards that "look close to the goal" and produces a 320-move plan.
- **A*'s cost:** like BFS it stores every node, and it computes hF for each one.

---

## 4. Advanced and current search methods

### 4.1 What we measured (same 60 puzzles, with hF)

| Search | Idea | d = 2: nodes / memory | d = 3: nodes / memory | d = 4: nodes / memory | Shortest plan |
|---|---|---|---|---|---|
| **A\*** (baseline) | keeps all open and closed nodes | 870 / 651 | 2,582 / 1,745 | 13,196 / 6,554 | 60/60 |
| **IDA\*** (Korf 1985) | depth-first with an f = g + h bound; raises the bound step by step | 847 / 3 | 3,348 / 4 | 71,153 / 5 | 60/60 |
| **IDA\* + transposition table** (Reinefeld & Marsland 1994) | does not re-expand a board seen by a shorter path in the same iteration | 847 / 15 | 3,090 / 45 | 59,130 / 742 | 60/60 |
| **Weighted A\*, w = 2** (Pohl 1970) | f = g + 2·h: more eager for the goal, plan at most 2× longer | 870 / 651 | 2,770 / 1,867 | 11,218 / 5,869 | 55/60 (worst +1 move) |

*Memory: number of boards held at once. For A* the reached table; for IDA* the path (and the TT).*

**Reading:**
- **IDA\***
  - *Gain:* **≈ 400×** less memory than A* at d = 3 and **≈ 1,300×** less at d = 4, still finding the shortest plan every time.
  - *Cost:* the node count grows with depth: 1.3× at d = 3, 5.4× at d = 4.
  - *Why:* IDA* remembers nothing. Because boards repeat in Abalone, it re-expands the same boards via different paths.
  - *Link to the course slide:* "IDA* works well with heuristics that take few distinct values." hF is a small integer (1–6), so the bound reaches the right value in a few steps.
- **Transposition table:** catches some of the repeats (17% fewer nodes at d = 4) while still using 9× less memory than A*. The full game uses the same idea (TT).
- **Weighted A\***
  - **No notable gain** here (7% more nodes at d = 3, 15% fewer at d = 4), because hF is already accurate: the weight barely changes the order. On top of that it lengthens 5 plans by one move.
  - *Lesson:* weighted A* helps when the heuristic is weak (the maze example on course slide 28); with a strong heuristic it adds nothing.

### 4.2 What we didn't measure: literature and fit for Abalone

| Method | Idea | What it solves | Cost | Fit for Abalone and why |
|---|---|---|---|---|
| **RBFS** (Korf 1993) | recursive best-first; remembers the f of the best alternative | A*'s order in linear memory | "changes its mind" often, re-expands | similar to IDA*; since hF is an integer IDA* already works well, no extra gain expected |
| **SMA\*** (Russell 1992) | A* until memory is full, then drops the worst leaf | softens A*'s memory wall | regenerates what it dropped | the safe alternative to A* under a memory limit |
| **Beam search** | keeps only the best k nodes on the frontier | speed and fixed memory | not complete, not optimal | for fast move choice; no plan-quality guarantee |
| **Partial-Expansion A\*** (Yoshizumi et al. 2000), **EPEA\*** (Felner et al. 2012) | when expanding a node, keeps only the children with the best f (EPEA* picks them without generating them) | A*'s memory under **large branching** | a node is expanded several times | **a very good fit for Abalone:** in our measurement A* + hF **never expands 98.6%** of the nodes it generates (73 generated per expanded node); PEA* cuts exactly this waste |
| **Breadth-first heuristic search / frontier search** (Zhou & Hansen 2006) | on unit-cost, undirected graphs drops the closed list and keeps only frontier layers | memory of BFS and A* | extra work to recover the solution path | Abalone has equal costs and reversible moves (an undirected graph): the conditions hold |
| **ARA\*** (Likhachev et al. 2003), **AWA\*** (Hansen & Zhou 2007) | first a fast plan with a large w, then improve by lowering w | "always an answer" under a time limit | repeated search | an alternative to IDS once the game is timed; with a strong hF the first plan is already good |
| **MM** (Holte et al. 2016), **NBS** (Chen et al. 2017) | bidirectional heuristic search; NBS expands at most 2× the necessary nodes | guaranteed meeting in the middle | needs heuristics in both directions | **not applicable:** no single goal board, backward step undefined |
| **HDA\*** (Kishimoto et al. 2009) | parallel A* that hashes nodes across cores | scales on many cores | communication overhead | an engineering improvement; doesn't change the algorithm itself |

### 4.3 For the full game (Chapter 6, next step)

| Method | In the Abalone literature | Why |
|---|---|---|
| **Alpha-beta + iterative deepening + TT + move ordering** | Aba-Pro (Aichholzer 2002) and AIBA (Chorus 2009) are strong this way; speed improvements in Papadopoulos et al. (2012) | accounts for the opponent's worst reply; anytime thanks to IDS; the TT catches repeating boards (our 3.7 paths/board finding) |
| **Monte-Carlo search** (random simulated games) | Chorus (2009): even with depth-limited simulations, alpha-beta won all 60 games; the Monte-Carlo player pushed off 3 marbles in total over 60 games | random games in Abalone are long and uninformative; a good evaluation function matters more |
| **MCTS** (UCT; course Chapter 6.4) | not tried in Chorus's thesis; combines Monte-Carlo with a tree | its strength in Abalone isn't measured in our sources; how it compares with alpha-beta is an open question |

**Why is our plan "IDS + A* + hF, then alpha-beta"?**
- **IDS:** little memory and anytime behaviour; it will be the skeleton of the full game.
- **A* + hF:** the fastest for sub-goals such as "push one marble off" while White is frozen.
- **Alpha-beta:** sits on top of IDS once the opponent comes back; in the literature Abalone's strongest programs use this structure.

---

## 5. Heuristics

### 5.1 On the slides

| Heuristic | What it looks at | Admissible, and why? | Consistent? | On this puzzle | Measured effect |
|---|---|---|---|---|---|
| h0 = 0 | nothing | yes (trivially) | yes | 0 | UCS: the worst |
| **hR** = max(h1, h2) | edge distance of opponent marbles; h1 = marbles to push off, h2 = ⌈nearest edge steps / 2⌉ | yes: one move pushes off at most 1 marble and pushes at most 2 opponent marbles 1 cell | yes | 1 (true 3) | no better than BFS (79,207 ≈ 79,189) |
| **hF** = max(hR, 1 + min max(our steps, its steps)) | also the distance of two of our marbles to the push point | yes: a relaxed problem (our marbles pass through each other, ≤ 1 cell per move) | yes: 0 violations on 103,125 edges | 3 (true 3) | 31× fewer nodes than BFS, b* 42.6 → 13.4 |

**Why does hF dominate?** hF contains hR (hF ≥ hR on every board). A dominant heuristic never makes A* expand more nodes, except for ties.

**Why is hR blind?**
- The real work in pushing a marble off is bringing our marbles into formation; hR looks only at the opponent and can't see that.
- It says 1 on almost every board. The only thing it does is mark goal boards with 0, which reduces A* to BFS with an early goal test.

### 5.2 Advanced and current heuristic methods

| Method | Idea | Admissible? | How it fits Abalone | Cost |
|---|---|---|---|---|
| **Relaxed problem** (AIMA 3.6.2) | relax a rule, solve the easy problem exactly | yes | **hF is an example**: our marbles may pass through each other | the cruder the relaxation, the weaker the estimate |
| **Delete relaxation: h_max** (Bonet & Geffner 2001) | the most expensive sub-goal | yes | the "max" in hF is the single-sub-goal case of this family | can stay weak |
| **h_add, h_FF** (Bonet & Geffner 2001; Hoffmann & Nebel 2001) | sum the sub-goal costs / count a relaxed plan | **no** (may overestimate) | strong guidance for pushing off several marbles (k > 1) | no shortest-plan guarantee: "good enough" search like weighted A* |
| **Pattern database (PDB)** (Culberson & Schaeffer 1998), **additive PDB** (Felner, Korf & Hanan 2004) | solve part of the problem (a pattern) exactly in advance and store it in a table | yes | sees what hF misses: computes exactly the **blocking and the outnumbering condition** of a few marbles around the push point; the 12-fold symmetry shrinks the table | table size grows exponentially with the pattern; precomputation time |
| **Landmarks: ALT** (Goldberg & Harrelson 2005) and **LM-cut** (Helmert & Domshlak 2009) | "landmarks" every solution must pass through | yes | hF already follows landmark logic: "two of our marbles must reach the push point". Summing landmarks would be strong for pushing off several marbles | finding landmarks and partitioning costs |
| **Combining heuristics: max** | take the largest | yes | what we did (max of hR and the formation bound) | only as strong as the best one |
| **Cost partitioning** (Katz & Domshlak 2008; Seipp, Keller & Helmert 2020) | split each move's cost among heuristics and add them | yes | one move both walks our marbles and pushes the opponent; splitting the cost between the two lets "our path + the opponent's path" be added, stronger than max | finding the partition takes computation |
| **Merge-and-shrink** (Helmert et al. 2014) | abstraction by merging and shrinking the state space | yes | general-purpose; doesn't use Abalone-specific knowledge | expensive for large abstractions |
| **Good-enough (inadmissible) heuristic + weighted A\*** (AIMA 3.5.4) | a more accurate h that sometimes overestimates | no | fast plans; at most W× longer (with weighted A*) | may lose the shortest plan |
| **Heuristic learned from experience** (AIMA 3.6.6) | learn from (board, true cost) pairs of solved instances | no guarantee | we have the exact d of 60 puzzles: an estimate of d could be learned from features such as hR, hF and distances | admissibility is lost; so is the shortest-plan guarantee |
| **Game evaluation function** (Chapter 6) | the winning value of a board (Aba-Pro, ABLA, AIBA features: centre, cohesion, marbles off) | different goal | Assignment 1's U is of this kind; **not an h**: it says "how good is the board", not "how many moves to the goal", and is used at the depth limit | can't be used with A*; used with alpha-beta |

**What separates heuristics?**
- **Admissibility:** gives the shortest-plan guarantee.
- **Accuracy:** lowers b*.
- **Computation cost:** paid at every node.

hF balances all three: admissible, equal to the true cost here, and computed with a small distance search. A stronger next step is a **pattern database** around the push point, because the two things hF misses (marbles blocking each other, the 3-on-2 outnumbering condition) can only be seen by small, exactly solved parts.

---

## 6. Summary: what we chose, why, and what's next

| Situation | Choice | Why | Alternative, and why not |
|---|---|---|---|
| Uninformed, skeleton of the full game | **IDS** | BFS's answer, DFS's memory, +2% work, anytime | BFS (memory), DFS (cycles), UCS (47× the work with equal costs) |
| Sub-goal with White frozen | **A\* + hF** | 60/60 shortest, 31× fewer nodes than BFS | greedy (long plans), weighted A* (no gain here) |
| Tight memory | **IDA\*** (+ TT if needed) | ≈ 1,300× less memory at d = 4 (≈ 9× with TT), still the shortest plan | cost: 5.4× the nodes at d = 4 (4.5× with TT) |
| Large branching | **EPEA\*** (next step) | A* never expands 98.6% of what it generates | — |
| Better heuristic | **pattern database**, cost partitioning | sees blocking and outnumbering; hR and formation can be added | a learned heuristic loses the guarantee |
| When the opponent comes back | **alpha-beta + IDS + TT** (Chapter 6), a plan for every reply in Chapter 4 | Abalone's strongest programs use this structure | Monte-Carlo search: lost 60 of 60 to alpha-beta in Abalone (Chorus 2009) |

---

## 7. Discussion: would the choice change with different resources?

First, two terms, because the discussion is about the difference between them:
- **Shortest plan (optimal solution):** the algorithm's guarantee. BFS, IDS, A* and IDA* give it; greedy search doesn't.
- **Most suitable choice:** the method that gives the best result with the time, memory and compute we have. It changes when the resources change.

### 7.1 What does more compute buy? The exponential wall

The tree we measured grows about **51×** per ply. So more compute buys very little depth:

| Compute increase | Depth gained (plies), BFS: log₅₁(increase) | With A* + hF: log₁₃.₄(increase) |
|---|---|---|
| 2× (a faster computer) | 0.18 | 0.27 |
| 8× (8 cores) | 0.53 | 0.80 |
| 100× (C instead of Python) | 1.17 | 1.77 |

**Conclusion:** a good heuristic is worth more than hardware.
- **Hardware:** turns an increase into depth only by its logarithm.
- **Heuristic:** shrinks the base of the exponent; b* goes from 42.6 to 13.4.
  - Rough estimate: gain at depth d ≈ (42.6 / 13.4)ᵈ = 3.18ᵈ.
  - At d = 3 that is 32×; we measured 31×.
  - Extending the same ratio to d = 10 gives ≈ 100,000× (an extrapolation, not a measurement).

**Even unlimited compute hits a physical wall.** Assume a machine generating 10¹⁸ nodes per second (today's largest supercomputers operate at this order):
- **Ply 10:** ≈ 9 × 10¹⁶ paths, 0.09 seconds; time stops being a problem. But BFS's memory is still ≈ 7.6 PB.
- **Ply 20:** ≈ 10³⁴ paths, ≈ 3 × 10⁸ years. The average game is 87 plies.
- **The whole game tree** (60⁸⁷ ≈ 5 × 10¹⁵⁴): ≈ 10¹²⁹ years, ≈ 10¹¹⁹ times the age of the universe.

### 7.2 Scenarios: which method for which resources?

| Scenario | Best result | Why | Our choice and why |
|---|---|---|---|
| **Unlimited compute and memory** (theoretical) | **Solve the game:** retrograde analysis, with every board's win/loss/draw value in a table | perfect play; no search or heuristic needed | **Physically impossible:** 4 × 10²⁵ boards × 2 bits ≈ **10 yottabytes**. Among the largest solved games, checkers has 5 × 10²⁰ boards (Schaeffer et al. 2007); Abalone is ≈ 10⁵ times larger. Abalone has only been solved on small boards (Gutstadt et al. 2023: the 2×2×3 board is a draw). |
| **Unlimited memory, limited time** | **A\* + pattern database** (precomputed table) | the table is computed once and is cheap per query; a more accurate h → fewer nodes | We chose **A\* + hF**: no precomputation, consistency proven, already equal to the true cost on the example puzzle. A pattern database is the next step. |
| **Tight memory (one 16 GB laptop), plenty of time** | **IDA\*** (+ TT if needed) | memory only as large as the path (≈ 1,300× less at d = 4), still the shortest plan | Our puzzles are small (A* holds at most a few thousand boards), so A*'s memory isn't a problem and A* generates fewer nodes (5.4× fewer at d = 4). If memory gets tight we switch to IDA*. |
| **Many cores** | **parallel A\* (HDA\*)**, parallel alpha-beta | splits the work across cores | Faster, but doesn't remove the wall: 8 cores ≈ +0.5 ply. Algorithm and heuristic first, then parallelism. |
| **Speed is everything, plan quality doesn't matter** | **greedy** or weighted A* | ignores the path and expands what looks closest to the goal; finds a fast plan with few nodes on most problems | Our measurement doesn't confirm this: greedy generated **8.5× more** nodes than A* + hF (d = 3: 21,972 vs 2,582) and made a 320-move plan; it found the shortest plan on only 8 of 60 puzzles. Weighted A* brings no gain with hF. |
| **The real game: an opponent, limited time per move** | **IDS + alpha-beta + TT**, with Assignment 1's U at the depth limit | anytime, accounts for the opponent's worst reply, remembers repeating boards | **This is our plan.** In the literature Abalone's strongest programs use this structure (Aba-Pro, AIBA). |

### 7.3 Compute doesn't fix a modelling error

Even with unlimited compute, Chapter 3's search answers the wrong question:
- **The problem:** in a tree where both sides move, a "solution" is just one path. That path also chooses White's moves in our favour, as if White were helping us.
- **With more compute:** we find deeper but equally optimistic plans. On slide 4, 6 of 75 replies broke the plan; a good opponent picks one of those 6.
- **The right order:**
  1. Fix the **model** first. Chapter 4 builds a separate plan for each opponent reply (AND-OR search); Chapter 6 chooses against the worst reply (minimax).
  2. Then improve the **algorithm and heuristic**: alpha-beta, a good evaluation function.
  3. **Hardware** comes last.

### 7.4 So why is this choice the "most suitable"?

Our constraints:
- a single laptop and Python (≈ 238,000 nodes per second);
- time on the order of seconds in the presentation and in the game;
- the assignment asks for correctness to be **demonstrated**.

| Decision | Why it is the most suitable | When we would choose something else |
|---|---|---|
| **Cost = 1 per move** | the rules count turns; this way BFS and IDS guarantee the shortest plan | if moves really had different costs (e.g. thinking time in a timed game): UCS or A* |
| **A\* + hF for sub-goals** | fewest nodes (31× fewer than BFS), shortest-plan guarantee, no precomputation, consistency both proven and measured | tight memory: IDA*; even larger branching: EPEA*; time for precomputation: pattern database |
| **IDS as the skeleton of the full game** | DFS's memory, BFS's answer, only +2% work; anytime | unlimited memory and no time limit: BFS would also do, but no better than IDS |
| **Alpha-beta for the opponent (next step)** | tried and strongest in Abalone; random-simulation Monte-Carlo lost 60 of 60 to it | unlimited compute and memory: solve the game outright, but that is physically impossible |

**In one sentence:** with unlimited resources, solving the game would be best. But Abalone's size makes that physically impossible, so the most suitable choice is a heuristic that shrinks the tree (hF), an anytime skeleton with little memory (IDS), and a game search that accounts for the opponent (alpha-beta).

---

## References

**Course and deck sources:**
- S. Russell, P. Norvig. *Artificial Intelligence: A Modern Approach*, 4th ed., Chapters 3 and 6; S. Okyay, week 04–05 slides. Criteria, uninformed search, A*, IDA*, RBFS, SMA*, beam, weighted A*, relaxed problems, PDB, landmarks, learned heuristics.
- O. Aichholzer, F. Aurenhammer, T. Werner. *Algorithmic Fun: Abalone*, TU Graz, 2002. Aba-Pro.
- P. Chorus. *Implementing a computer player for Abalone using alpha-beta and Monte-Carlo search*, MSc thesis, Maastricht, 2009. Alpha-beta vs Monte-Carlo.
- A. Papadopoulos et al. *Exploring optimization strategies in Abalone for alpha-beta search*, IEEE CIG, 2012.
- N. Lemmens. *Constructing an Abalone game-playing agent*, BSc thesis, Maastricht, 2005. Mean branching ≈ 60, mean game 87 plies.
- J. Gutstadt, K. Hogenson, J. Koerner. *Solving Abalone on small boards*, arXiv:2308.03208, 2023. Solutions on small boards (discussion, §7.2).
- J. Schaeffer et al. Checkers is solved. *Science* 317, 2007. Checkers solved, 5 × 10²⁰ boards (discussion, §7.2).

**Search algorithms:**
- I. Pohl. Heuristic search viewed as path finding in a graph. *Artificial Intelligence* 1, 1970. Weighted A*.
- R. E. Korf. Depth-first iterative-deepening: an optimal admissible tree search. *Artificial Intelligence* 27, 1985. IDA*.
- S. Russell. Efficient memory-bounded search methods. *ECAI*, 1992. SMA*.
- R. E. Korf. Linear-space best-first search. *Artificial Intelligence* 62, 1993. RBFS.
- A. Reinefeld, T. A. Marsland. Enhanced iterative-deepening search. *IEEE TPAMI* 16, 1994. IDA* with a transposition table.
- T. Yoshizumi, T. Miura, T. Ishida. [A* with partial expansion for large branching factor problems](https://mlanthology.org/aaai/2000/yoshizumi2000aaai-partial). *AAAI*, 2000. PEA*.
- M. Likhachev, G. Gordon, S. Thrun. ARA*: Anytime A* with provable bounds on sub-optimality. *NIPS*, 2003.
- R. Zhou, E. A. Hansen. Breadth-first heuristic search. *Artificial Intelligence* 170, 2006.
- E. A. Hansen, R. Zhou. Anytime heuristic search. *JAIR* 28, 2007. AWA*.
- A. Kishimoto, A. Fukunaga, A. Botea. Scalable, parallel best-first search for optimal sequential planning. *ICAPS*, 2009. HDA*.
- A. Felner et al. [Partial-expansion A* with selective node generation](https://ojs.aaai.org/index.php/AAAI/article/view/8137). *AAAI*, 2012. EPEA*.
- R. C. Holte, A. Felner, G. Sharon, N. R. Sturtevant. [Bidirectional search that is guaranteed to meet in the middle](https://ojs.aaai.org/index.php/AAAI/article/view/10436). *AAAI*, 2016. MM.
- J. Chen, R. C. Holte, S. Zilles, N. R. Sturtevant. [Front-to-end bidirectional heuristic search with near-optimal node expansions](https://www.ijcai.org/proceedings/2017/69). *IJCAI*, 2017. NBS.

**Heuristics:**
- J. C. Culberson, J. Schaeffer. Pattern databases. *Computational Intelligence* 14, 1998.
- B. Bonet, H. Geffner. Planning as heuristic search. *Artificial Intelligence* 129, 2001. h_max, h_add.
- J. Hoffmann, B. Nebel. The FF planning system. *JAIR* 14, 2001. h_FF.
- A. Felner, R. E. Korf, S. Hanan. Additive pattern database heuristics. *JAIR* 22, 2004.
- A. V. Goldberg, C. Harrelson. Computing the shortest path: A* search meets graph theory. *SODA*, 2005. ALT landmarks.
- M. Katz, C. Domshlak. Optimal additive composition of abstraction-based admissible heuristics. *ICAPS*, 2008. Cost partitioning.
- M. Helmert, C. Domshlak. Landmarks, critical paths and abstractions: what's the difference anyway? *ICAPS*, 2009. LM-cut.
- M. Helmert, P. Haslum, J. Hoffmann, R. Nissim. Merge-and-shrink abstraction. *Journal of the ACM* 61, 2014.
- J. Seipp, T. Keller, M. Helmert. [Saturated cost partitioning for optimal classical planning](https://jair.org/index.php/jair/article/view/11673). *JAIR* 67, 2020.

**Our measurements:**
- `python abalone_search.py bench`: BFS, UCS, A* hR, A* hF and greedy search, 60 puzzles (seed 314).
- `python abalone_search.py more`: IDA*, IDA* + TT and weighted A* (w = 2) on the same puzzles. Output: `results/more.json`.

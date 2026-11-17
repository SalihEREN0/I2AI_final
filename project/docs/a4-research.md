# Assignment 4 research: what the final agent should use, and where reinforcement learning fits

Date: 2026-10-27; section 4 updated 2026-10-28 with the measured results. Demo: 2026-11-04, online, 11 minutes per pair including Q&A.
Builds on [rl-literature-and-plan.md](rl-literature-and-plan.md) (the long-term deep-RL plan); this note narrows it to what can be built, measured and defended in eight days.

## 0. Summary

- **Goal of this stage:** a final agent that puts Chapters 1–4, 6 (adversarial search) and the basics of 19 (learning from examples), 22 (deep learning) and 23 (reinforcement learning) to work (AIMA 4th Global ed. numbering). It should build on the literature on Abalone, with sources, and add a model of our own. Its progress since Assignment 3 is shown with games played under the same conditions, and ablations show what each part contributes. It ends in a live game against an opponent, with a developer view showing the moves considered, their values and the reason for the choice.
- **Questions this note answers:** is there an RL algorithm that solves Abalone? Which algorithms does our agent use, why was each integrated or left out, and what did each contribute? Can RL be added to the agent itself?
- **Design principle:** a board alone shows only the result, not the AI. The screen also shows diagnostics, the calculations behind each decision and the methods at work (section 7).
- **What the literature says:**
  - Every strong Abalone program is **alpha-beta with a hand-built evaluation** (ABA-PRO, ABLA, Lemmens, Chorus). Pure Monte-Carlo search lost **all 60 games** to alpha-beta in Chorus (2009).
  - **No RL agent for Abalone has been shown to beat a strong alpha-beta agent.** The best published attempt, AlphaZero on Abalone (Claußen 2022), stayed behind minimax because of compute, a slow Python engine and drawn games. The PPO project (BY571) publishes no results; the MCTS + policy/value-net project (newmbewb) reports only wins against its own earlier generations, at ~3 days per generation on an RTX 3080 Ti.
  - The RL that has worked on Abalone is **TD(λ) self-play on an evaluation function** (Abalearn 2003: intermediate level, and only after adding risk-sensitivity, because defensive self-play never ends games).
- **Recommendation:**
  1. Core (Ch. 6): iterative-deepening alpha-beta with move ordering, killer moves, transposition table and quiescence search on pushes. This is the literature's answer and the biggest strength gain over Assignment 3.
  2. RL that we integrate (Ch. 23): **TDLeaf(λ) self-play to learn U's weights**, with the marble difference in the reward to break draws (Abalearn's lesson). Cheap on CPU, plugs straight into alpha-beta, and the learned weights stay readable in the developer view.
  3. Supervised comparison (Ch. 19): fit the same weights by **logistic regression on game outcomes** (Texel/Buro method) from the same self-play positions. One table then compares hand-set, supervised and RL weights.
  4. Small own model (Ch. 22): a **small value network** trained on the same data and dropped into alpha-beta in place of the linear U (the NNUE idea). Kept only if it beats the linear version under equal time.
  5. Reviewed and not integrated, with reasons: AlphaZero / Gumbel AlphaZero, MuZero, PPO, DQN, pure MCTS (section 4).
- **Outcome (measured, section 4):** items 1 and 2 are the final agent, which beats the Assignment 3 agent (greedy U + AND–OR) 48–2–0; the learned weights are the largest single contribution (0.83 at equal time). Item 3 predicts results best but loses to TDLeaf unless fitted on TDLeaf's own games. Item 4 did not win and is not integrated. A3's AND–OR planner runs in the developer view.

## 1. Goals mapped to work

| Goal | What we deliver | Evidence on screen |
|---|---|---|
| Open with the agent summary page | updated summary page (PEAS, U, chapters used) | first slide |
| Ch. 1–4 | agent, PEAS, rules engine (two generators agree on 2,000 positions), A*/AND–OR, local search results from A2–A3 | reused, one slide |
| Ch. 6 adversarial search | alpha-beta + enhancements; node counts with and without each | ablation table, live node counter |
| Ch. 19 learning from examples | logistic regression of outcome on U's features | fitted weights, loss curve |
| Ch. 22 deep learning (basics) | small value network as evaluation | train/val loss, match result vs linear U |
| Ch. 23 reinforcement learning (basics) | TDLeaf(λ) self-play weight learning | weights over training games, Elo/win-rate curve |
| Literature with sources | sections 2–3 of this note | references slide |
| Progress since A3, same conditions | A3 agent (greedy on U + AND–OR) vs final agent, both colours, fixed openings | match table with confidence intervals |
| What each part contributes | ablations (section 6) | ablation table |
| Adversarial demo + developer mode | user mode / developer mode views | live |

## 2. Classical Abalone agents: what the literature measured

| Work | Method | Evaluation features | Result |
|---|---|---|---|
| Aichholzer, Aurenhammer, Werner (TU Graz, 2002): **ABA-PRO** | alpha-beta, deep search | single feature: distance of marbles to own centre of mass (per Özcan & Hülagu) | reported never beaten at champion level; beat a three-time world champion (per Chorus) |
| **Özcan & Hülagu (Yeditepe, 2004): ABLA** | minimax + alpha-beta, 3–4 ply | f1 = Manhattan distance to centre (difference), f2 = friendly-neighbour count (difference); eval = −f1 + f2 | never lost to Random Soft (M) or shareware ABA-PRO (B, A); the 4-ply agent went 3–0–1 against the 3-ply one. **Draws came from cyclic, repeating moves.** |
| **Lemmens (Maastricht BSc, 2005)** | alpha-beta depth 2, move ordering (pushes first, larger groups first, towards centre), transposition table | ABLA's two features + break-strong-group, strengthen-group, marbles pushed off (own marbles weighted 50×); nine weight "modes" by game phase, set by trial and error | never lost to RandomAba, AliAba, NetAbalone; most games drawn by move loops. Average human game 87 plies, computer games ~130. Future work names **machine-learned weights**. |
| **Chorus (Maastricht MSc, 2009): AIBA** | alpha-beta + iterative deepening, move ordering, killer moves, transposition table, quiescence search; vs Monte-Carlo search | not checked in detail for this note | every enhancement helped; **alpha-beta won all 60 games against Monte-Carlo**, which pushed off only 3 marbles in total; level with ABA-PRO strength 3–4 (2 wins, 2 draws), lost both games to strength 9. A random simulated game can last 4,000 plies. |

Takeaways for us:

- Our U (A1) already holds the community's two core ideas (centre, cohesion) plus edge danger and marbles off. Lemmens' extra features (break-strong-group, strengthen-group) are cheap candidates for the learned version.
- **Weights in all of these are hand-set.** Lemmens explicitly lists learned weights as future work. That is the gap our RL step fills, and it is small enough to finish.
- **Draws by repetition** are the common failure of all classical agents. We need a ply cap, repetition detection and a marble-difference tie-break, both for play and for any learning signal.

## 3. Reinforcement learning on Abalone: everything we found

| Work | Method | Compute | Result | Lesson for us |
|---|---|---|---|---|
| [Schorr, 1999](https://github.com/FrederikSchorr/abalone-reinforcement-learning-1999) (TU Wien diploma thesis) | neural networks trained by self-play RL | 1999 PC | no strength numbers in the repository | earliest precedent |
| [Abalearn (Campos & Langlois, ECML 2003; ICGA J. 2003)](https://mlanthology.org/ecmlpkdd/2003/campos2003ecml-abalearn) | TD(λ) self-play, neural network on hand-designed features, shallow search | modest | intermediate level without expert data or deep search; **self-play only worked after making learning risk-seeking**, because defensive agents never finish games | the direct precedent for TD learning of an evaluation; we must reward progress (marble difference) or games stall |
| [Claußen, 2022](https://serwiss.bib.hs-hannover.de/frontdoor/index/index/year/2022/docId/2278) (Hochschule Hannover BSc) | AlphaZero (AlphaZero General, Python), 1,452 actions, 120 MCTS simulations | 3× P100 + 24 cores; ~40 s per self-play game; ~1 h training per iteration | best version (warm-started from minimax games) beat random every game, drew about half vs a heuristic agent, stayed **below minimax** | AlphaZero is out of reach in 8 days on a laptop; warm-starting from search games helped most |
| [newmbewb/abalone-ai](https://github.com/newmbewb/abalone-ai) | MCTS guided by small policy and value nets, 12-fold symmetry augmentation, generation 0 = pure MCTS | i3-7100 + RTX 3080 Ti, **~3 days per generation** | gen 1 beat gen 0 23–0, gen 2 beat gen 1 10–0; no results against outside agents | works, but too slow for us and unbenchmarked |
| [BY571/abalone-rl](https://github.com/BY571/abalone-rl) (MIT) | PPO self-play (TorchRL), 2,076-action space with legal masks, ELO tournament script | GPU | no published numbers; author notes reward shaping and exploration matter more than in sharper games | confirms sparse reward is the core difficulty |
| [Gutstadt et al., 2023](https://arxiv.org/abs/2308.03208) | retrograde solving of small boards | — | 2×2×2 board solved; 2×2×3 is a draw under perfect play | correctness reference for small boards, not an agent |

I also searched for 2024–2026 work (AlphaZero, MuZero, DQN, PPO on Abalone) and found nothing newer.

**Bottom line:** no published RL method "solves" Abalone or beats strong alpha-beta. The honest claim for the demo is: *we reviewed the RL work on Abalone, integrated the part that is known to work at our scale (TD learning of the evaluation, as in Abalearn and KnightCap), and measured what it added on top of classical search.*

## 4. Decision per algorithm, and what each gave

Updated 2026-10-28 with the measured results. Every match is 50 games (25 random 2-ply openings × both colours, both start positions), scored (wins + ½ draws) / games with a 95% confidence interval; details and raw games in [src/abalone/README.md](../src/abalone/README.md) and `experiments/`.

| Algorithm | Chapter | Decision | Why | What it gave (measured) |
|---|---|---|---|---|
| Alpha-beta + iterative deepening, move ordering, killers, TT, quiescence | 6 | **Integrated: the core** | strongest method for Abalone in every study above; Chorus measured each enhancement | the final agent beats the Assignment 3 agent (greedy U + AND–OR) **48–2–0**, and plain greedy U 50–0–0; alpha-beta with the hand-set U already scores 0.99 against the A3 agent. At equal time (0.5 s/move), switching off quiescence scores 0.70 (0.58–0.82) for the full agent, move ordering 0.64 (0.51–0.77); TT 0.58 and killers 0.53 are not significant at 50 games. Ordering cuts depth-3 search from ≈12,800 to ≈7,800 nodes per move (386 positions, experiments/search-cost). |
| **TDLeaf(λ)** self-play on U's weights ([Baxter et al. 2000](https://mlanthology.org/mlj/2000/baxter2000mlj-learning/)) | 23 | **Integrated: our RL step** | TD learning is the RL method with an Abalone precedent (Abalearn); TDLeaf trains the evaluation that alpha-beta then uses; linear weights stay readable | 1,200 self-play games in 6.4 minutes on 8 cores. Learned vs hand-set weights: 0.96 at depth 2, 0.93 at depth 3, **0.83 (0.74–0.92) at equal time**: the largest single contribution in the ablations. Most of the gain comes in the first 120 games (score 0.93). Marble-difference vs result-only reward: 24–1–25, no difference; both beat hand-set (0.96 / 0.89). |
| Logistic regression on outcomes (Texel tuning; Buro 1995) | 19 | **Built and compared; not in the final agent** | the supervised counterpart: positions labelled by the final result; tells us what RL adds beyond fitting outcomes | best predictor of results (test log-loss 0.511 vs TDLeaf 0.523, hand-set 0.574) and beats hand-set (0.88), but **loses to TDLeaf (0.28)** when fitted on the hand-set agent's games. Fitted on TDLeaf's own games it draws level (0.60, CI 0.48–0.72). |
| Small value network as evaluation (NNUE idea) | 22 | **Built and compared; not integrated** | lets a model learn feature interactions the linear U cannot; plugs into the same alpha-beta | 16 hidden units, 1,985 parameters, pure Python (≈30 µs per evaluation vs ≈16 µs for U). Predicts as well as TDLeaf (0.523) but plays only as well as hand-set (0.47) and **loses to TDLeaf 0–3–47** at equal depth and 1–0–49 at equal time. On TDLeaf's own games: still 0.07. The limit is the raw-board representation and ~1,000 games of data, not the source of the games. |
| AND–OR search for forced push-offs (A3's planner) | 4 | **Integrated as an analysis panel**, not in move choice | proves whether a capture can be forced against every reply; alpha-beta already searches those lines and weighs them against the rest of the position | reproduces Assignment 3's result; ≈10 ms per position. In 529 positions with a forceable push-off, alpha-beta chose a forcing move in 286 (54%): the panel shows where the agent prefers something else. |
| AlphaZero / Gumbel AlphaZero | 22–23 | **Reviewed, not integrated** | Claußen needed 3 GPUs and still stayed below minimax; newmbewb needed ~3 GPU-days per generation; our engine is pure Python (≈12,500 move-generations/s) and we have no CUDA GPU | cited as the state of the art and the long-term plan ([rl-literature-and-plan.md](rl-literature-and-plan.md)); our own value-network result (above) shows how far ~1,000 games are from enough |
| MuZero | 23 | **Reviewed, not integrated** | learns a model of the rules; we already have an exact, verified rules engine, and LightZero found AlphaZero more data-efficient when a simulator exists | — |
| PPO / policy gradient | 23 | **Reviewed, not integrated** | ~1,452–2,076 masked actions, sparse reward, no published Abalone result to compare to (BY571) | — |
| DQN / tabular Q-learning | 23 | **Reviewed, not integrated** | state space ~6.5×10²³ (Lemmens), so no table; DQN is built for single-agent settings, and self-play makes its target move | — |
| Pure Monte-Carlo search / MCTS with random rollouts | 6 | **Reviewed, not integrated** | lost 60 of 60 games to alpha-beta (Chorus); random games last up to 4,000 plies | one line in the talk, with the number |

**What reinforcement learning added, precisely.** TDLeaf's weights beat the hand-set ones clearly, and they beat logistic regression fitted on the hand-set agent's games. But logistic regression on the same six features, fitted on TDLeaf's *own* games, plays level with TDLeaf. So the advantage of RL here is not the update rule: it is that the learner generates its own training games, from the positions its own search reaches, and keeps adjusting as its play changes. Abalearn's warning that self-play stalls without a progress reward did not show up: the result-only reward learned the same direction, probably because the search already avoids repetitions and the random openings vary the games.

## 5. Feasibility, measured on our machine

- Hardware: Apple M3, 8 cores, 16 GB; no CUDA GPU.
- Our A2 engine (bitboards, standard library only): **≈12,500 node expansions/s and ≈550,000 generated children/s** on one core (2,000 expansions of the start position); perft(3) = 98,912 in 0.15 s.
- Estimate before building: on the order of 10³ nodes per move at depth 2 + quiescence, so several hundred training games per night with 8 processes. **Measured:** 1,200 TDLeaf self-play games at depth 2 took 6.4 minutes on 8 cores (alpha-beta needs far fewer nodes than estimated).
- The value network uses a pure-Python forward pass (no NumPy): ≈30 µs per evaluation against ≈16 µs for the linear U.

## 6. Experiment plan (same conditions, ablations)

Fixed conditions for every match: the same time (or node) budget per move, both colours equally, the same opening set (standard, Belgian Daisy, plus a fixed set of randomised 2-ply openings), a 200-ply cap scored by marble difference, repetition counted as a draw. Report wins / draws / losses, mean marble difference and a 95% confidence interval. We play 50 games per pairing, which resolves differences of roughly ±12 percentage points in score; a non-significant result is reported as "no effect detected".

| Experiment | Compares | Answers |
|---|---|---|
| E1 Progress since A3 | final agent vs A3 agent (greedy on U, AND–OR for forced captures) | how far the agent has come |
| E2 Search ablations | full alpha-beta vs without move ordering / TT / killers / quiescence | nodes per move, depth reached, win rate lost |
| E3 Evaluation | hand U vs logistic-regression U vs TDLeaf U (vs value net) at equal budget | what learning added (Ch. 19 vs 23 vs 22) |
| E4 Learning curve | TDLeaf checkpoints every N games vs hand U | does learning keep improving or plateau |
| E5 Reward design | TDLeaf with win/loss only vs with marble difference | Abalearn's draw problem, on our agent |
| E6 Convergence | TDLeaf for 3,200 games with a decaying step, three seeds, vs the 1,200-game run | whether the still-moving weights matter |

Hypotheses and pass criteria are written before each run, as in the long-term plan.

## 7. Developer view: the backend made visible

The view shows the moves the agent considers, how promising each looks, and why it picks one. A board alone would show only the result, so the screen also shows diagnostics, the calculations behind each decision and the methods at work. That makes the developer view a main deliverable, not decoration.

**Design rule: the agent writes a trace, and the UI only draws it.** Every decision produces one structured record: the position, the candidate moves with their values, the depth reached in each iteration, node, cutoff and transposition-table counts, U's feature breakdown, the active weight set and the time taken. Three things read the same record:
- the live developer view,
- the headless matches in section 6,
- the slides.

So every number on screen or on a slide comes from a logged calculation.

| Panel | Course topic | What it shows, computed live |
|---|---|---|
| Agent loop | Ch. 2 | percept → state → chosen action; PEAS card; current utility U |
| Candidate moves | Ch. 6 | the top-k root moves with alpha-beta value, rank, nodes searched under each, and which were cut off |
| Search statistics | Ch. 3, 6 | depth reached per iterative-deepening step; nodes, nodes/s, effective branching factor b* (as in A2), alpha-beta cutoff rate, TT hit rate, quiescence extensions |
| Principal variation | Ch. 6 | the expected line of play, drawn as ghost arrows on the board |
| Why this move | Ch. 6, 19 | U = Σ wᵢ·fᵢ split by feature for the chosen move and the runner-up; the difference per feature is the "why" |
| Forced sequences | Ch. 4 | AND–OR check: can a push-off be forced against every reply? (A3's planner, reused) |
| Learning | Ch. 19, 22, 23 | the active weight set (hand / logistic regression / TDLeaf / value net); the TD error of the last move, V(sₜ₊₁) − V(sₜ); the weights' path over training; the predicted win probability |
| Game timeline | Ch. 6, 23 | evaluation and win probability across plies, marbles off, repetition counter |
| Live ablation | Ch. 6, 19–23 | switches for move ordering, TT, quiescence and learned weights. Re-searching the same position shows what is lost (depth, nodes, chosen move). |

**User mode** shows only the board, whose turn it is, the score and the clock. A single key switches modes during play, so the audience can follow the game and then see why the agent played as it did.

What this means for the build:
- The search must count its own work (nodes, cutoffs, TT hits, per-root-move subtree sizes) at negligible cost; we measure that cost once.
- The per-move trace is a JSON log. A finished game can be replayed in developer view, which is our backup if the live demo fails.
- Panels that need extra computation (live ablation, AND–OR) run on request, not every move, so they don't slow the agent's clock.

## 8. Open points

- **Repetition rule** (resolved 2026-10-27). The official rules don't forbid repetition. We count the third occurrence of a position as a draw and stop at 200 plies, scored by marble difference ([game.py](../src/abalone/game.py)). The search scores a move back into an earlier position as a draw, so an agent that is ahead avoids repeating.
- **Abalearn full text.** Only the abstract was readable online; the exact features and TD parameters are not confirmed here.
- **Do the learned weights converge?** (resolved 2026-10-28). Strength levels off by ~1,300 self-play games; a 3,200-game run with a decaying step plays even with the 1,200-game one. Edge and exposed keep moving because the two features are correlated 0.71, so self-play constrains their difference only weakly; three seeds move the same way along it. The 1,200-game run stays the final agent (README, E6).
- **Value network** (resolved 2026-10-27). Built in pure Python, so the repo stays standard-library only.

## References

- Aichholzer, Aurenhammer, Werner. *Algorithmic Fun: Abalone*. TU Graz, 2002. [PDF](http://www.ist.tugraz.at/staff/aichholzer/research/rp/abalone/tele1-02_aich-abalone.pdf)
- Baxter, Tridgell, Weaver. *Learning to Play Chess Using Temporal Differences*. Machine Learning 40, 2000. [link](https://mlanthology.org/mlj/2000/baxter2000mlj-learning/)
- Buro. *Statistical Feature Combination for the Evaluation of Game Positions*. JAIR 3, 1995. (logistic regression of evaluation weights)
- Campos, Langlois. *Abalearn: A Risk-Sensitive Approach to Self-Play Learning in Abalone*. ECML 2003, pp. 35–46. [link](https://mlanthology.org/ecmlpkdd/2003/campos2003ecml-abalearn)
- Chorus. *Implementing a Computer Player for Abalone using Alpha-Beta and Monte-Carlo Search*. MSc thesis, Maastricht, 2009. [PDF](https://project.dke.maastrichtuniversity.nl/games/files/msc/pcreport.pdf)
- Claußen. *Mastering the Game of Abalone using Deep Reinforcement Learning and Self-Play*. BSc thesis, Hochschule Hannover, 2022. [record](https://serwiss.bib.hs-hannover.de/frontdoor/index/index/year/2022/docId/2278)
- Gutstadt, Hogenson, Koerner. *Solving Abalone on Small Boards*. 2023. [arXiv](https://arxiv.org/abs/2308.03208)
- Lemmens. *Constructing an Abalone Game-Playing Agent*. BSc paper, Maastricht, 2005. [PDF](https://project.dke.maastrichtuniversity.nl/games/files/bsc/Lemmens_BSc-paper.pdf)
- Özcan, Hülagu. *A Simple Intelligent Agent for Playing Abalone Game: ABLA*. Proc. 13th Turkish Symposium on AI and Neural Networks (TAINN), 2004, pp. 281–290. [PDF](https://people.cs.nott.ac.uk/pszeo/docs/publications/ABLA_id136final.pdf)
- Russell, Norvig. *Artificial Intelligence: A Modern Approach*, 4th Global ed. Ch. 6, 19, 22, 23.
- Texel's tuning method. [Chess Programming Wiki](https://www.chessprogramming.org/Texel%27s_Tuning_Method)
- Software: [BY571/abalone-rl](https://github.com/BY571/abalone-rl), [newmbewb/abalone-ai](https://github.com/newmbewb/abalone-ai), [Schorr 1999](https://github.com/FrederikSchorr/abalone-reinforcement-learning-1999)
- Further deep-RL references: see [rl-literature-and-plan.md](rl-literature-and-plan.md).

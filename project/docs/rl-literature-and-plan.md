# Reinforcement learning for Abalone: literature and plan

Date: 2026-10-06. Preliminary research for an experimental study, independent of the course project.

## 0. Summary

- **Proposed method:** Gumbel AlphaZero ([Danihelka et al., 2022](https://mlanthology.org/iclr/2022/danihelka2022iclr-policy/)).
  - The Abalone environment runs vectorised on the GPU with JAX, in the style of [Pgx](https://arxiv.org/abs/2303.17503).
  - Search uses the [mctx](https://github.com/google-deepmind/mctx) library.
  - Three additions on top:
    - [KataGo](https://arxiv.org/abs/1902.10565)'s efficiency improvements,
    - a value target that also learns the marble difference,
    - a warm start from alpha-beta games.
- **Baseline:** always measured against alpha-beta with a hand-crafted evaluation function. The metric is an Elo ladder.
- **Why this method:**
  - The strongest Abalone agents to date are minimax/alpha-beta based ([Aichholzer et al., 2002](http://www.ist.tugraz.at/staff/aichholzer/research/rp/abalone/tele1-02_aich-abalone.pdf); [Chorus, 2009](https://project.dke.maastrichtuniversity.nl/games/files/msc/pcreport.pdf)).
  - The only thorough study applying AlphaZero to Abalone ([Claußen, 2022](https://serwiss.bib.hs-hannover.de/frontdoor/index/index/year/2022/docId/2278)) fell behind minimax because of limited compute, a slow Python engine, and games drifting into draws.
  - The literature has a direct fix for each of these three problems (section 4). Unlimited CPU/GPU access also removes the biggest obstacle.
- **Open ground:** we found no published, strong deep-RL Abalone agent. A well-measured result, even a negative one, would be new knowledge.

## 1. Introduction to RL: concepts and reading order

### Concepts to know

| Concept | In Abalone |
|---|---|
| Agent, environment, state, action | the player; the game engine; the board and side to move; a legal move |
| Reward, return | +1 / −1 at the end of the game, 0 for a draw (or the marble difference) |
| Policy π(a\|s) | the probability of choosing each move in a position |
| Value function V(s) | the expected game result from this position |
| Exploration / exploitation | a known good move, or an untried one |
| MDP | the state, action, transition, reward framework; the zero-sum version for two-player games |
| TD learning | updating V(s) towards the estimate of the next position (TD-Gammon, Abalearn) |
| Policy gradient, PPO | updating the policy directly in the direction that increases reward |
| Self-play | the agent generating data by playing against a copy of itself |
| MCTS (PUCT) | tree search guided by the network's suggestions; AlphaZero's policy-improvement step |
| Replay buffer | the pool of (position, search policy, result) triples from self-play |
| Elo | a rating that compares agents' strength |

### Reading and implementation order

1. Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed. ([free](http://incompleteideas.net/book/the-book-2nd.html)). Chapters 1–6 are the basics, 8 planning, 13 policy gradients.
2. OpenAI [Spinning Up](https://spinningup.openai.com), Part 1: concepts and terminology.
3. Papers, in this order:
   - AlphaGo Zero (Silver et al., 2017),
   - AlphaZero ([Silver et al., 2018](https://arxiv.org/abs/1712.01815)),
   - [KataGo](https://arxiv.org/abs/1902.10565),
   - [Gumbel](https://mlanthology.org/iclr/2022/danihelka2022iclr-policy/).
4. Practice: run Gumbel AlphaZero end to end on a small game (Connect Four or Othello) with [Pgx](https://github.com/sotetsuk/pgx) and [mctx](https://github.com/google-deepmind/mctx) first. This is the shortest way to learn the pipeline.

## 2. Literature: how RL in games developed

| Year | Work | Idea | Relevance for Abalone |
|---|---|---|---|
| 1992–95 | TD-Gammon (Tesauro) | TD(λ) with a neural network, self-play only | world-class backgammon; Abalearn's starting point |
| 1999 | [KnightCap / TDLeaf(λ)](https://mlanthology.org/mlj/2000/baxter2000mlj-learning/) | applying TD learning to the leaves of an alpha-beta search | chess rating from 1650 to 2150 in 308 games and 3 days; a cheap way to learn evaluation weights |
| 2003 | [Abalearn](https://www.researchgate.net/publication/220174489_ABALEARN_a_Program_that_Learns_how_to_Play_Abalone) | self-play with TD(λ) and a neural network; a risk-taking setting | the first learning agent for Abalone; intermediate level; managed draws against Aba-Pro up to depth 5 ([Claußen, 2022](https://serwiss.bib.hs-hannover.de/frontdoor/index/index/year/2022/docId/2278)) |
| 2017 | [Expert Iteration](https://discovery.ucl.ac.uk/10038400/) | search is the "expert", the network the "apprentice"; the network imitates the search | trained from scratch in Hex and beat MoHex 1.0 |
| 2017–18 | AlphaGo Zero, [AlphaZero](https://arxiv.org/abs/1712.01815) | MCTS and a ResNet, self-play only | chess, shogi, Go; self-play on 5,000 TPUs, training on 64 TPUs; 9 hours for chess |
| 2018–20 | [NNUE](https://www.chessprogramming.org/NNUE) (Yu Nasu, later Stockfish) | alpha-beta with a small, incrementally updated evaluation network | the hybrid path: a learned network together with strong classical search |
| 2019 | [KataGo](https://arxiv.org/abs/1902.10565) | efficiency improvements to AlphaZero | 50× less compute; passed ELF in 19 days on fewer than 30 GPUs; most improvements are game-independent |
| 2020 | MuZero | also learns the model; no rule knowledge needed | we know the rules; [LightZero](https://arxiv.org/abs/2310.08348) showed AlphaZero is more data-efficient when a simulator exists |
| 2020 | [Polygames](https://arxiv.org/abs/2001.09832) | fully convolutional network with global pooling: board-size-independent bots | beat the strongest humans at Hex 19×19; a model for transfer from small to large boards |
| 2020 | [Warm-Start AlphaZero](https://arxiv.org/abs/2004.12357) | compensating for the network's weakness in early iterations with rollouts/RAVE | the warm start clearly increased Elo |
| 2022 | [Gumbel AlphaZero/MuZero](https://mlanthology.org/iclr/2022/danihelka2022iclr-policy/) | guaranteed policy improvement even with few simulations | on par with the best in Go, chess and Atari; learns even with 2 simulations |
| 2023 | [Pgx](https://arxiv.org/abs/2303.17503) | simulating thousands of games in parallel on the GPU with JAX | 10–100× faster than Python implementations; ships a Gumbel AlphaZero example |
| 2023 | [LightZero](https://github.com/opendilab/LightZero) | a common benchmark for MCTS and RL algorithms | with few simulations Gumbel MuZero beats standard MuZero |
| 2024 | [Self-play survey](https://arxiv.org/abs/2408.01072) | a general survey of self-play methods | background reading |

**State of the art:** for two-player, perfect-information games with known rules, the AlphaZero family is the strongest method. Improvements such as KataGo and Gumbel have cut the compute cost by two orders of magnitude. In chess, the hybrid that combines strong classical search with a small learned network (NNUE) is also at the very top.

## 3. RL work specific to Abalone

- **[Schorr, 1999](https://github.com/FrederikSchorr/abalone-reinforcement-learning-1999) (TU Wien diploma thesis).** Neural-network players trained by self-play. The repository has no strength measurement; it's in the thesis PDF.
- **Abalearn (2003).** TD(λ) and a neural network. According to Abalearn, self-play was ineffective at first because the agent repeated the same moves and never finished the game. A risk-taking setting was added as the fix.
- **[Claußen, 2022](https://serwiss.bib.hs-hannover.de/frontdoor/index/index/year/2022/docId/2278) (Hochschule Hannover, bachelor's thesis).** The most thorough study applying AlphaZero to Abalone.
  - **Setup:**
    - the AlphaZero General (Python) framework.
    - The hexagonal board embedded in a 9×9 axial matrix.
    - A fixed action space of 1,452 moves; 120 MCTS simulations per move.
    - Games cut at 200 moves and scored by marble difference.
    - Hardware: 3×P100 and a 24-core CPU.
  - **Bottlenecks:**
    - An Abalone game takes on average 40× longer than Othello (40 s vs 1 s).
    - Network training takes about an hour per iteration.
    - Most games end in a draw or with a small margin.
  - **Best version:** a network warm-started with games from Verloop's minimax agent. It beat the random agent every game and drew about half its games against the heuristic agent, but stayed weaker than minimax.
  - **The author's conclusion:** Abalone's strategic complexity is low compared with chess, so human intuitions are strong. The method is promising but needs much more compute or efficiency.
- **GitHub projects:**
  - [BY571/abalone-rl](https://github.com/BY571/abalone-rl): self-play with PPO, TorchRL. Uses a 2,076-action space and notes that reward design and exploration matter more than in tactical games. No results report.
  - [AlSaeed/AlphaZero](https://github.com/AlSaeed/AlphaZero): unfinished.
  - [gym-abalone](https://github.com/towzeur/gym-abalone): a Gym environment.
  - [MichielVerloop/AbaloneAI](https://github.com/MichielVerloop/AbaloneAI): an open-source Java alpha-beta agent. Claußen used it as a benchmark opponent; a ready-made opponent for us too.
- **Theoretical result:** [Gutstadt et al., 2023](https://arxiv.org/abs/2308.03208). The 2×2×2 board is fully solved; on the 2×2×3 board perfect play is a draw. A ready reference for testing an agent's correctness on small boards.

## 4. What makes Abalone hard for RL, and proposed fixes

| Difficulty | Evidence | Proposed fix |
|---|---|---|
| Long, unending games drifting to draws | most of Claußen's games are draws; a random game can stretch to 4,000 plies (Chorus); the rules don't forbid repetition | a move limit (initially 200) scored by marble difference; repetition detection; varied openings (standard, Belgian Daisy, random first moves); resigning clearly decided games |
| Sparse reward (only on the 6th marble) | Claußen's score distribution | a value target that includes the marble difference besides win/loss; an extra "final marble difference" prediction head (like KataGo's score target) |
| Large action space | 1,452 (Claußen) or 2,076 (BY571) encoded actions, ~60 legal moves per position | a fixed move encoding with a legal-move mask; Gumbel, which works with few simulations |
| Hexagonal board | convolutional networks expect a square grid | 9×9 axial embedding with a valid-cell mask; data augmentation with the 12 symmetries (6 rotations × mirror) |
| Slow engine | 40 s per game in Python | write the engine vectorised in JAX on the GPU; thousands of games in parallel |
| Cold start | Claußen's best result came from a warm-started network | warm start from alpha-beta games, as a controlled experiment |
| Playing only against itself | overfitting to previous versions | also play against past best checkpoints (Polygames) |

## 5. Proposed method

### 5.1 Two paths, shared infrastructure

- **Path A, main experiment:** Gumbel AlphaZero, from scratch or warm-started.
- **Path B, hybrid:** train a small value network on Path A's data and use it as alpha-beta's evaluation function (the NNUE approach). This gives a comparison and a fast agent at play time.
- **Baseline:** alpha-beta with the hand-crafted U (centre, cohesion, edge, pushed off). Its weights will be tuned with TDLeaf(λ); this will be the team's first and cheapest RL experience.

### 5.2 Environment (the biggest technical risk)

- **A readable Python reference engine:** legal move generation and application.
- **A JAX engine:** state arrays (9×9 axial, side to move, pushed-off counts, move counter), a vectorised legal-move mask, `step`, `terminal`, `reward`. Written to Pgx's API, the mctx examples run directly.
- **Validation:**
  - A "perft" test: legal-move counts to depth d from given positions must match in three engines (Python, JAX, Verloop's Java engine).
  - Testing that the 12 symmetries work correctly.
  - A rules bug invalidates all training, so this step is never skipped.

### 5.3 Network and training

- **Input planes:**
  - our marbles,
  - the opponent's marbles,
  - the valid-cell mask,
  - both sides' pushed-off counts (as constant planes),
  - time left to the move limit.
- **Network:** a fully convolutional ResNet with global pooling, like Polygames; transferable from small to large boards.
- **Outputs:**
  - policy: the encoded move space with the legal mask,
  - value: win / draw / loss,
  - auxiliary head: the final marble difference.
- **Search:** Gumbel AlphaZero, mctx. Start with few simulations (e.g. 16–32) and increase if needed.
- **To take from KataGo:**
  - playout cap randomisation,
  - auxiliary targets,
  - global pooling.
- **No evaluation gate:** AlphaZero updates a single network continuously; a new network isn't discarded for failing to beat the old one. In Claußen, the gate led to only 4 of 16 networks being accepted.
- **Curriculum:**
  - Small hexagonal boards first (side 3 or 4 cells, fewer marbles), then the full board.
  - On a small board the pipeline can be tested end to end in a few hours.
  - The boards solved by Gutstadt et al. serve as a correctness reference.

## 6. Measurement and experiment rules

- **Opponent ladder:**
  - random,
  - greedy (1 move and U),
  - alpha-beta with U (depth 2, 3, 4),
  - Verloop's AbaloneAI,
  - our own previous checkpoints.
- **Rating:** Elo, with BayesElo or Ordo.
  - Colours are balanced in every pairing, and a fixed opening set is used.
  - The number of games is fixed in advance from the target confidence interval.
- **Measured:**
  - win, draw and loss rates,
  - mean marble difference and mean game length,
  - positions generated per second, GPU utilisation,
  - Elo gain per GPU-hour.
- **In every experiment:**
  - the hypothesis and pass criterion are written before the run,
  - only one variable changes between baseline and variant,
  - at least 2 different random seeds are used,
  - all runs are logged under `experiments/`.

## 7. Stages

| Stage | Work | Done when |
|---|---|---|
| 0. Learning | Sutton and Barto 1–6, Spinning Up; Gumbel AlphaZero on Connect Four or Othello with Pgx and mctx | on the small game the agent clearly beats random and a simple heuristic agent |
| 1. Engine | Python reference and JAX engine; perft and symmetry tests; speed measurement | the three engines give the same move counts; games per second measured |
| 2. Baselines | random, greedy, alpha-beta with U; weight tuning with TDLeaf(λ); Elo ladder | the ladder is set up; tuned weights statistically beat the initial weights, or the failure is documented |
| 3. Small board | Gumbel AlphaZero on a small board | the learning curve rises; it approaches the correct result on solved small boards |
| 4. Full board | Gumbel AlphaZero; controlled experiments: warm start (on/off), value target (result only / result and marble difference), number of simulations | Elo curves and compute cost recorded; the best version measured against the alpha-beta baseline |
| 5. Hybrid | plug the value network into alpha-beta as its evaluation function | hybrid, pure MCTS agent and classical alpha-beta compared on the same ladder |
| 6. Report | results, including negative ones | method, compute cost and Elo comparison written up |

I don't give time estimates. The main uncertainty is how much compute learning in Abalone needs before it saturates, and only the measurements in stages 3–4 will show that. Stages 1–2 are classical software work and need no compute.

## 8. Risks and open decisions

- **Risks:**
  - Rules bugs: prevented by perft tests and the three-engine comparison.
  - Locking into draws: prevented by the marble-difference target, opening variety and resignation.
  - Marble-difference reward causing unwanted behaviour (reward hacking): winning always takes priority.
  - Measurement noise: prevented by enough games and confidence intervals.
- **Decisions needed:**
  1. Framework: JAX (Pgx and mctx; recommended) or PyTorch (LightZero).
  2. Rule variant: standard start, Belgian Daisy, or both; repetition and move-limit rules.
  3. Success criterion. It isn't proven that AlphaZero will surpass alpha-beta in Abalone (Claußen 2022 stayed behind minimax); that is the research itself. Instead of a single threshold, a continue-or-stop decision is made at each rung. Numerical thresholds are written before the run:

     | Rung | Question | Example criterion |
     |---|---|---|
     | B0 Pipeline | Is what we built correct? | the Pgx example gets stronger at Connect Four; the Abalone engines' legal-move counts match exactly |
     | B1 Small board | Is there learning in Abalone? | Elo rises; clearly beats the random and greedy agents |
     | B2 Full board | Does the learning signal persist? | beats the random and greedy agents; the draw rate falls |
     | B3 On par with classical | Does it catch up with known methods? | the level of alpha-beta with U at depth 2–3 |
     | B4 The real goal | Does it surpass classical? | surpasses depth-4 alpha-beta or Verloop's agent in Elo |

     If the Elo curve flattens on a rung, diagnose the cause first, then add compute. Wherever it stops, the result is written up.

## References

- Aichholzer, Aurenhammer, Werner. *Algorithmic Fun: Abalone*. TU Graz, 2002. [PDF](http://www.ist.tugraz.at/staff/aichholzer/research/rp/abalone/tele1-02_aich-abalone.pdf)
- Anthony, Tian, Barber. *Thinking Fast and Slow with Deep Learning and Tree Search*. NeurIPS 2017. [UCL](https://discovery.ucl.ac.uk/10038400/)
- Baxter, Tridgell, Weaver. *Learning to Play Chess Using Temporal Differences*. Machine Learning, 2000. [link](https://mlanthology.org/mlj/2000/baxter2000mlj-learning/)
- Cazenave et al. *Polygames: Improved Zero Learning*. ICGA Journal, 2020. [arXiv](https://arxiv.org/abs/2001.09832)
- Chorus. *Implementing a Computer Player for Abalone using Alpha-Beta and Monte-Carlo Search*. Master's thesis, Maastricht, 2009. [PDF](https://project.dke.maastrichtuniversity.nl/games/files/msc/pcreport.pdf)
- Claußen. *Mastering the Game of Abalone using Deep Reinforcement Learning and Self-Play*. Bachelor's thesis, Hochschule Hannover, 2022. [record](https://serwiss.bib.hs-hannover.de/frontdoor/index/index/year/2022/docId/2278)
- Danihelka, Guez, Schrittwieser, Silver. *Policy Improvement by Planning with Gumbel*. ICLR 2022. [link](https://mlanthology.org/iclr/2022/danihelka2022iclr-policy/)
- Gutstadt, Hogenson, Koerner. *Solving Abalone on Small Boards*. 2023. [arXiv](https://arxiv.org/abs/2308.03208)
- Koyamada et al. *Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning*. 2023. [arXiv](https://arxiv.org/abs/2303.17503)
- Langlois, Campos. *ABALEARN: a Program that Learns how to Play Abalone*. ICGA Journal 26(4), 2003. [ResearchGate](https://www.researchgate.net/publication/220174489_ABALEARN_a_Program_that_Learns_how_to_Play_Abalone)
- Niu et al. *LightZero: A Unified Benchmark for MCTS in General Sequential Decision Scenarios*. NeurIPS 2023. [arXiv](https://arxiv.org/abs/2310.08348)
- Silver et al. *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*. 2017; Science 2018. [arXiv](https://arxiv.org/abs/1712.01815)
- Sutton, Barto. *Reinforcement Learning: An Introduction*, 2nd ed. 2018. [book](http://incompleteideas.net/book/the-book-2nd.html)
- Wang, Preuss, Plaat. *Warm-Start AlphaZero Self-Play Search Enhancements*. PPSN 2020. [arXiv](https://arxiv.org/abs/2004.12357)
- Wu. *Accelerating Self-Play Learning in Go* (KataGo). 2019. [arXiv](https://arxiv.org/abs/1902.10565)
- Zhang et al. *A Survey on Self-play Methods in Reinforcement Learning*. 2024. [arXiv](https://arxiv.org/abs/2408.01072)
- Software: [mctx](https://github.com/google-deepmind/mctx), [Pgx](https://github.com/sotetsuk/pgx), [LightZero](https://github.com/opendilab/LightZero), [MichielVerloop/AbaloneAI](https://github.com/MichielVerloop/AbaloneAI), [BY571/abalone-rl](https://github.com/BY571/abalone-rl), [gym-abalone](https://github.com/towzeur/gym-abalone), [Schorr 1999](https://github.com/FrederikSchorr/abalone-reinforcement-learning-1999), [NNUE](https://www.chessprogramming.org/NNUE)

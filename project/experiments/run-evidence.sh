#!/bin/sh
# Evidence runs for Assignment 4 (docs/a4-research.md §6). Run from the repository root:
#   sh experiments/run-evidence.sh
# Each match: 25 random 2-ply openings, both start positions, each played with both colours (50 games).
# Matches use all CPU cores; the runs are sequential so they don't compete for them.
set -e
export PYTHONPATH=src
LEARNED=experiments/tdleaf-d2-marbles/weights-final.json
T=0.5   # seconds per move in the equal-time matches

match() {  # match OUT A B
  out=$1; shift
  echo "== $out: $1 vs $2"
  python3 -m abalone match "$1" "$2" --openings 25 --out "experiments/$out" > /dev/null
  python3 -c "import json; s=json.load(open('experiments/$out/summary.json')); print('  ', s['a_wins'], s['draws'], s['a_losses'], 'score', s['a_score'], s['a_score_ci95'], 'reasons', s['reasons'], 'margin', s['a_mean_marble_margin'], '%ss' % s['seconds'])"
}

# E1 progress since Assignment 3: the final agent vs the A3 agent as presented (greedy on U + AND–OR for a push-off
# it can force), then the steps in between: AND–OR's own gain over plain greedy U, and alpha-beta with the hand-set U
match e1-final-vs-a3 "ab:time=$T,weights=$LEARNED,name=final" greedy
match e1-final-vs-a3-andor "ab:time=$T,weights=$LEARNED,name=final" greedy-andor
match e1-a3-andor-vs-greedy greedy-andor greedy
match e1-hand-vs-a3-andor "ab:time=$T,name=hand-U" greedy-andor

# E2 ablations at equal time: the final agent vs itself with one part switched off
FULL="ab:time=$T,weights=$LEARNED,name=full"
match e2-ablate-tt          "$FULL" "ab:time=$T,weights=$LEARNED,tt=0,name=no-tt"
match e2-ablate-ordering    "$FULL" "ab:time=$T,weights=$LEARNED,ordering=0,name=no-ordering"
match e2-ablate-killers     "$FULL" "ab:time=$T,weights=$LEARNED,killers=0,name=no-killers"
match e2-ablate-quiescence  "$FULL" "ab:time=$T,weights=$LEARNED,quiescence=0,name=no-quiescence"
match e2-ablate-learning    "$FULL" "ab:time=$T,name=hand-weights"

# E4 learning curve: TDLeaf checkpoints vs the hand-set weights, depth 2 (weights read back from log.jsonl).
# The 150-iteration run of the paper, a checkpoint every 15 iterations (120 games); the 400-iteration run is below.
mkdir -p experiments/e4-learning-curve
python3 - <<'EOF'
import json
for line in open("experiments/tdleaf-d2-marbles/log.jsonl"):
    r = json.loads(line)
    if r["iteration"] % 15 == 0:
        json.dump(r["weights"], open("experiments/e4-learning-curve/checkpoint-%04d.json" % r["iteration"], "w"), indent=1)
EOF
for it in 0015 0030 0045 0060 0075 0090 0105 0120 0135 0150; do
  match "e4-learning-curve/it$it" "ab:depth=2,weights=experiments/e4-learning-curve/checkpoint-$it.json,name=it$it" "ab:depth=2,name=hand"
done

# E5 reward design: the same training with the result-only reward (draws give 0)
python3 -m abalone train --out experiments/tdleaf-d2-result --iterations 150 --games 8 --depth 2 --alpha 10 --reward result | awk 'NR % 25 == 0'
match e5-result-vs-hand    "ab:depth=2,weights=experiments/tdleaf-d2-result/weights-final.json,name=result-reward" "ab:depth=2,name=hand"
match e5-marbles-vs-result "ab:depth=2,weights=$LEARNED,name=marbles-reward" "ab:depth=2,weights=experiments/tdleaf-d2-result/weights-final.json,name=result-reward"
# Convergence check: the same TDLeaf run for 400 iterations with a decaying step size, three seeds. Strength
# levels off by ~160 iterations; edge and exposed keep moving along the direction the positions constrain least.
for s in 1 2 3; do
  out=experiments/tdleaf-d2-marbles-400; [ $s = 1 ] || out=$out-seed$s
  python3 -m abalone train --out $out --iterations 400 --games 8 --depth 2 --alpha 10 --alpha-decay 200 --seed $s | awk 'NR % 100 == 0'
done
python3 -m abalone train --out experiments/tdleaf-d2-result-400 --iterations 400 --games 8 --depth 2 --alpha 10 --alpha-decay 200 --reward result | awk 'NR % 100 == 0'
mkdir -p experiments/e4-learning-curve-400
python3 - <<'EOF'
import json
for line in open("experiments/tdleaf-d2-marbles-400/log.jsonl"):
    r = json.loads(line)
    if r["iteration"] % 40 == 0:
        json.dump(r["weights"], open("experiments/e4-learning-curve-400/checkpoint-%04d.json" % r["iteration"], "w"), indent=1)
EOF
for it in 0040 0080 0120 0160 0200 0240 0280 0320 0360 0400; do
  match "e4-learning-curve-400/it$it" "ab:depth=2,weights=experiments/e4-learning-curve-400/checkpoint-$it.json,name=it$it" "ab:depth=2,name=hand"
done
match convergence-400-vs-150-d2   "ab:depth=2,weights=experiments/tdleaf-d2-marbles-400/weights-final.json,name=it400" "ab:depth=2,weights=$LEARNED,name=it150"
match convergence-400-vs-150-time "ab:time=$T,weights=experiments/tdleaf-d2-marbles-400/weights-final.json,name=it400" "ab:time=$T,weights=$LEARNED,name=it150"
python3 experiments/convergence/analyse.py
echo "== done"

#!/bin/sh
# Experiments for the midterm paper (November 2026): the final agent on the faster engine, which searches exactly
# like the Assignment 4 agent at a fixed depth (tests/test_speed.py) but faster, so it searches deeper in the
# same time. Run from the repository root:  sh experiments/run-final.sh
# Results go to experiments/final/<name>/; the Assignment 4 results in experiments/ stay as they were.
# Each match uses all CPU cores, so the runs are sequential. A finished match is skipped, so after an
# interruption the script continues where it stopped.
set -e
export PYTHONPATH=src
PY=${PYTHON:-python}
LEARNED=experiments/tdleaf-d2-marbles/weights-final.json
T=0.5   # seconds per move
mkdir -p experiments/final
LOG=experiments/final/run.log

match() {  # match NAME OPENINGS A B   (games = 2 x OPENINGS: every opening with both colours)
  name=$1; n=$2; shift 2
  if [ -f "experiments/final/$name/summary.json" ]; then echo "skip $name (done)" | tee -a $LOG; return; fi
  echo "$(date '+%F %T') start $name: $1 vs $2 ($((2 * n)) games)" | tee -a $LOG
  $PY -m abalone match "$1" "$2" --openings "$n" --out "experiments/final/$name" > /dev/null
  $PY -c "import json; s=json.load(open('experiments/final/$name/summary.json')); print('  ', s['a_wins'], s['draws'], s['a_losses'], 'score', s['a_score'], s['a_score_ci95'], 'reasons', s['reasons'], '%ss' % s['seconds'])" | tee -a $LOG
}

FINAL="ab:time=$T,weights=$LEARNED,name=final"

# 1. The faster engine against the agent as presented in Assignment 4: same weights, same rules, same clock
match final-vs-a4 50 "$FINAL" "a4:time=$T,weights=$LEARNED,name=a4-agent"

# 2. E1 again: progress since Assignment 3
match e1-final-vs-a3-andor 25 "$FINAL" greedy-andor
match e1-hand-vs-a3-andor 25 "ab:time=$T,name=hand-U" greedy-andor

# 3. E2 again: the final agent against itself with one part switched off, at equal time
match e2-ablate-learning   25 "$FINAL" "ab:time=$T,name=hand-weights"
match e2-ablate-quiescence 25 "$FINAL" "ab:time=$T,weights=$LEARNED,quiescence=0,name=no-quiescence"
match e2-ablate-ordering   25 "$FINAL" "ab:time=$T,weights=$LEARNED,ordering=0,name=no-ordering"
match e2-ablate-tt         25 "$FINAL" "ab:time=$T,weights=$LEARNED,tt=0,name=no-tt"
match e2-ablate-killers    25 "$FINAL" "ab:time=$T,weights=$LEARNED,killers=0,name=no-killers"

# 4. The alternatives of Chapter 6, with the same evaluation and the same clock
match mcts-vs-final    25 "mcts:time=$T,weights=$LEARNED,name=mcts" "$FINAL"
match minimax-vs-final 25 "mm:time=$T,weights=$LEARNED,name=minimax" "$FINAL"

# 5. Timing. Each run is one process pinned to one logical CPU with high priority and no power throttling
#    (src/abalone/steady.py), so nothing else may run meanwhile, and the laptop must be on AC power: on battery
#    every core runs slower. Node counts do not depend on this; times and the depth reached in 0.5 s do.
export BENCH_CPU=${BENCH_CPU:-0}
on_battery() {  # true only where Windows reports a battery that is not on AC power
  command -v powershell > /dev/null &&
    [ "$(powershell -NoProfile -Command '(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus).PowerOnline' 2> /dev/null | tr -d '\r')" = "False" ]
}
timing() {  # timing NAME FOLDER   (runs FOLDER/run.py, which writes FOLDER/summary.json)
  if [ -f "$2/summary.json" ]; then echo "skip $1 (done)" | tee -a $LOG; return; fi
  if on_battery; then echo "$(date '+%F %T') $1 not run: on battery" | tee -a $LOG; exit 1; fi
  echo "$(date '+%F %T') start $1 (CPU $BENCH_CPU)" | tee -a $LOG
  $PY "$2/run.py" > "$2/run.log"
}
timing engine-speed    experiments/engine-speed       # the two engines at a fixed depth and in 0.5 s
timing search-cost     experiments/final/search-cost   # what pruning and each enhancement save, with minimax
timing baselines-depth experiments/final/baselines-depth   # how far each searcher gets in 0.5 s

# 6. The three search enhancements 50 games could not resolve, with four times the games
match e2-ablate-tt-200       100 "$FINAL" "ab:time=$T,weights=$LEARNED,tt=0,name=no-tt"
match e2-ablate-killers-200  100 "$FINAL" "ab:time=$T,weights=$LEARNED,killers=0,name=no-killers"
match e2-ablate-ordering-200 100 "$FINAL" "ab:time=$T,weights=$LEARNED,ordering=0,name=no-ordering"

echo "$(date '+%F %T') all done" | tee -a $LOG

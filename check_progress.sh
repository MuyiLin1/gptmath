#!/bin/zsh
# Shows which review experiment is running, its latest progress lines, and whether everything is done.
cd "${0:A:h}"
OUT=results/review
STATUS=$OUT/status.txt
[[ -f $STATUS ]] || { echo "Not started yet (no $STATUS)."; exit 0; }
echo "=== status ($(date '+%T')) ==="
cat $STATUS
if grep -q "ALL DONE" $STATUS; then
  echo "\n>>> ALL DONE. Logs: export/$OUT/logs/   Results: export/$OUT/*.json"
  exit 0
fi
if pgrep -f run_review_experiments.sh > /dev/null; then echo "\nrunner: alive"; else echo "\nrunner: NOT running (crashed or stopped?)"; fi
current=$(awk '/START/ && $3!="RUNNER" {n=$4} /END/ {d=$4} END {if (n!=d) print n}' $STATUS)
if [[ -n $current ]]; then
  echo "\n=== latest lines of $current ==="
  awk '{a[NR]=$0} END {for (i=(NR>6?NR-5:1); i<=NR; i++) print a[i]}' $OUT/logs/$current.log
fi

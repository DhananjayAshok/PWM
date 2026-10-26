#!/usr/bin/env bash
# Exits (printing the reason) when the run needs attention, so a supervising session can be woken by it. Every 15 min:
#   - sky cannot reach the API server
#   - one of our jobs FAILED that was not failed when the watchdog started
#   - no new data or run output for 2 h while our jobs are RUNNING
#   - the reaper is not running
#   bash cluster/sail/watchdog.sh    (run in the background)
source configs/config.env || { echo "configs/config.env not found"; exit 1; }
# `sky jobs queue -a` (finished jobs included) has a WORKSPACE column: the job name is field 4
seen=$(mktemp)
sky jobs queue -a 2>/dev/null | awk '$4 ~ /^(pwm-|dd-|ddre-|vj2|dv3)/ && /FAILED/ {print $1}' | sort > "$seen"
START=$(date +%s)
while true; do
  sleep 900
  q=$(timeout 300 sky jobs queue -a 2>&1) || { echo "SKY UNREACHABLE: $(echo "$q" | tail -2)"; exit 0; }
  new=$(comm -13 "$seen" <(echo "$q" | awk '$4 ~ /^(pwm-|dd-|ddre-|vj2|dv3)/ && /FAILED/ {print $1}' | sort))
  [[ -n "$new" ]] && { echo "FAILED jobs: $new"; exit 0; }
  running=$(echo "$q" | awk '$4 ~ /^(pwm-|dd-|ddre-|vj2|dv3)/ && /RUNNING/' | wc -l)
  recent=$(find "${storage_dir%/}/data" "${storage_dir%/}/sweeps" "${storage_dir%/}/checkpoints" "${results_dir%/}/sweeps" -maxdepth 4 \( -name episodes.jsonl -o -name "*.md" -o -name "*.json" -o -name train.log \) \
           -newermt "-2 hours" 2>/dev/null | head -1)
  if [[ $running -gt 0 && -z "$recent" && $(( $(date +%s) - START )) -gt 7200 ]]; then
    echo "NO PROGRESS for 2 h with $running jobs RUNNING"; exit 0
  fi
  ps -eo cmd | grep -Eq "reaper\.sh" || { echo "NOT RUNNING: reaper"; exit 0; }
done

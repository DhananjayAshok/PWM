#!/usr/bin/env bash
# Dev-box loop: every 10 min, cancel our jobs whose work is finished but which still show RUNNING (on this cluster a
# child process can keep a job alive after its script ends). Finished = the job log's tail shows an end marker of our
# scripts; a 5 min grace lets the job exit by itself first. Light (sky queue/logs calls only).
#   setsid nohup bash cluster/sail/reaper.sh >> "$results_dir/logs/reaper.log" 2>&1 < /dev/null & disown
# Markers are printed only when a job's whole script ends (never by inner steps), and the log must not grow during the
# grace period. Each pass also deletes dead temporary files (> 2 h old) of interrupted writes in storage.
source configs/config.env || { echo "configs/config.env not found"; exit 1; }
MARK='ALL DONE|STOPPED|done at step|panel written|CONFIG OK'
while true; do
  for id in $(sky jobs queue --skip-finished -a 2>/dev/null | awk '$3 ~ /^(pwm-|dd-|vj2|dv3)/ && /RUNNING/ {print $1}'); do
    before=$(timeout 120 sky jobs logs "$id" --no-follow 2>/dev/null | tail -15)
    if echo "$before" | grep -Eq "$MARK"; then
      sleep 300
      after=$(timeout 120 sky jobs logs "$id" --no-follow 2>/dev/null | tail -15)
      if [[ "$after" == "$before" ]] && sky jobs queue --skip-finished -a 2>/dev/null | awk -v i="$id" '$1==i' | grep -q RUNNING; then
        echo "$(date -u +%F' '%H:%M) cancel $id: finished but still RUNNING"
        sky jobs cancel -y "$id" > /dev/null 2>&1
      fi
    fi
  done
  find "${storage_dir%/}/checkpoints" "${storage_dir%/}/models" "${storage_dir%/}/predictions" -name "*.tmp*" -mmin +120 \
       -type f -delete 2>/dev/null
  sleep 600
done

#!/usr/bin/env bash
# Wait (bounded) for a detached tmux job, then print a compact status.
# usage: bash tools/wait_stage.sh <tmux-session> [max_seconds=540] [logfile=logs/<session>.log]
# Keep max_seconds under the tool-call timeout (call with a 600000 ms timeout).
S="$1"; MAX="${2:-540}"; LOG="${3:-logs/$1.log}"; t=0
[ -z "$S" ] && { echo "usage: $0 <tmux-session> [max_seconds] [logfile]"; exit 1; }
while tmux has-session -t "$S" 2>/dev/null && [ "$t" -lt "$MAX" ]; do sleep 15; t=$((t + 15)); done
if tmux has-session -t "$S" 2>/dev/null; then echo "STATUS: RUNNING (waited ${t}s)"; else echo "STATUS: FINISHED (waited ${t}s)"; fi
echo "--- last 12 lines of $LOG ---"
tail -n 12 "$LOG" 2>/dev/null | cut -c1-300
ERR=$(grep -n -iE 'traceback|error|killed|memoryerror|out of memory|no space' "$LOG" 2>/dev/null | tail -n 5 | cut -c1-300)
[ -n "$ERR" ] && { echo "--- error lines ---"; echo "$ERR"; }
echo "--- mem avail: $(free -g | awk '/Mem/{print $7}') GB | disk free: $(df -h . | awk 'NR==2{print $4}')"

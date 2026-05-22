#!/usr/bin/env bash
# Watchdog wrapper for build_vlm_dataset.py.
#
# Runs the builder in --resume mode. If it crashes (non-zero exit), waits
# 30s and restarts — the builder's resume key (sample_token, instruction)
# means already-written records in vlm_dataset/records.jsonl are skipped, so
# no work is duplicated. Exits cleanly only when the builder finishes with
# exit code 0 (all target records processed).
#
# Usage:
#   cd /teamspace/studios/this_studio
#   nohup bash PGP_ego/run_vlm_builder_resumable.sh [--high_ade_only] \
#       > logs/vlm_builder_watchdog.log 2>&1 &
#
# Hard stop: kill the watchdog PID; otherwise it will restart the builder
# forever (until clean exit).

set -u
cd /teamspace/studios/this_studio

OUT_DIR="vlm_dataset"
EXTRA_ARGS=("$@")
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
BUILDER_LOG="$LOG_DIR/build_vlm_dataset.log"
WATCHDOG_PID=$$
RETRY_DELAY=30
MAX_CONSECUTIVE_FAILS=20   # bail if we crash >20 times in a row with no progress

prev_record_count=$(wc -l < "$OUT_DIR/records.jsonl" 2>/dev/null || echo 0)
consecutive_fails=0
attempt=0

echo "[watchdog $(date -Is)] starting; baseline records=$prev_record_count; extra_args=${EXTRA_ARGS[*]:-}"
echo "[watchdog $(date -Is)] builder log: $BUILDER_LOG"

while true; do
    attempt=$((attempt + 1))
    echo "[watchdog $(date -Is)] === attempt $attempt — launching builder ==="
    python3 -u PGP_ego/build_vlm_dataset.py --out_dir "$OUT_DIR" --resume \
        "${EXTRA_ARGS[@]}" >> "$BUILDER_LOG" 2>&1
    ec=$?
    now_records=$(wc -l < "$OUT_DIR/records.jsonl" 2>/dev/null || echo 0)
    delta=$((now_records - prev_record_count))
    echo "[watchdog $(date -Is)] builder exited code=$ec; records: $prev_record_count -> $now_records (+$delta)"

    if [ $ec -eq 0 ]; then
        echo "[watchdog $(date -Is)] clean exit — done."
        break
    fi

    if [ $delta -gt 0 ]; then
        consecutive_fails=0
    else
        consecutive_fails=$((consecutive_fails + 1))
        echo "[watchdog $(date -Is)] no new records this attempt; consecutive_fails=$consecutive_fails"
        if [ $consecutive_fails -ge $MAX_CONSECUTIVE_FAILS ]; then
            echo "[watchdog $(date -Is)] ABORT: $MAX_CONSECUTIVE_FAILS consecutive attempts with zero progress; giving up."
            exit 2
        fi
    fi
    prev_record_count=$now_records

    echo "[watchdog $(date -Is)] sleeping ${RETRY_DELAY}s before retry"
    sleep $RETRY_DELAY
done

echo "[watchdog $(date -Is)] final record count: $(wc -l < "$OUT_DIR/records.jsonl")"

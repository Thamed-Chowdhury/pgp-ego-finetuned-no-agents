#!/usr/bin/env bash
# Run the deterministic inference once for a given seed.
#
# Usage: ./run_single_seed.sh <SEED> [<OUT_SUBDIR>]
set -e

SEED="${1:?seed required}"
OUT_SUBDIR="${2:-seed_${SEED}}"

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"

OUT_DIR="$HERE/results/$OUT_SUBDIR"
mkdir -p "$OUT_DIR" "$HERE/logs"

# Strict-determinism env (CUBLAS workspace, PYTHONHASHSEED).
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED="$SEED"

LOG="$HERE/logs/seed_${SEED}.log"
echo "[run_single_seed] seed=$SEED  out_dir=$OUT_DIR  log=$LOG"

cd "$ROOT/PGP_ego"
python3 -u "$HERE/run_deterministic_inference.py" \
    --config         "$HERE/configs/pgp_ego_gatx2_lvm_ranked_text_stage3.yml" \
    --test_root      "$ROOT/nuscenes_data/v1-test" \
    --trainval_stats "$HERE/data/stats.pickle" \
    --test_preproc   "$HERE/data/test_preproc" \
    --checkpoint     "$HERE/checkpoints/stage3_best.tar" \
    --doscenes_repo  "$ROOT/pgp-ego-finetuned/doScenes_repo" \
    --text_emb_pkl   "$HERE/data/doscenes_gemini_embeddings.pkl" \
    --out_dir        "$OUT_DIR" \
    --seed           "$SEED" \
    --batch_size 4 2>&1 | tee "$LOG"

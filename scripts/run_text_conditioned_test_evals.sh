#!/usr/bin/env bash
# Eval the text-conditioned stage-3 checkpoint on the 3 doScenes test variants.
set +e
ROOT=/teamspace/studios/this_studio
CKPT=$ROOT/pgp_ego_output_dsvt_ranked_text_stage3/checkpoints/best.tar
TEST_ROOT=$ROOT/nuscenes_data/v1-test
DOSCENES_REPO=$ROOT/pgp-ego-finetuned/doScenes_repo
EMB_PKL=$ROOT/doscenes_gemini_embeddings.pkl
CFG=$ROOT/PGP_ego/configs/pgp_ego_gatx2_lvm_ranked_text_stage3.yml
LOG=$ROOT/logs/text_stage3_test_eval.log
mkdir -p $ROOT/logs
exec > >(tee -a "$LOG") 2>&1

echo "===== $(date) text-conditioned test evals ====="
echo "ckpt: $CKPT"
echo "emb : $EMB_PKL"

cd $ROOT/PGP_ego
for VARIANT in baseline dsvt dsvt_presence; do
    case $VARIANT in
        baseline)      PRE=$ROOT/pgp_ego_test_preprocessed ;;
        dsvt)          PRE=$ROOT/pgp_ego_test_preprocessed_dsvt ;;
        dsvt_presence) PRE=$ROOT/pgp_ego_test_preprocessed_dsvt_presence ;;
    esac
    OUT=$ROOT/test_text_stage3_${VARIANT}_results
    echo "===== eval variant=$VARIANT  pre=$PRE  out=$OUT"
    python3 -u run_doscenes_test_text_conditioned.py \
        --config         $CFG \
        --test_root      $TEST_ROOT \
        --trainval_stats $ROOT/pgp_ego_preprocessed/stats.pickle \
        --test_preproc   $PRE \
        --checkpoint     $CKPT \
        --doscenes_repo  $DOSCENES_REPO \
        --text_emb_pkl   $EMB_PKL \
        --out_dir        $OUT \
        --skip_preprocess \
        --batch_size 4 2>&1 | tail -30
done
echo "===== DONE $(date)"

#!/usr/bin/env bash
# After LiDAR parts 4-10 are extracted:
#   1) build PGP-filtered DSVT info pkl
#   2) run DSVT inference on the union (~9160 keyframes)
#   3) inject DSVT detections into a copy of pgp_ego_preprocessed
#   4) fine-tune from the existing best.tar
#   5) evaluate the fine-tuned model with the 3 test variants
#
# Logs to logs/finetune_dsvt.log
set -e -o pipefail
ROOT=/teamspace/studios/this_studio
LOG=$ROOT/logs/finetune_dsvt.log
mkdir -p $ROOT/logs
exec > >(tee -a "$LOG") 2>&1

echo "===== $(date) finetune-dsvt pipeline ====="

cd $ROOT

# 1) build filtered info pkl
echo "[1/5] build_pgp_train_info_pkl.py"
python3 -u PGP_ego/build_pgp_train_info_pkl.py

# 2) run DSVT inference on the PGP-needed keyframes
echo "[2/5] DSVT inference (~28 min on L4)"
cd $ROOT/DSVT/tools
PCDET_SKIP_GT_DB=1 python3 -u test.py \
    --cfg_file cfgs/dsvt_models/dsvt_plain_1f_onestage_nusences_pgp_train.yaml \
    --ckpt $ROOT/DSVT/checkpoints/DSVT_Nuscenes_val.pth \
    --batch_size 2 --workers 4 --extra_tag pgp_train 2>&1 | tail -n 80 || true
# The internal nuScenes eval will fail with "Samples in split doesn't match..." — that's expected.
DSVT_OUT=$(ls $ROOT/DSVT/output/dsvt_models/dsvt_plain_1f_onestage_nusences_pgp_train/pgp_train/eval/*/val/default/final_result/data/results_nusc.json 2>/dev/null | tail -1)
echo "DSVT results_nusc.json: $DSVT_OUT"
[ -f "$DSVT_OUT" ] || { echo "MISSING DSVT results"; exit 1; }
cd $ROOT

# 3) inject DSVT into a fresh copy of pgp_ego_preprocessed
echo "[3/5] inject_lidar_agents_trainval.py"
mkdir -p $ROOT/pgp_ego_preprocessed_dsvt
python3 -u PGP_ego/inject_lidar_agents_trainval.py \
    --results_nusc "$DSVT_OUT" \
    --in_dir  $ROOT/pgp_ego_preprocessed \
    --out_dir $ROOT/pgp_ego_preprocessed_dsvt

# 4) fine-tune from best.tar (5 epochs, --just-weights resets optim/sched/epoch)
echo "[4/5] fine-tune from pgp_ego_output2/checkpoints/best.tar"
cd $ROOT/PGP_ego
mkdir -p $ROOT/pgp_ego_output_dsvt
python3 -u train.py \
    -c configs/pgp_ego_gatx2_lvm_traversal_finetune.yml \
    -r $ROOT/nuscenes_data \
    -d $ROOT/pgp_ego_preprocessed_dsvt \
    -o $ROOT/pgp_ego_output_dsvt \
    -n 5 \
    -w $ROOT/pgp_ego_output2/checkpoints/best.tar \
    --just-weights

# 5) evaluate the new checkpoint on the test split for each agent-injection variant
echo "[5/5] evaluate fine-tuned model on test set"
NEW_CKPT=$ROOT/pgp_ego_output_dsvt/checkpoints/best.tar
TEST_ROOT=$ROOT/nuscenes_data/v1-test
DOSCENES_REPO=$ROOT/pgp-ego-finetuned/doScenes_repo
[ -d "$DOSCENES_REPO" ] || DOSCENES_REPO=$(find $ROOT -maxdepth 4 -type d -name doScenes_repo 2>/dev/null | head -1)
echo "doScenes_repo: $DOSCENES_REPO"

cd $ROOT/PGP_ego
for VARIANT in dsvt_presence dsvt baseline; do
    if [ "$VARIANT" = "baseline" ]; then PRE=$ROOT/pgp_ego_test_preprocessed
    elif [ "$VARIANT" = "dsvt" ]; then PRE=$ROOT/pgp_ego_test_preprocessed_dsvt
    else PRE=$ROOT/pgp_ego_test_preprocessed_dsvt_presence; fi
    OUT=$ROOT/test_finetuned_${VARIANT}_results
    echo "===== eval variant: $VARIANT  out=$OUT"
    python3 -u run_doscenes_test_baseline.py \
        --test_root      $TEST_ROOT \
        --trainval_stats $ROOT/pgp_ego_preprocessed/stats.pickle \
        --test_preproc   $PRE \
        --checkpoint     $NEW_CKPT \
        --doscenes_repo  $DOSCENES_REPO \
        --out_dir        $OUT \
        --skip_preprocess \
        --batch_size 4
done

echo "===== DONE  $(date)"

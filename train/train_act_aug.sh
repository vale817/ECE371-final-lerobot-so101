#!/usr/bin/env bash
# Train ACT augmentation ablations on AutoDL.
#
# Usage:
#   AUG=photo bash train/train_act_aug.sh
#   AUG=photo_geo bash train/train_act_aug.sh
#
# Resize is included in every transform set and max_num_transforms equals the
# number of transforms. This guarantees that both camera streams are always
# resized to 360x640 before ACT stacks their features.

set -euo pipefail

REPO_ID="${REPO_ID:-Valaris1028/task1_grab}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/lerobot/Valaris1028/task1_grab}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/train}"
AUG="${AUG:-photo}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"

case "${AUG}" in
  photo)
    JOB="${JOB:-act_task1_grab_aug_photo}"
    MAX_NUM_TRANSFORMS=2
    TFS='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}},"color":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.85,1.15],"contrast":[0.85,1.15],"saturation":[0.85,1.15],"hue":[-0.03,0.03]}}}'
    ;;
  photo_geo)
    JOB="${JOB:-act_task1_grab_aug_photo_geo}"
    MAX_NUM_TRANSFORMS=3
    TFS='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}},"color":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.85,1.15],"contrast":[0.85,1.15],"saturation":[0.85,1.15],"hue":[-0.03,0.03]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-3.0,3.0],"translate":[0.03,0.03],"scale":[0.97,1.03]}}}'
    ;;
  *)
    echo "ERROR: AUG must be 'photo' or 'photo_geo', got '${AUG}'" >&2
    exit 2
    ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --output_dir="${OUTPUT_ROOT}/${JOB}" \
  --job_name="${JOB}" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms="${MAX_NUM_TRANSFORMS}" \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs="${TFS}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --num_workers="${NUM_WORKERS}" \
  --wandb.enable=false

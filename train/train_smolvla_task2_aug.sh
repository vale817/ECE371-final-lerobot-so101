#!/bin/bash
# Train SmolVLA on task2 (pick red/blue block) with data augmentation.
# Runs two experiments sequentially: photo → photo_geo
#
# Usage:
#   bash train/train_smolvla_task2_aug.sh
#
# Or override params:
#   STEPS=20000 BATCH_SIZE=64 GPU=6 bash train/train_smolvla_task2_aug.sh

set -euo pipefail

# ── Configurable params ──
GPU="${GPU:-6}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# ── Fixed params ──
export HF_ENDPOINT=https://hf-mirror.com
DATASET_REPO="Valaris1028/task2_pick_color_merged"
DATASET_ROOT="$HOME/.cache/huggingface/lerobot/Valaris1028/task2_pick_color_merged"
RENAME_MAP='{"observation.images.handeye": "observation.images.camera1", "observation.images.fixed": "observation.images.camera2"}'

# ── Augmentation configs (conservative for color-classification task) ──
# Photo: Resize + ColorJitter (mild, with tighter hue for red/blue distinction)
PHOTO_MAX_TRANSFORMS=2
PHOTO_TFS='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}},"color":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1],"contrast":[0.9,1.1],"saturation":[0.85,1.15],"hue":[-0.02,0.02]}}}'

# Photo+Geo: Resize + ColorJitter + RandomAffine
PHOTO_GEO_MAX_TRANSFORMS=3
PHOTO_GEO_TFS='{"resize":{"weight":1.0,"type":"Resize","kwargs":{"size":[360,640]}},"color":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1],"contrast":[0.9,1.1],"saturation":[0.85,1.15],"hue":[-0.02,0.02]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-3.0,3.0],"translate":[0.03,0.03],"scale":[0.97,1.03]}}}'

# ── Helper functions ──

check_dataset() {
    if [ ! -d "${DATASET_ROOT}" ]; then
        echo "❌ Dataset not found at ${DATASET_ROOT}"
        echo "   Run: python3 scripts/merge_task2_datasets.py"
        exit 1
    fi
    echo "✅ Dataset found at ${DATASET_ROOT}"
}

check_gpu() {
    export CUDA_VISIBLE_DEVICES=${GPU}
    echo "GPU ${GPU} status:"
    nvidia-smi --id=${GPU} --query-gpu=name,memory.total,memory.free --format=csv,noheader
}

train_aug() {
    local aug_name="$1"
    local max_transforms="$2"
    local tfs="$3"
    local output_dir="/tmp/smolvla_task2_aug_${aug_name}"
    local job_name="smolvla_task2_aug_${aug_name}"
    local log_file="/tmp/smolvla_task2_aug_${aug_name}.log"
    local oss_dir="/oss-mt-sysu-release/zhangqi/Projects/Lerobot-so101/outputs/train/smolvla_task2_aug_${aug_name}"

    echo ""
    echo "========================================="
    echo "  SmolVLA Task2 Augmentation: ${aug_name}"
    echo "========================================="
    echo "  Dataset:    ${DATASET_REPO}"
    echo "  GPU:        ${GPU}"
    echo "  Steps:      ${STEPS}"
    echo "  Batch:      ${BATCH_SIZE}"
    echo "  Augment:    ${aug_name}"
    echo "  Output:     ${output_dir}"
    echo "========================================="

    # Clean previous output if exists
    if [ -d "${output_dir}" ]; then
        echo "⚠️  Removing previous output at ${output_dir}"
        rm -rf "${output_dir}"
    fi

    # ── Train ──
    export CUDA_VISIBLE_DEVICES=${GPU}

    lerobot-train \
      --dataset.repo_id=${DATASET_REPO} \
      --dataset.root=${DATASET_ROOT} \
      --dataset.video_backend=pyav \
      --policy.path=lerobot/smolvla_base \
      --rename_map="${RENAME_MAP}" \
      --output_dir=${output_dir} \
      --job_name=${job_name} \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --policy.freeze_vision_encoder=false \
      --policy.train_expert_only=false \
      --dataset.image_transforms.enable=true \
      --dataset.image_transforms.max_num_transforms=${max_transforms} \
      --dataset.image_transforms.random_order=false \
      --dataset.image_transforms.tfs="${tfs}" \
      --batch_size=${BATCH_SIZE} \
      --steps=${STEPS} \
      --save_freq=${SAVE_FREQ} \
      --log_freq=${LOG_FREQ} \
      --num_workers=${NUM_WORKERS} \
      --wandb.enable=false \
      2>&1 | tee ${log_file}

    # ── Post-training: copy to OSS ──
    echo ""
    echo "========================================="
    echo "  ${aug_name} training complete!"
    echo "  Copying to OSS..."
    echo "========================================="

    mkdir -p ${oss_dir}

    # Copy checkpoints
    if [ -d "${output_dir}/checkpoints" ]; then
        echo "  Copying checkpoints..."
        cp -r ${output_dir}/checkpoints ${oss_dir}/
    fi

    # Copy config and logs
    for f in train_config.json config.json; do
        if [ -f "${output_dir}/${f}" ]; then
            cp ${output_dir}/${f} ${oss_dir}/
        fi
    done

    if [ -f "${log_file}" ]; then
        cp ${log_file} ${oss_dir}/training.log
    fi

    echo "✅ ${aug_name} done! Output saved to ${oss_dir}"

    # Free up local disk
    echo "  Cleaning local output..."
    rm -rf ${output_dir}
}

# ── Main ──

echo "========================================="
echo "  SmolVLA Task2 Augmented Training"
echo "  Experiments: photo → photo_geo"
echo "========================================="

check_dataset
check_gpu

# ── Experiment 1: Photo augmentation ──
train_aug "photo" "${PHOTO_MAX_TRANSFORMS}" "${PHOTO_TFS}"

# ── Experiment 2: Photo+Geo augmentation ──
train_aug "photo_geo" "${PHOTO_GEO_MAX_TRANSFORMS}" "${PHOTO_GEO_TFS}"

echo ""
echo "========================================="
echo "  🎉 All augmentation experiments done!"
echo "========================================="
echo ""
echo "  Results:"
echo "    Baseline:   outputs/train/smolvla_task2_20k/"
echo "    Photo:      outputs/train/smolvla_task2_aug_photo/"
echo "    Photo+Geo:  outputs/train/smolvla_task2_aug_photo_geo/"
echo ""

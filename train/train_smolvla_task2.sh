#!/bin/bash
# Train SmolVLA on task2 (pick red/blue block) merged dataset
# 20k steps, batch_size=64, GPU 6

set -e

# ── Environment ──
export CUDA_VISIBLE_DEVICES=6
export HF_ENDPOINT=https://hf-mirror.com

# ── Dataset ──
DATASET_REPO="Valaris1028/task2_pick_color_merged"
DATASET_ROOT="$HOME/.cache/huggingface/lerobot/Valaris1028/task2_pick_color_merged"

# ── Training params ──
BATCH_SIZE=64
STEPS=20000
SAVE_FREQ=2000
LOG_FREQ=100
NUM_WORKERS=8

# ── Output ──
OUTPUT_DIR="/tmp/smolvla_task2_20k"
JOB_NAME="smolvla_task2_20k"
LOG_FILE="/tmp/smolvla_task2_20k.log"

echo "========================================="
echo "  SmolVLA Task2 Training"
echo "========================================="
echo "  Dataset:  ${DATASET_REPO}"
echo "  GPU:      ${CUDA_VISIBLE_DEVICES}"
echo "  Steps:    ${STEPS}"
echo "  Batch:    ${BATCH_SIZE}"
echo "  Output:   ${OUTPUT_DIR}"
echo "========================================="

# Check dataset exists
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "❌ Dataset not found at ${DATASET_ROOT}"
    echo "   Run: python3 scripts/merge_task2_datasets.py"
    exit 1
fi

# Check GPU
nvidia-smi --id=${CUDA_VISIBLE_DEVICES} --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Clean previous output if exists
if [ -d "${OUTPUT_DIR}" ]; then
    echo "⚠️  Removing previous output at ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
fi

# ── Train ──
lerobot-train \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.video_backend=pyav \
  --policy.path=lerobot/smolvla_base \
  --rename_map='{"observation.images.handeye": "observation.images.camera1", "observation.images.fixed": "observation.images.camera2"}' \
  --output_dir=${OUTPUT_DIR} \
  --job_name=${JOB_NAME} \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --batch_size=${BATCH_SIZE} \
  --steps=${STEPS} \
  --save_freq=${SAVE_FREQ} \
  --log_freq=${LOG_FREQ} \
  --num_workers=${NUM_WORKERS} \
  --wandb.enable=false \
  2>&1 | tee ${LOG_FILE}

# ── Post-training: copy to OSS ──
echo ""
echo "========================================="
echo "  Training complete! Copying to OSS..."
echo "========================================="

OSS_DIR="/oss-mt-sysu-release/zhangqi/Projects/Lerobot-so101/outputs/train/smolvla_task2_20k"
mkdir -p ${OSS_DIR}

# Copy checkpoints
if [ -d "${OUTPUT_DIR}/checkpoints" ]; then
    echo "  Copying checkpoints..."
    cp -r ${OUTPUT_DIR}/checkpoints ${OSS_DIR}/
fi

# Copy config and logs
for f in train_config.json config.json; do
    if [ -f "${OUTPUT_DIR}/${f}" ]; then
        cp ${OUTPUT_DIR}/${f} ${OSS_DIR}/
    fi
done

if [ -f "${LOG_FILE}" ]; then
    cp ${LOG_FILE} ${OSS_DIR}/training.log
fi

echo "✅ Done! Output saved to ${OSS_DIR}"

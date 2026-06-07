#!/usr/bin/env bash
# Train an ACT policy on task1_grab. Run this ON THE GPU SERVER (CUDA).
#
# Dataset: 60 episodes, 35989 frames, 30 fps, robot_type=so_follower,
#          2 cameras (observation.images.handeye, observation.images.fixed),
#          6-DOF action/state. LeRobotDataset v3.0.
#
# Prereqs on the server:
#   pip install -U "lerobot[act]"     # or: pip install -e ".[act]" from a lerobot checkout
#   hf auth login                     # only needed to PULL a private HF dataset / push the policy
#
# Option A (default): dataset is on the HF Hub -> lerobot pulls it automatically.
# Option B: copy the dataset to the server first (see RSYNC block below), no HF needed.

set -euo pipefail

HF_USER="Valaris1028"
REPO_ID="${HF_USER}/task1_grab"
JOB="act_task1_grab"

# ---- Option B: copy dataset from your Mac instead of pulling from HF ----------
# Run this FROM YOUR MAC (not on the server). Adjust user@host and path.
#   rsync -avhP ~/.cache/huggingface/lerobot/Valaris1028/task1_grab/ \
#       user@gpu-server:~/.cache/huggingface/lerobot/Valaris1028/task1_grab/
# Then on the server add --dataset.root to point at the local copy:
#   --dataset.root="${HOME}/.cache/huggingface/lerobot/${REPO_ID}"
# ------------------------------------------------------------------------------

lerobot-train \
  --dataset.repo_id="${REPO_ID}" \
  --policy.type=act \
  --output_dir="outputs/train/${JOB}" \
  --job_name="${JOB}" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=10000 \
  --log_freq=200 \
  --wandb.enable=false

# Notes:
# - ACT auto-detects both cameras from the dataset features; no manual camera config needed.
# - Default ACT chunk_size/n_action_steps=100 works well for SO-101 grasping.
#   For shorter horizons add: --policy.chunk_size=50 --policy.n_action_steps=50
# - To watch training curves, set --wandb.enable=true (run `wandb login` first).
# - Checkpoints land in outputs/train/act_task1_grab/checkpoints/ ; the last one
#   is what you point lerobot-record/eval at via --policy.path=...
# - To resume an interrupted run: add --resume=true (keep the same output_dir).

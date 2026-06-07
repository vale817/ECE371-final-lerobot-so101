#!/usr/bin/env bash
# Upload the merged task1_grab dataset to the HuggingFace Hub.
# Run this ONCE from your Mac (where the dataset lives). After it succeeds,
# the GPU server can pull the dataset automatically via --dataset.repo_id.
#
# Prereqs:
#   pip install -U "huggingface_hub[cli]"
#   hf auth login        # paste a WRITE token from https://huggingface.co/settings/tokens
#
# Usage: bash scripts/upload_to_hf.sh

set -euo pipefail

HF_USER="Valaris1028"
REPO_ID="${HF_USER}/task1_grab"
LOCAL_ROOT="${HOME}/.cache/huggingface/lerobot/${REPO_ID}"

if [ ! -d "${LOCAL_ROOT}" ]; then
  echo "ERROR: dataset folder not found: ${LOCAL_ROOT}"
  exit 1
fi

echo "Uploading ${LOCAL_ROOT}"
echo "      -> https://huggingface.co/datasets/${REPO_ID}"

# Create the dataset repo (no-op if it already exists), then upload the folder.
hf repo create "${REPO_ID}" --repo-type dataset -y || true
hf upload "${REPO_ID}" "${LOCAL_ROOT}" . --repo-type dataset

echo ""
echo "Done. Verify at: https://huggingface.co/datasets/${REPO_ID}"
echo "On the GPU server you can now train with --dataset.repo_id=${REPO_ID}"

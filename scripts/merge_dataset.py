# Merge multiple local LeRobot datasets into one using lerobot-edit-dataset.
# Usage: python3 merge_dataset.py
# Output: ~/.cache/huggingface/lerobot/Valaris1028/task1_grab

import os
import subprocess

HF_USER = "Valaris1028"
BASE = os.path.expanduser(f"~/.cache/huggingface/lerobot/{HF_USER}")

INPUT_DATASETS = ["1", "2", "3"]
INPUT_BASE = os.path.join(BASE, "grab_the_cube")

OUTPUT_REPO_ID = f"{HF_USER}/task1_grab"
OUTPUT_ROOT = os.path.join(BASE, "task1_grab")

repo_ids = [f"{HF_USER}/grab_the_cube_{d}" for d in INPUT_DATASETS]
roots = [os.path.join(INPUT_BASE, d) for d in INPUT_DATASETS]

repo_ids_str = "[" + ", ".join(repo_ids) + "]"
roots_str = "[" + ", ".join(roots) + "]"

cmd = [
    "lerobot-edit-dataset",
    "--operation.type=merge",
    f"--operation.repo_ids={repo_ids_str}",
    f"--operation.roots={roots_str}",
    f"--new_repo_id={OUTPUT_REPO_ID}",
    f"--new_root={OUTPUT_ROOT}",
    "--push_to_hub=false",
]

print("Running merge...")
print(f"  Input: {INPUT_BASE}/1, /2, /3")
print(f"  Output: {OUTPUT_ROOT}")
subprocess.run(cmd, check=True)
print(f"\nDone! Merged dataset saved to: {OUTPUT_ROOT}")

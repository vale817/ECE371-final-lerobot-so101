#!/usr/bin/env python3
"""
Merge task2_pick_red and task2_pick_blue datasets into one,
then modify task descriptions for each set of episodes.

Steps:
  1. Download both datasets from HuggingFace
  2. Merge into one dataset using lerobot-edit-dataset
  3. Modify task descriptions per episode group:
     - Red episodes (0-30): "pick up the red block and place it onto the red stack"
     - Blue episodes (31-61): "pick up the blue block and place it onto the blue stack"
  4. Verify the merged dataset

Usage:
  HF_ENDPOINT=https://hf-mirror.com python3 scripts/merge_task2_datasets.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HF_USER = "Valaris1028"
BASE = Path(os.path.expanduser(f"~/.cache/huggingface/lerobot/{HF_USER}"))

RED_REPO = f"{HF_USER}/task2_pick_red_merged_31"
BLUE_REPO = f"{HF_USER}/task2_pick_blue_merged_31"
MERGED_REPO = f"{HF_USER}/task2_pick_color_merged"

RED_TASK = "pick up the red block and place it onto the red stack"
BLUE_TASK = "pick up the blue block and place it onto the blue stack"

RED_EPISODES = 31  # episodes 0-30
BLUE_EPISODES = 31  # episodes 31-61 after merge


def run_cmd(cmd, desc=None):
    """Run a command and print output in real-time."""
    if desc:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n  ❌ FAILED (exit code {result.returncode})")
        sys.exit(1)
    return result


def verify_dataset(root):
    """Verify the merged dataset structure and task distribution."""
    print(f"\n{'='*60}")
    print(f"  Verification")
    print(f"{'='*60}")

    # Check info.json
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"  ❌ meta/info.json not found at {info_path}")
        return False

    with open(info_path) as f:
        info = json.load(f)

    print(f"  Total episodes: {info.get('total_episodes')}")
    print(f"  Total frames: {info.get('total_frames')}")
    print(f"  FPS: {info.get('fps')}")

    # Check tasks.parquet
    import pandas as pd
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        print(f"  ❌ meta/tasks.parquet not found")
        return False

    tasks_df = pd.read_parquet(tasks_path)
    print(f"\n  Tasks:")
    for task_name, row in tasks_df.iterrows():
        print(f"    task_index={row['task_index']}: \"{task_name}\"")

    # Verify we have exactly 2 tasks
    if len(tasks_df) != 2:
        print(f"  ❌ Expected 2 tasks, got {len(tasks_df)}")
        return False

    # Check that both expected tasks are present
    task_names = set(tasks_df.index)
    expected = {RED_TASK, BLUE_TASK}
    if task_names != expected:
        print(f"  ❌ Task mismatch!")
        print(f"    Expected: {expected}")
        print(f"    Got: {task_names}")
        return False

    # Check episode distribution
    episodes_dir = root / "meta" / "episodes"
    if episodes_dir.exists():
        ep_files = sorted(episodes_dir.glob("*.parquet"))
        total_eps = len(ep_files)
        red_count = 0
        blue_count = 0
        for ep_file in ep_files:
            ep_df = pd.read_parquet(ep_file)
            for _, row in ep_df.iterrows():
                tasks_list = row.get("tasks", [])
                if isinstance(tasks_list, list) and len(tasks_list) > 0:
                    if RED_TASK in tasks_list:
                        red_count += 1
                    elif BLUE_TASK in tasks_list:
                        blue_count += 1

        print(f"\n  Episode distribution:")
        print(f"    Red episodes: {red_count}")
        print(f"    Blue episodes: {blue_count}")
        print(f"    Total: {red_count + blue_count}")

        if red_count != RED_EPISODES or blue_count != BLUE_EPISODES:
            print(f"  ⚠️  Expected {RED_EPISODES} red + {BLUE_EPISODES} blue, got {red_count} + {blue_count}")
        else:
            print(f"  ✅ Episode counts match expected")

    print(f"\n  ✅ Dataset verification passed!")
    print(f"  Location: {root}")
    return True


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    merged_root = BASE / "task2_pick_color_merged"

    # ── Step 1: Download datasets (lerobot-edit-dataset will handle this) ──
    print(f"\n{'='*60}")
    print(f"  Step 1: Pre-download datasets")
    print(f"{'='*60}")

    for repo in [RED_REPO, BLUE_REPO]:
        local_path = BASE / repo.split("/")[-1]
        if local_path.exists():
            print(f"  ✅ {repo} already cached at {local_path}")
        else:
            print(f"  ⬇️  Downloading {repo}...")
            run_cmd(
                ["python3", "-c",
                 f"from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
                 f"LeRobotDataset('{repo}', video_backend='pyav')"],
                desc=f"Download {repo}"
            )

    # ── Step 2: Merge datasets ──
    print(f"\n{'='*60}")
    print(f"  Step 2: Merge datasets")
    print(f"{'='*60}")

    if merged_root.exists():
        print(f"  ⚠️  Merged dataset already exists at {merged_root}")
        print(f"  Removing existing merged dataset...")
        import shutil
        shutil.rmtree(merged_root)

    red_root = BASE / "task2_pick_red_merged_31"
    blue_root = BASE / "task2_pick_blue_merged_31"

    run_cmd([
        "lerobot-edit-dataset",
        "--operation.type=merge",
        f"--operation.repo_ids=[\"{RED_REPO}\", \"{BLUE_REPO}\"]",
        f"--operation.roots=[\"{red_root}\", \"{blue_root}\"]",
        f"--new_repo_id={MERGED_REPO}",
        f"--new_root={merged_root}",
        "--push_to_hub=false",
    ], desc="Merge red + blue datasets")

    # ── Step 3: Modify task descriptions ──
    print(f"\n{'='*60}")
    print(f"  Step 3: Modify task descriptions")
    print(f"{'='*60}")

    # Build episode_tasks mapping
    # After merge: red episodes are 0-30, blue episodes are 31-61
    episode_tasks = {}
    for i in range(RED_EPISODES):
        episode_tasks[str(i)] = RED_TASK
    for i in range(RED_EPISODES, RED_EPISODES + BLUE_EPISODES):
        episode_tasks[str(i)] = BLUE_TASK

    episode_tasks_json = json.dumps(episode_tasks)

    run_cmd([
        "lerobot-edit-dataset",
        f"--repo_id={MERGED_REPO}",
        f"--root={merged_root}",
        "--operation.type=modify_tasks",
        f"--operation.episode_tasks={episode_tasks_json}",
        "--push_to_hub=false",
    ], desc="Modify task descriptions per episode")

    # ── Step 4: Verify ──
    if not verify_dataset(merged_root):
        print("\n  ❌ Verification failed!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ✅ All done! Merged dataset ready at:")
    print(f"  {merged_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

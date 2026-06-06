# Check the number of episodes in each LeRobot dataset folder.
# Usage: python3 check_episodes.py <base_dir>
# Example: python3 check_episodes.py ~/.cache/huggingface/lerobot/Valaris1028/grab_the_cube

import pandas as pd
import os


def check_episodes(base_dir: str) -> None:
    base = os.path.expanduser(base_dir)
    if not os.path.exists(base):
        print(f"Directory not found: {base}")
        return

    for dataset in sorted(os.listdir(base)):
        parquet = os.path.join(base, dataset, "data/chunk-000/file-000.parquet")
        if os.path.exists(parquet):
            df = pd.read_parquet(parquet)
            n = df["episode_index"].nunique()
            print(f"{dataset}: {n} episodes")
        else:
            print(f"{dataset}: no data")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check number of episodes in each dataset.")
    parser.add_argument("base_dir", help="Base directory containing dataset folders")
    args = parser.parse_args()

    check_episodes(args.base_dir)

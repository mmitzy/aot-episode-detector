import os
import re
from pathlib import Path

import pandas as pd

# ---------- CONFIG ---------- #

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory where extracted frames are stored
FRAMES_ROOT = PROJECT_ROOT / "data" / "frames"

# Frames per second used during extraction
FPS = 2.0

# Where to save the index
INDEX_CSV = PROJECT_ROOT / "data" / "index.csv"

# ---------------------------- #

def main():
    print("DEBUG: build_index.py started")

    if not FRAMES_ROOT.exists():
        raise SystemExit(f"Frames root not found: {FRAMES_ROOT}")

    rows = []
    episode_dirs = [p for p in FRAMES_ROOT.iterdir() if p.is_dir()]
    episode_dirs = sorted(episode_dirs, key=lambda p: p.name)

    print(f"Found {len(episode_dirs)} episode folders under {FRAMES_ROOT}")

    frame_re = re.compile(r"frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

    for ep_dir in episode_dirs:
        episode_id = ep_dir.name  # e.g. S1E1, S4Esp1, SoadE2
        frame_files = sorted(ep_dir.glob("frame_*.*"))

        if not frame_files:
            print(f"WARNING: no frames in {ep_dir}")
            continue

        print(f"Processing {episode_id}: {len(frame_files)} frames")

        for fpath in frame_files:
            m = frame_re.match(fpath.name)
            if not m:
                # Skip weird files that don't match the pattern
                print(f"  Skipping non-frame file: {fpath.name}")
                continue

            frame_idx = int(m.group(1))  # e.g. "000123" -> 123
            # Time in seconds; since we started at frame_000001 after 0.5s, you can
            # either do (frame_idx - 1)/FPS or frame_idx/FPS; we'll choose (idx-1)
            time_sec = (frame_idx - 1) / FPS

            # Save path relative to project root (nicer for portability)
            rel_path = fpath.relative_to(PROJECT_ROOT)

            rows.append(
                {
                    "episode_id": episode_id,
                    "frame_path": str(rel_path).replace("\\", "/"),
                    "frame_idx": frame_idx,
                    "time_sec": time_sec,
                }
            )

    if not rows:
        raise SystemExit("No frames found, nothing to index.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["episode_id", "frame_idx"]).reset_index(drop=True)

    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_CSV, index=False)

    print(f"\nWrote index with {len(df)} rows to: {INDEX_CSV}")
    print("Sample:")
    print(df.head())


if __name__ == "__main__":
    main()

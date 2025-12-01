# A simple test for correct label mapping
from pathlib import Path
import traceback

from dataset import AOTEpisodeDataset, get_transforms


def main():
    print("=== Starting full-label sanity check ===")

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    train_csv = data_dir / "train_core.csv"
    frames_root = project_root

    print("Project root:", project_root)
    print("Frames root:", frames_root)
    print("Train CSV:", train_csv)

    # Use eval transform for determinism
    _, eval_transform = get_transforms(image_size=224)

    # Create dataset (build mapping automatically from train split)
    ds = AOTEpisodeDataset(
        csv_path=str(train_csv),
        frames_root=str(frames_root),
        episode_to_idx=None,
        transform=eval_transform,
    )

    print("\nDataset created OK")
    print("  Number of samples:", len(ds))

    # Mapping
    episode_to_idx = ds.episode_to_idx
    idx_to_episode = {v: k for k, v in episode_to_idx.items()}

    print("  Number of classes (episodes):", len(episode_to_idx))
    print("\nFull episode_id -> label mapping (in sort order):")
    for ep, idx in episode_to_idx.items():
        print(f"    {ep} -> {idx}")

    df = ds.df

    print("\nNow printing ONE sample per label/episode...")
    print("(This will open and transform ~1 image per episode, so give it a bit.)")

    # labels in sorted order
    sorted_labels = sorted(idx_to_episode.keys())

    for label in sorted_labels:
        ep_id = idx_to_episode[label]

        # Find first row for this episode in the dataframe
        rows_for_ep = df[df["episode_id"] == ep_id]
        if rows_for_ep.empty:
            print(f"\n[WARN] No rows found in CSV for episode_id={ep_id}, label={label}")
            continue

        row = rows_for_ep.iloc[0]
        idx_in_dataset = row.name  # dataframe index matches dataset index

        img, lbl, meta = ds[idx_in_dataset]

        print("\n--- Episode sample ---")
        print("  label:", lbl)
        print("  episode_id:", meta["episode_id"])
        print("  frame_idx:", meta["frame_idx"])
        print("  time_sec:", meta["time_sec"])
        print("  frame_path (relative):", meta["frame_path"])
        print("  absolute path:", frames_root / meta["frame_path"])
        print("  image tensor shape:", img.shape)

    print("\n=== Full-label sanity check finished successfully ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("=== Sanity check crashed with an exception ===")
        traceback.print_exc()

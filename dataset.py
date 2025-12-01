from pathlib import Path
from typing import Dict, Optional, Tuple
import re

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


def episode_sort_key(ep: str):
    """
    Produce a sortable key for episode IDs like:
      - 'S1E1', 'S1E2', ..., 'S3E12'
      - 'S4Esp1' (specials)
      - anything weird (e.g. 'SoadE2') goes to the end.

    This is ONLY for human-friendly label ordering.
    The model itself doesn't care about the actual numbers.
    """
    # Normal episodes: S{season}E{episode}
    m = re.match(r"^S(\d+)[Ee](\d+)$", ep)
    if m:
        season = int(m.group(1))
        episode = int(m.group(2))
        return (season, episode, 0, ep)  # 0 = normal episode

    # Specials: S{season}Esp{special_index}
    m = re.match(r"^S(\d+)Esp(\d+)$", ep)
    if m:
        season = int(m.group(1))
        special = int(m.group(2))
        # Push specials after normal episodes by adding a big offset
        return (season, 1000 + special, 1, ep)  # 1 = special

    # Fallback: weird IDs at the very end but deterministic
    return (9999, 9999, 2, ep)  # 2 = unknown pattern


class AOTEpisodeDataset(Dataset):
    """
    Dataset for AOT episode frames.

    CSV columns required:
        - episode_id : string (e.g. "S1E1", "S4Esp1", "SoadE2")
        - frame_path : string, relative to *project root*,
                       e.g. "data/frames/S1E1/frame_000542.jpg"
        - frame_idx  : int
        - time_sec   : float

    frames_root should be the project root, e.g.
        C:/Users/danib/Projects/aot-episode-detector

    so that:
        img_path = frames_root / frame_path
    points to the actual file.
    """

    def __init__(
        self,
        csv_path: str,
        frames_root: str,
        episode_to_idx: Optional[Dict[str, int]] = None,
        transform=None,
    ):
        # Normalize paths
        self.csv_path = Path(csv_path)
        self.frames_root = Path(frames_root)

        # Load CSV
        self.df = pd.read_csv(self.csv_path)

        # Sanity check required columns
        required_cols = {"episode_id", "frame_path", "frame_idx", "time_sec"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV {csv_path} is missing required columns: {missing}")

        # Build or accept mapping episode_id -> label index
        if episode_to_idx is None:
            unique_episodes = sorted(
                self.df["episode_id"].unique(),
                key=episode_sort_key,
            )
            self.episode_to_idx: Dict[str, int] = {
                ep: i for i, ep in enumerate(unique_episodes)
            }
        else:
            self.episode_to_idx = episode_to_idx

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # frame_path is like "data/frames/S1E1/frame_000894.jpg"
        frame_rel = row["frame_path"]
        img_path = self.frames_root / frame_rel

        if not img_path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {img_path}\n"
                f"(frames_root={self.frames_root}, frame_path={frame_rel})"
            )

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        episode_id = row["episode_id"]
        label = self.episode_to_idx[episode_id]

        meta = {
            "episode_id": episode_id,
            "frame_idx": int(row["frame_idx"]),
            "time_sec": float(row["time_sec"]),
            "frame_path": frame_rel,
        }

        return img, label, meta


def get_transforms(image_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    """
    Basic ImageNet-style transforms.
    """
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str],
    frames_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
):
    """
    Create train/val/test DataLoaders sharing the same episode_to_idx mapping.

    frames_root must be the project root (parent of "data").
    """
    frames_root_path = str(Path(frames_root))
    train_transform, eval_transform = get_transforms(image_size=image_size)

    # Build mapping from train split using the same episode_sort_key
    train_df = pd.read_csv(train_csv)
    unique_episodes = sorted(
        train_df["episode_id"].unique(),
        key=episode_sort_key,
    )
    episode_to_idx: Dict[str, int] = {ep: i for i, ep in enumerate(unique_episodes)}
    num_classes = len(episode_to_idx)

    train_ds = AOTEpisodeDataset(
        csv_path=train_csv,
        frames_root=frames_root_path,
        episode_to_idx=episode_to_idx,
        transform=train_transform,
    )
    val_ds = AOTEpisodeDataset(
        csv_path=val_csv,
        frames_root=frames_root_path,
        episode_to_idx=episode_to_idx,
        transform=eval_transform,
    )

    test_ds = None
    if test_csv is not None:
        test_ds = AOTEpisodeDataset(
            csv_path=test_csv,
            frames_root=frames_root_path,
            episode_to_idx=episode_to_idx,
            transform=eval_transform,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader, episode_to_idx, num_classes

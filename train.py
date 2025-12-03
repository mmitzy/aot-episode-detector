# train.py
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm  # progress bars

from dataset import create_dataloaders
from model import AOTEpisodeNet


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm progress bar over batches
    loop = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", ncols=100)

    for images, labels, metas in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, embeddings = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        # update bar postfix (live)
        loop.set_postfix({
            "loss": f"{running_loss / total:.4f}",
            "acc": f"{(correct / total) * 100:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100.0
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, epoch, num_epochs, split_name="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} [{split_name}]", ncols=100)

    with torch.no_grad():
        for images, labels, metas in loop:
            images = images.to(device)
            labels = labels.to(device)

            logits, embeddings = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            loop.set_postfix({
                "loss": f"{running_loss / total:.4f}",
                "acc": f"{(correct / total) * 100:.2f}%"
            })

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100.0
    return epoch_loss, epoch_acc


def main():
    # ===== Paths =====
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"

    train_csv = data_dir / "train_core.csv"
    val_csv   = data_dir / "val_core.csv"
    test_csv  = data_dir / "test_core.csv"

    frames_root = project_root

    # ===== Hyperparameters =====
    num_epochs = 5       # you can tweak this
    batch_size = 32      # start with 32; drop to 16 if it's too slow
    lr = 1e-3
    weight_decay = 1e-4

    # ===== Device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== DataLoaders =====
    train_loader, val_loader, test_loader, episode_to_idx, num_classes = create_dataloaders(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        test_csv=str(test_csv),
        frames_root=str(frames_root),
        batch_size=batch_size,
        num_workers=0,   # keep 0 on Windows for now
        image_size=224,
    )

    print(f"Number of episodes (classes): {num_classes}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch: {len(val_loader)}")

    # ===== Model =====
    model = AOTEpisodeNet(
        num_classes=num_classes,
        embed_dim=128,
        backbone="resnet18",
        pretrained=True,
    ).to(device)

    # ===== Loss & Optimizer & Scheduler =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # ===== Training loop =====
    best_val_acc = 0.0
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")

        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, epoch, num_epochs, split_name="Val"
        )

        scheduler.step()
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch} done in {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "episode_to_idx": episode_to_idx,
                },
                ckpt_path,
            )
            print(f"  >>> New best model saved with val_acc={val_acc:.2f}%")

    print("\nTraining finished.")
    print(f"Best val accuracy: {best_val_acc:.2f}%")

    # ===== Optional: final test evaluation =====
    if test_loader is not None:
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, epoch=num_epochs, num_epochs=num_epochs, split_name="Test"
        )
        print(f"\nTest Loss: {test_loss:.4f}  |  Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()

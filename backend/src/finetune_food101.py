import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision.transforms import v2
from transformers import AutoImageProcessor, AutoModel
from dotenv import load_dotenv
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", verbose=True)
MODEL_ID = os.getenv("DINO_MODEL_ID", "facebook/dinov3-vits16-pretrain-lvd1689m")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

NUM_CLASSES = 101
BATCH_SIZE = 64
EPOCHS = 10
LR_HEAD = 1e-3
LR_BACKBONE = 1e-5  # much smaller — fine-tunes only the last few transformer blocks
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading processor and deriving transforms...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=ACCESS_TOKEN)

# Derive resize / crop sizes from the processor so they stay in sync with the model
_size = processor.size
IMAGE_SIZE = _size.get("shortest_edge", _size.get("height", 224))
_crop = processor.crop_size or {}
CROP_SIZE = _crop.get("height", IMAGE_SIZE)

MEAN = processor.image_mean
STD = processor.image_std

train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop(CROP_SIZE, scale=(0.6, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ]
)

val_transforms = v2.Compose(
    [
        v2.Resize(IMAGE_SIZE),
        v2.CenterCrop(CROP_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ]
)

train_dataset = Food101(
    root=DATA_DIR, split="train", transform=train_transforms, download=True
)
val_dataset = Food101(
    root=DATA_DIR, split="test", transform=val_transforms, download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# ── Model ─────────────────────────────────────────────────────────────────────
print("Loading model...")
backbone = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
hidden_size = backbone.config.hidden_size

# Freeze everything, then unfreeze the last 2 transformer blocks + layernorm
for param in backbone.parameters():
    param.requires_grad = False

encoder_layers = backbone.encoder.layer
for layer in encoder_layers[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

for param in backbone.layernorm.parameters():
    param.requires_grad = True

classifier_head = nn.Linear(hidden_size, NUM_CLASSES).to(DEVICE)

# ── Optimiser ─────────────────────────────────────────────────────────────────
backbone_params = [p for p in backbone.parameters() if p.requires_grad]
head_params = list(classifier_head.parameters())

optimiser = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ],
    weight_decay=WEIGHT_DECAY,
)

total_steps = EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_cls_token(images: torch.Tensor) -> torch.Tensor:
    outputs = backbone(pixel_values=images)
    return outputs.last_hidden_state[:, 0, :]


def run_epoch(loader: DataLoader, train: bool, epoch: int) -> tuple[float, float]:
    backbone.train(train)
    classifier_head.train(train)

    phase = "train" if train else "val"
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    bar = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}/{EPOCHS} [{phase}]",
        leave=False,
        unit="batch",
        dynamic_ncols=True,
    )

    with ctx:
        for images, labels in bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            cls = get_cls_token(images)
            logits = classifier_head(cls)
            loss = criterion(logits, labels)

            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(backbone_params) + head_params, 1.0)
                optimiser.step()
                scheduler.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

            bar.set_postfix(
                loss=f"{total_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
                lr=f"{optimiser.param_groups[1]['lr']:.2e}",
            )

    bar.close()
    return total_loss / total, correct / total


# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0

print(f"Device: {DEVICE}  |  Model: {MODEL_ID}")
print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
print(
    f"Trainable params: {sum(p.numel() for p in backbone_params) + sum(p.numel() for p in head_params):,}\n"
)

epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Epochs", unit="epoch", dynamic_ncols=True)

for epoch in epoch_bar:
    train_loss, train_acc = run_epoch(train_loader, train=True, epoch=epoch)
    val_loss, val_acc = run_epoch(val_loader, train=False, epoch=epoch)

    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        torch.save(
            {
                "epoch": epoch,
                "backbone_state": backbone.state_dict(),
                "head_state": classifier_head.state_dict(),
                "val_acc": val_acc,
            },
            CHECKPOINT_DIR / "dinov3_food101_best.pt",
        )

    epoch_bar.write(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train loss {train_loss:.4f}  acc {train_acc:.4f} | "
        f"val loss {val_loss:.4f}  acc {val_acc:.4f}" + (" ✓ saved" if improved else "")
    )

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
print(f"Checkpoint: {CHECKPOINT_DIR / 'dinov3_food101_best.pt'}")

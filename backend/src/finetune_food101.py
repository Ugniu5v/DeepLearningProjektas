import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", verbose=True)
MODEL_ID = os.getenv("DINO_MODEL_ID", "facebook/dinov3-vits16-pretrain-lvd1689m")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

NUM_CLASSES = 101
BATCH_SIZE = 256  # total batch split across all GPUs by DataParallel
EPOCHS = 15
LR_HEAD = 3e-3
LR_BACKBONE = 3e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
UNFREEZE_BLOCKS = 6

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_state_dict(model: nn.Module) -> dict:
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def get_cls_token(images: torch.Tensor) -> torch.Tensor:
    outputs = backbone(pixel_values=images)
    return outputs.last_hidden_state[:, 0, :]


def run_epoch(loader: DataLoader, train: bool, epoch: int) -> tuple[float, float]:
    backbone.train(train)
    classifier_head.train(train)

    phase = "mokymas" if train else "validacija"
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    bar = tqdm(
        loader,
        desc=f"Epocha {epoch:02d}/{EPOCHS} [{phase}]",
        leave=False,
        unit="partija",
        dynamic_ncols=True,
    )

    with ctx:
        for images, labels in bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.amp.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
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
                nuostolis=f"{total_loss / total:.4f}",
                tikslumas=f"{correct / total:.4f}",
                lr=f"{optimiser.param_groups[1]['lr']:.2e}",
            )

    bar.close()
    return total_loss / total, correct / total


def save_plots(history: dict, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DINOv3 – Food-101 treniravimo eiga", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Mokymas", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Validacija", markersize=4)
    axes[0].set_title("Nuostolis pagal epochą")
    axes[0].set_xlabel("Epocha")
    axes[0].set_ylabel("Nuostolis")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs,
        [a * 100 for a in history["train_acc"]],
        "b-o",
        label="Mokymas",
        markersize=4,
    )
    axes[1].plot(
        epochs,
        [a * 100 for a in history["val_acc"]],
        "r-o",
        label="Validacija",
        markersize=4,
    )
    axes[1].set_title("Tikslumas pagal epochą")
    axes[1].set_xlabel("Epocha")
    axes[1].set_ylabel("Tikslumas (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = save_dir / "treniravimo_eiga.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Grafikai išsaugoti: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    n_gpus = torch.cuda.device_count()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Kraunamas procesorius ir transformacijos...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=ACCESS_TOKEN)

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

    train_dataset = Food101(root=DATA_DIR, split="train", transform=train_transforms, download=True)
    val_dataset = Food101(root=DATA_DIR, split="test", transform=val_transforms, download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Kraunamas modelis...")
    backbone = AutoModel.from_pretrained(MODEL_ID, token=ACCESS_TOKEN).to(DEVICE)
    hidden_size = backbone.config.hidden_size

    for param in backbone.parameters():
        param.requires_grad = False
    for layer in backbone.model.layer[-UNFREEZE_BLOCKS:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in backbone.norm.parameters():
        param.requires_grad = True

    backbone_params = [p for p in backbone.parameters() if p.requires_grad]

    if n_gpus > 1:
        print(f"DataParallel: naudojama {n_gpus} GPU")
        backbone = nn.DataParallel(backbone)

    classifier_head = nn.Linear(hidden_size, NUM_CLASSES).to(DEVICE)
    head_params = list(classifier_head.parameters())

    # ── Optimiser ─────────────────────────────────────────────────────────────
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

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history: dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"Įrenginys: {DEVICE}  |  GPU skaičius: {n_gpus}  |  Modelis: {MODEL_ID}")
    print(f"Mokymo partijos: {len(train_loader)}  |  Validacijos partijos: {len(val_loader)}")
    print(
        f"Treniruojami parametrai: "
        f"{sum(p.numel() for p in backbone_params) + sum(p.numel() for p in head_params):,}\n"
    )

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Epochos", unit="epocha", dynamic_ncols=True)

    for epoch in epoch_bar:
        train_loss, train_acc = run_epoch(train_loader, train=True, epoch=epoch)
        val_loss, val_acc = run_epoch(val_loader, train=False, epoch=epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": get_state_dict(backbone),
                    "head_state": get_state_dict(classifier_head),
                    "val_acc": val_acc,
                },
                CHECKPOINT_DIR / "dinov3_food101_best.pt",
            )

        epoch_bar.write(
            f"Epocha {epoch:02d}/{EPOCHS} | "
            f"mokymo nuostolis {train_loss:.4f}  tikslumas {train_acc:.4f} | "
            f"validacijos nuostolis {val_loss:.4f}  tikslumas {val_acc:.4f}"
            + (" ✓ išsaugota" if improved else "")
        )

        save_plots(history, PLOTS_DIR)

    print(f"\nTreniravimas baigtas. Geriausias validacijos tikslumas: {best_val_acc:.4f}")
    print(f"Patikros taškas: {CHECKPOINT_DIR / 'dinov3_food101_best.pt'}")
    print(f"Grafikai: {PLOTS_DIR / 'treniravimo_eiga.png'}")

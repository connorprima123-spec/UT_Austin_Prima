"""
Usage:
    python3 -m homework.train_planner --model_name mlp_planner
    python3 -m homework.train_planner --model_name transformer_planner
    python3 -m homework.train_planner --model_name cnn_planner --num_epoch 60

Or from a notebook:
    from homework.train_planner import train
    train(model_name="mlp_planner", transform_pipeline="state_only", lr=1e-3, batch_size=128, num_epoch=50)
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import load_model, save_model

# which batch keys each model is allowed to see
MODEL_INPUTS = {
    "mlp_planner": ["track_left", "track_right"],
    "transformer_planner": ["track_left", "track_right"],
    "cnn_planner": ["image"],
}
# default pipeline per model; state-only models skip image loading (much faster)
MODEL_PIPELINE = {
    "mlp_planner": "state_only",
    "transformer_planner": "state_only",
    "cnn_planner": "default",
}
# dataset folders to look for (relative to the working directory) when no path is given
DATA_ROOT_CANDIDATES = ["drive_data", "road_data"]


def find_data_path(split: str) -> str:
    for root in DATA_ROOT_CANDIDATES:
        path = Path(root) / split
        if path.is_dir():
            return str(path)
    raise FileNotFoundError(
        f"Could not find a '{split}' folder under any of {DATA_ROOT_CANDIDATES}. "
        "Pass train_path / val_path explicitly."
    )


def masked_l1(preds, labels, mask):
    err = (preds - labels).abs() * mask[..., None]
    return err.sum() / (mask.sum() * 2).clamp(min=1)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    model_name: str = "mlp_planner",
    transform_pipeline: str | None = None,
    train_path: str | None = None,
    val_path: str | None = None,
    num_epoch: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = 2024,
    log_dir: str = "logs",
):
    device = get_device()
    print(f"Using {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_path = train_path or find_data_path("train")
    val_path = val_path or find_data_path("val")
    pipeline = transform_pipeline or MODEL_PIPELINE[model_name]
    keys = MODEL_INPUTS[model_name]

    model = load_model(model_name).to(device)

    train_data = load_data(
        train_path, transform_pipeline=pipeline, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    val_data = load_data(
        val_path, transform_pipeline=pipeline, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
    metric = PlannerMetric()

    # optional tensorboard logging (view with: %tensorboard --logdir logs)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        run_name = f"{model_name}_lr{lr}_{datetime.now().strftime('%m%d_%H%M%S')}"
        writer = SummaryWriter(Path(log_dir) / run_name)
    except ImportError:
        pass

    best_l1 = float("inf")

    for epoch in range(num_epoch):
        # ---- train ----
        model.train()
        losses = []
        for batch in train_data:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**{k: batch[k] for k in keys})
            loss = masked_l1(preds, batch["waypoints"], batch["waypoints_mask"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        # ---- validate ----
        model.eval()
        metric.reset()
        with torch.inference_mode():
            for batch in val_data:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = model(**{k: batch[k] for k in keys})
                metric.add(preds, batch["waypoints"], batch["waypoints_mask"])
        m = metric.compute()

        train_loss = float(np.mean(losses))
        print(
            f"Epoch {epoch + 1:3d}/{num_epoch} | train_loss {train_loss:.4f} | "
            f"val l1 {m['l1_error']:.4f} (long {m['longitudinal_error']:.4f}, lat {m['lateral_error']:.4f})"
        )

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/l1_error", m["l1_error"], epoch)
            writer.add_scalar("val/longitudinal_error", m["longitudinal_error"], epoch)
            writer.add_scalar("val/lateral_error", m["lateral_error"], epoch)

        # keep the best checkpoint
        if m["l1_error"] < best_l1:
            best_l1 = m["l1_error"]
            model.cpu()
            path = save_model(model)
            model.to(device)
            print(f"  -> saved best model to {path}")

    if writer is not None:
        writer.close()

    print(f"Best val l1 error: {best_l1:.4f}")
    return best_l1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_INPUTS))
    parser.add_argument("--transform_pipeline", type=str, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--log_dir", type=str, default="logs")

    train(**vars(parser.parse_args()))

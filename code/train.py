"""
train.py

Training script for the stochastic depth re-implementation.

Usage
-----
# Stochastic depth (reproduces Table 1, CIFAR-10+ column):
python code/train.py --dataset cifar10 --p_L 0.5

# Constant-depth baseline (p_L=1.0 means all blocks always active):
python code/train.py --dataset cifar10 --p_L 1.0

# CIFAR-100:
python code/train.py --dataset cifar100 --p_L 0.5

Key differences from the authors' main.lua
-------------------------------------------
- No separate gate-open/gate-close loop before each mini-batch; the drop
  decision lives entirely inside StochasticBlock.forward().
- Learning-rate schedule uses PyTorch's MultiStepLR instead of manual
  epoch-fraction comparisons.
- Progress and results are logged to a JSON file (one dict per epoch) rather
  than a Torch-serialised table.
- Model checkpointing saves the full training state so a run can be resumed.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from model import cifar_resnet110
from data  import build_loaders


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stochastic Depth – CIFAR training")
    p.add_argument("--dataset",     default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--p_L",         type=float, default=0.5,
                   help="Survival probability of the last block (1.0 = constant depth)")
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=0.1)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--momentum",    type=float, default=0.9)
    p.add_argument("--data_root",   default="./data")
    p.add_argument("--out_dir",     default="./results")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--resume",         default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--max_batches",    type=int, default=None,
                   help="Cap batches per epoch for a quick smoke test (e.g. 20).")
    p.add_argument("--survival_mode",  default="linear",
                   choices=["linear", "uniform", "constant"],
                   help="Survival probability schedule (linear=Eq.4, uniform=fixed p_L).")
    p.add_argument("--blocks_per_group", type=int, default=18,
                   help="Residual blocks per group. depth=6n+2: n=18->110L, n=9->56L etc.")
    p.add_argument("--log_grad",          action="store_true",
                   help="Log mean gradient magnitude of first conv layer each epoch (for Fig 7).")
    p.add_argument("--test_every_epoch",  action="store_true",
                   help="Evaluate test set every epoch for smooth Fig 3 curves. "
                        "Omit for faster training (test eval only on val improvement).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    """Return (error_rate_pct, avg_cross_entropy) over loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item()
        correct    += logits.argmax(dim=1).eq(labels).sum().item()
        n          += labels.size(0)
    error = 100.0 * (1.0 - correct / n)
    return error, total_loss / n


def save_checkpoint(path: str, epoch: int, model, optimizer, scheduler,
                    best_val_err: float, log: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":        epoch,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "best_val_err": best_val_err,
        "log":          log,
    }, path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")
    print(f"dataset: {args.dataset}  p_L={args.p_L}  survival_mode={args.survival_mode}  "
          f"blocks_per_group={args.blocks_per_group}  epochs={args.epochs}")
    if args.max_batches is not None:
        print(f"*** SMOKE TEST: capped at {args.max_batches} batches/epoch ***")

    # ── Data ─────────────────────────────────────────────────────────────────
    num_classes = 10 if args.dataset == "cifar10" else 100
    train_loader, val_loader, test_loader = build_loaders(
        dataset_name=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    from model import cifar_resnet_custom
    model = cifar_resnet_custom(
        num_classes=num_classes,
        blocks_per_group=args.blocks_per_group,
        p_L=args.p_L,
        survival_mode=args.survival_mode,
    ).to(device)
    print(f"parameters: {model.count_parameters():,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Hyperparameters from the paper (Section 4): SGD, momentum=0.9,
    # nesterov, weight_decay=1e-4, lr drops by ×10 at epochs 250 and 375.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[250, 375],
        gamma=0.1,
    )
    criterion = nn.CrossEntropyLoss()

    # ── Output paths ──────────────────────────────────────────────────────────
    tag       = f"{args.dataset}_pL{args.p_L}_mode{args.survival_mode}_n{args.blocks_per_group}"
    ckpt_last = os.path.join(args.out_dir, "checkpoints", f"{tag}_last.pt")
    ckpt_best = os.path.join(args.out_dir, "checkpoints", f"{tag}_best.pt")
    log_path  = os.path.join(args.out_dir, "logs",        f"{tag}.json")
    os.makedirs(os.path.dirname(ckpt_last), exist_ok=True)
    os.makedirs(os.path.dirname(log_path),  exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_err  = float("inf")
    best_test_err = float("inf")
    epoch_log: list[dict] = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_err = ckpt["best_val_err"]
        epoch_log    = ckpt.get("log", [])
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n{'epoch':>6}  {'train_loss':>10}  {'val_err%':>8}  "
          f"{'test_err%':>9}  {'lr':>8}  {'secs':>6}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss, n_seen = 0.0, 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            # Log first conv gradient magnitude if requested (for Fig 7 reproduction)
            if args.log_grad and n_seen == 0:  # once per epoch, first batch
                grad_mag = model.stem[0].weight.grad.abs().mean().item()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n_seen       += labels.size(0)

        scheduler.step()

        train_loss = running_loss / n_seen
        val_err, _ = evaluate(model, val_loader, device)
        elapsed    = time.perf_counter() - t0

        if args.test_every_epoch:
            # Per-epoch test evaluation — smooth curves matching paper Figure 3.
            test_err, _ = evaluate(model, test_loader, device)
            if val_err < best_val_err:
                best_val_err  = val_err
                best_test_err = test_err
                save_checkpoint(ckpt_best, epoch, model, optimizer, scheduler,
                                best_val_err, epoch_log)
        else:
            # Evaluate on test only when val improves (saves time).
            # test_err in the log is the running best — staircase shape,
            # but final values are identical.
            if val_err < best_val_err:
                best_val_err  = val_err
                test_err, _   = evaluate(model, test_loader, device)
                best_test_err = test_err
                save_checkpoint(ckpt_best, epoch, model, optimizer, scheduler,
                                best_val_err, epoch_log)

        current_lr = scheduler.get_last_lr()[0]
        record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "val_err":    round(val_err, 3),
            "test_err":   round(test_err if args.test_every_epoch else best_test_err, 3),
            "lr":         current_lr,
            "secs":       round(elapsed, 1),
        }
        if args.log_grad:
            record["grad_mag"] = round(grad_mag, 10)
        epoch_log.append(record)

        # Overwrite log every epoch so partial runs are inspectable
        with open(log_path, "w") as f:
            json.dump(epoch_log, f, indent=2)

        if epoch % 25 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>10.4f}  {val_err:>8.2f}  "
                  f"{best_test_err:>9.2f}  {current_lr:>8.5f}  {elapsed:>6.1f}")

        save_checkpoint(ckpt_last, epoch, model, optimizer, scheduler,
                        best_val_err, epoch_log)

    print(f"\nDone. Best val error: {best_val_err:.2f}%  "
          f"Test error at best val: {best_test_err:.2f}%")
    print(f"Best checkpoint : {ckpt_best}")
    print(f"Training log    : {log_path}")


if __name__ == "__main__":
    main()

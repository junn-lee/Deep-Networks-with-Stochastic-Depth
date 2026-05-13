"""
evaluate.py

Load a saved checkpoint and report test error.

Usage:
    python code/evaluate.py \
        --checkpoint results/checkpoints/cifar10_pL0.5_best.pt \
        --dataset cifar10 --p_L 0.5
"""

import argparse
import torch

from model import cifar_resnet110
from data  import build_loaders
from train import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset",    default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--p_L",        type=float, default=0.5)
    p.add_argument("--data_root",  default="./data")
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--num_workers",type=int,   default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 10 if args.dataset == "cifar10" else 100
    _, _, test_loader = build_loaders(
        dataset_name=args.dataset, root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = cifar_resnet110(num_classes=num_classes, p_L=args.p_L).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(best val err: {ckpt['best_val_err']:.2f}%)")

    err, loss = evaluate(model, test_loader, device)
    print(f"Test error : {err:.2f}%")
    print(f"Test loss  : {loss:.4f}")


if __name__ == "__main__":
    main()

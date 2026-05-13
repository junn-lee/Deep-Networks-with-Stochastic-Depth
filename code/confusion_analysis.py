"""
confusion_analysis.py

Compares per-class and pairwise confusion between the constant-depth baseline
and stochastic-depth model on the CIFAR-10 test set.

Usage:
    python confusion_analysis.py \
        --baseline  results/fig3/baseline/checkpoints/cifar10_pL1.0_modeconstant_n18_best.pt \
        --stochastic results/fig3/stochastic/checkpoints/cifar10_pL0.5_modelinear_n18_best.pt \
        --data_root ./data \
        --out_dir   ./confusion_results

Outputs (all saved to --out_dir):
    confusion_baseline.png       raw confusion matrix heatmap
    confusion_stochastic.png     raw confusion matrix heatmap
    confusion_diff.png           (stochastic - baseline) difference heatmap
    per_class_error.png          per-class error rate bar chart
    summary.txt                  printed summary table
"""

import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix

from model import cifar_resnet110
from data  import build_loaders


CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Pairs known to be visually similar
INTERESTING_PAIRS = [
    ('cat',        'dog'),
    ('automobile', 'truck'),
    ('deer',       'horse'),
    ('airplane',   'bird'),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline',    required=True, help='Path to baseline best checkpoint')
    p.add_argument('--stochastic',  required=True, help='Path to stochastic best checkpoint')
    p.add_argument('--data_root',   default='./data')
    p.add_argument('--batch_size',  type=int, default=128)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--out_dir',     default='./confusion_results')
    return p.parse_args()


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, device):
    """Return (all_preds, all_labels) as numpy arrays."""
    model.eval()
    preds, labels = [], []
    for imgs, lbs in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(lbs.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def load_model(ckpt_path, p_L, device):
    model = cifar_resnet110(num_classes=10, p_L=p_L).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded {ckpt_path}  (epoch {ckpt["epoch"]}, '
          f'best val err {ckpt["best_val_err"]:.2f}%)')
    return model


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_confusion(cm, title, path, normalize=True):
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        fmt = '.1f'
        cbar_label = 'row-normalized (%)'
    else:
        cm_plot = cm
        fmt = 'd'
        cbar_label = 'count'

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_plot, cmap='Blues')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=12)

    thresh = cm_plot.max() / 2.0
    for i in range(10):
        for j in range(10):
            val = cm_plot[i, j]
            txt = f'{val:{fmt}}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    color='white' if val > thresh else 'black')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_diff(cm_stoch, cm_base, path):
    """
    Difference heatmap: stochastic_error_rate - baseline_error_rate (per cell).
    Negative = stochastic makes FEWER errors (good, shown in blue).
    Positive = stochastic makes MORE errors (bad, shown in red).
    Diagonal is zeroed out (correct predictions don't count as errors).
    """
    def off_diag_rate(cm):
        r = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        np.fill_diagonal(r, 0)
        return r

    diff = off_diag_rate(cm_stoch) - off_diag_rate(cm_base)

    vmax = np.abs(diff).max()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Δ confusion rate (stoch − base, %)', fontsize=10)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title('Confusion rate difference (stochastic − baseline)', fontsize=12)

    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            ax.text(j, i, f'{diff[i,j]:+.1f}', ha='center', va='center',
                    fontsize=7, color='black')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_per_class_error(cm_base, cm_stoch, path):
    err_base  = (1 - cm_base.diagonal()  / cm_base.sum(axis=1))  * 100
    err_stoch = (1 - cm_stoch.diagonal() / cm_stoch.sum(axis=1)) * 100

    x = np.arange(10)
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, err_base,  w, label='Constant depth', color='#d62728', alpha=0.85)
    ax.bar(x + w/2, err_stoch, w, label='Stochastic depth', color='#1f77b4', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Error rate (%)', fontsize=11)
    ax.set_title('Per-class test error: constant vs stochastic depth', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(cm_base, cm_stoch, out_path):
    lines = []

    # Overall accuracy
    acc_base  = cm_base.diagonal().sum()  / cm_base.sum()  * 100
    acc_stoch = cm_stoch.diagonal().sum() / cm_stoch.sum() * 100
    lines.append('=== Overall accuracy ===')
    lines.append(f'  Constant depth  : {acc_base:.2f}%')
    lines.append(f'  Stochastic depth: {acc_stoch:.2f}%')
    lines.append('')

    # Per-class error
    lines.append(f'=== Per-class error (%) ===')
    lines.append(f'{"Class":<12} {"Baseline":>10} {"Stochastic":>12} {"Δ":>8}')
    lines.append('-' * 46)
    err_base  = (1 - cm_base.diagonal()  / cm_base.sum(axis=1))  * 100
    err_stoch = (1 - cm_stoch.diagonal() / cm_stoch.sum(axis=1)) * 100
    for i, cls in enumerate(CLASSES):
        delta = err_stoch[i] - err_base[i]
        lines.append(f'{cls:<12} {err_base[i]:>10.1f} {err_stoch[i]:>12.1f} {delta:>+8.1f}')
    lines.append('')

    # Interesting pairwise confusions
    lines.append('=== Pairwise confusion rates for visually similar classes (%) ===')
    lines.append(f'{"True → Pred":<22} {"Baseline":>10} {"Stochastic":>12} {"Δ":>8}')
    lines.append('-' * 56)

    def pairwise(cm, true_cls, pred_cls):
        i, j = CLASSES.index(true_cls), CLASSES.index(pred_cls)
        return cm[i, j] / cm[i].sum() * 100

    for cls_a, cls_b in INTERESTING_PAIRS:
        for src, tgt in [(cls_a, cls_b), (cls_b, cls_a)]:
            r_base  = pairwise(cm_base,  src, tgt)
            r_stoch = pairwise(cm_stoch, src, tgt)
            delta   = r_stoch - r_base
            lines.append(f'{src+" → "+tgt:<22} {r_base:>10.1f} {r_stoch:>12.1f} {delta:>+8.1f}')
        lines.append('')

    text = '\n'.join(lines)
    print(text)
    with open(out_path, 'w') as f:
        f.write(text)
    print(f'Saved: {out_path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    _, _, test_loader = build_loaders(
        dataset_name='cifar10',
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_base  = load_model(args.baseline,   p_L=1.0, device=device)
    model_stoch = load_model(args.stochastic, p_L=0.5, device=device)

    print('Running inference on test set...')
    preds_base,  labels = get_predictions(model_base,  test_loader, device)
    preds_stoch, _      = get_predictions(model_stoch, test_loader, device)

    cm_base  = confusion_matrix(labels, preds_base)
    cm_stoch = confusion_matrix(labels, preds_stoch)

    plot_confusion(cm_base,  'Constant depth — confusion matrix (%)',
                   os.path.join(args.out_dir, 'confusion_baseline.png'))
    plot_confusion(cm_stoch, 'Stochastic depth — confusion matrix (%)',
                   os.path.join(args.out_dir, 'confusion_stochastic.png'))
    plot_diff(cm_stoch, cm_base,
              os.path.join(args.out_dir, 'confusion_diff.png'))
    plot_per_class_error(cm_base, cm_stoch,
                         os.path.join(args.out_dir, 'per_class_error.png'))
    print_summary(cm_base, cm_stoch,
                  os.path.join(args.out_dir, 'summary.txt'))


if __name__ == '__main__':
    main()

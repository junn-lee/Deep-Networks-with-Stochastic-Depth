"""
model.py

Re-implementation of stochastic depth from:
  "Deep Networks with Stochastic Depth", Huang et al., ECCV 2016
  https://arxiv.org/abs/1603.09382

Design choices that differ from the authors' Torch 7 code:
  - The full network is one nn.Module with a ModuleList of blocks; survival
    probabilities are computed once at construction and stored as plain floats
    on each block, rather than being set as mutable .deathRate attributes after
    the model is built and discovered by scanning the module list.
  - Skip connections that need to change spatial size or channel count use a
    1x1 convolution + BN projection (He et al. Option B) instead of average
    pooling followed by zero-channel-padding (Option A used in the authors' code).
  - The Bernoulli drop decision is made inside forward() with a single
    torch.rand() < p comparison; there is no separate "gate" boolean that the
    training loop sets before each mini-batch.
  - ReLU is placed inside the block (after the residual add) rather than added
    as a separate module after each block in the sequential model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class StochasticBlock(nn.Module):
    """
    One residual block with an optional stochastic depth drop.

    The residual branch is:   Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN
    The skip branch is:
      - identity when in_ch == out_ch and stride == 1
      - 1x1 Conv(stride) -> BN  otherwise  (Option B from He et al. 2016)

    During training the entire residual branch is dropped with probability
    (1 - survival_prob) by sampling a single Bernoulli scalar per forward call.
    During evaluation the residual branch output is scaled by survival_prob
    (Eq. 5 in the paper).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, survival_prob: float):
        super().__init__()
        self.survival_prob = survival_prob

        # Residual branch: Conv-BN-ReLU-Conv-BN
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        # Skip branch: projection if dimensions change, identity otherwise
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        if self.training:
            # Drop the entire residual branch for this mini-batch with
            # probability (1 - survival_prob). A single scalar is sampled,
            # so all spatial positions share the same fate (batch-level drop).
            keep = torch.rand(1).item() < self.survival_prob
            if not keep:
                return F.relu(identity)
            return F.relu(self.residual(x) + identity)
        else:
            # Scale residual by survival probability at test time (Eq. 5)
            return F.relu(self.survival_prob * self.residual(x) + identity)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

def _survival_probs(num_blocks: int, p_L: float, mode: str = "linear") -> list[float]:
    """
    Survival probability schedule (Section 3 of the paper).

    mode="linear"  — Eq. 4: p_l = 1 - (l/L)(1 - p_L). Earlier layers survive
                     more often since they compute low-level features reused
                     throughout the network.
    mode="uniform" — p_l = p_L for all l. Single global survival probability.
                     Used in Figure 8 left of the paper to compare against linear.
    mode="constant"— p_l = 1.0 for all l. Equivalent to the baseline ResNet.
    """
    if mode == "uniform":
        return [p_L] * num_blocks
    if mode == "constant":
        return [1.0] * num_blocks
    # linear (default)
    return [1.0 - (l / num_blocks) * (1.0 - p_L) for l in range(1, num_blocks + 1)]


class StochasticDepthNet(nn.Module):
    """
    CIFAR ResNet with stochastic depth.

    For depth=110: blocks_per_group=18, giving 3*18=54 residual blocks total.
    The architecture mirrors He et al. (2016) for CIFAR:
      stem  : Conv(3->16, 3x3) -> BN -> ReLU
      group1: 18 blocks, 16 channels, stride 1
      group2: 18 blocks, 32 channels, stride 2 on first block
      group3: 18 blocks, 64 channels, stride 2 on first block
      head  : global average pool -> linear -> logits

    Args:
        num_classes:      10 for CIFAR-10, 100 for CIFAR-100.
        blocks_per_group: 18 for the 110-layer network (depth = 6n+2, n=18).
        p_L:              Survival probability of the last block under linear
                          decay. Set to 1.0 for constant-depth baseline.
    """

    CHANNEL_SCHEDULE = [16, 32, 64]   # filters per group

    def __init__(self, num_classes: int = 10, blocks_per_group: int = 18,
                 p_L: float = 0.5, survival_mode: str = "linear"):
        super().__init__()

        total_blocks = 3 * blocks_per_group
        probs = _survival_probs(total_blocks, p_L, mode=survival_mode)

        # Stem (always active, not subject to stochastic depth)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Three groups of residual blocks
        channels = self.CHANNEL_SCHEDULE
        block_idx = 0
        groups = []
        for g, out_ch in enumerate(channels):
            in_ch = 16 if g == 0 else channels[g - 1]
            stride_first = 1 if g == 0 else 2
            group_blocks = []
            for b in range(blocks_per_group):
                stride = stride_first if b == 0 else 1
                in_c   = in_ch        if b == 0 else out_ch
                group_blocks.append(
                    StochasticBlock(in_c, out_ch, stride, probs[block_idx])
                )
                block_idx += 1
            groups.append(nn.Sequential(*group_blocks))

        # Store as a ModuleList so parameters are registered
        self.groups = nn.ModuleList(groups)

        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for group in self.groups:
            out = group(out)
        # Global average pool: (N, C, H, W) -> (N, C)
        out = out.mean(dim=[2, 3])
        return self.classifier(out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def cifar_resnet110(num_classes: int = 10, p_L: float = 0.5,
                    survival_mode: str = "linear") -> StochasticDepthNet:
    """110-layer ResNet with stochastic depth (p_L=0.5) or baseline (p_L=1.0)."""
    return StochasticDepthNet(num_classes=num_classes, blocks_per_group=18,
                              p_L=p_L, survival_mode=survival_mode)


def cifar_resnet_custom(num_classes: int = 10, blocks_per_group: int = 18,
                        p_L: float = 0.5, survival_mode: str = "linear") -> StochasticDepthNet:
    """ResNet with configurable depth (blocks_per_group) and survival schedule.
    depth = 6 * blocks_per_group + 2
    blocks_per_group=3  -> 20-layer
    blocks_per_group=6  -> 38-layer
    blocks_per_group=9  -> 56-layer
    blocks_per_group=12 -> 74-layer
    blocks_per_group=15 -> 92-layer
    blocks_per_group=18 -> 110-layer
    """
    return StochasticDepthNet(num_classes=num_classes,
                              blocks_per_group=blocks_per_group,
                              p_L=p_L, survival_mode=survival_mode)


def cifar_resnet110_baseline(num_classes: int = 10) -> StochasticDepthNet:
    """Constant-depth 110-layer ResNet baseline (all survival probs = 1.0)."""
    return StochasticDepthNet(num_classes=num_classes, blocks_per_group=18, p_L=1.0)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    batch = torch.randn(4, 3, 32, 32)

    for p_L, label in [(0.5, "stochastic"), (1.0, "baseline")]:
        net = cifar_resnet110(num_classes=10, p_L=p_L)
        net.train()
        out_train = net(batch)
        net.eval()
        with torch.no_grad():
            out_eval = net(batch)
        print(f"[{label}] train logits: {out_train.shape}, "
              f"eval logits: {out_eval.shape}, "
              f"params: {net.count_parameters():,}")

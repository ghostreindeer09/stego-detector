from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import get_kv_kernel_5x5


class LearnableHPF(nn.Module):
    """Learnable 5x5 HPF initialized from KV kernel (per RGB channel)."""

    def __init__(self):
        super().__init__()
        kv = get_kv_kernel_5x5().reshape(1, 1, 5, 5)
        kv = torch.from_numpy(kv).repeat(3, 1, 1, 1)
        self.weight = nn.Parameter(kv.clone(), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, bias=None, padding=2, groups=3)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx = self.fc(self.max_pool(x).view(b, c))
        attn = (avg + mx).view(b, c, 1, 1)
        return x * attn


class SRNet(nn.Module):
    """
    SRNet with fixed KV high-pass filter as the first non-trainable layer.
    Input: [B, 3, H, W] RGB. KV HPF extracts residuals; backbone classifies.
    """

    def __init__(
        self,
        num_classes: int = 1,
        use_kv_hpf: bool = True,
        use_learnable_hpf: bool = False,
        use_channel_attention: bool = False,
    ):
        super().__init__()
        from .features import KVHighPassFilter

        self.use_kv_hpf = use_kv_hpf
        self.use_learnable_hpf = use_learnable_hpf
        if use_learnable_hpf:
            self.kv_hpf = LearnableHPF()
        elif use_kv_hpf:
            self.kv_hpf = KVHighPassFilter()
        else:
            self.kv_hpf = nn.Identity()
        self.backbone = SRNetBackbone(
            in_channels=3,
            num_classes=num_classes,
            use_channel_attention=use_channel_attention,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.kv_hpf(x)
        return self.backbone(x)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class SRNetBackbone(nn.Module):
    """
    Simplified SRNet-inspired residual CNN for steganalysis on noise residual maps.
    """

    def __init__(self, in_channels: int, num_classes: int = 1, use_channel_attention: bool = False):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 128, stride=2)
        self.layer4 = BasicBlock(128, 256, stride=2)
        self.layer5 = BasicBlock(256, 256, stride=2)
        self.channel_attention = ChannelAttention(256) if use_channel_attention else nn.Identity()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.channel_attention(x)
        feat_map = x
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits, feat_map


class GradCAM:
    """
    Grad-CAM implementation for SRNetBackbone.
    """

    def __init__(self, model: SRNetBackbone, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, _input, output):
            self.activations = output

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for h in self.hook_handles:
            h.remove()

    def generate(self, scores: torch.Tensor, class_idx: Optional[int] = None) -> torch.Tensor:
        """
        scores: [B, 1] logits of the model (before sigmoid).
        Returns heatmap: [H, W] GPU tensor in [0,1].
        """
        if class_idx is None:
            class_idx = 0

        self.model.zero_grad()

        target = scores[:, class_idx].sum()
        target.backward(retain_graph=True)

        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]

        alpha = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.squeeze(0).squeeze(0)


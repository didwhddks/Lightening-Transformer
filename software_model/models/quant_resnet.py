# -*- coding: utf-8 -*-
"""
LSQ-aware Quantized ResNet for ImageNet-1K training.
Follows the same quantization conventions as quant_vit.py:
  - Stem conv (first layer): fixed 8-bit
  - All intermediate conv layers: wbits/abits via LSQ
  - Classifier FC (last layer): fixed 8-bit
  - When wbits > 16: fall back to full-precision modules
"""

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from ops.quantize import QuantAct, QuantLinear, QuantConv2d
from ops._quant_base import Qmodes


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
    }


class QuantBasicBlock(nn.Module):
    """BasicBlock with LSQ-quantized conv layers (used in ResNet-18/34)."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 wbits=32, abits=32, offset=False,
                 input_noise_std=0, output_noise_std=0, enable_linear_noise=False):
        super().__init__()
        if wbits <= 16:
            self.conv1 = QuantConv2d(
                inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                nbits=wbits, nbits_a=abits, mode=Qmodes.layer_wise, offset=offset,
                input_noise_std=input_noise_std, output_noise_std=output_noise_std,
                enable_linear_noise=enable_linear_noise)
            self.conv2 = QuantConv2d(
                planes, planes, kernel_size=3, padding=1, bias=False,
                nbits=wbits, nbits_a=abits, mode=Qmodes.layer_wise, offset=offset,
                input_noise_std=input_noise_std, output_noise_std=output_noise_std,
                enable_linear_noise=enable_linear_noise)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x                     # (B, inplanes, H, W)

        out = self.conv1(x)              # (B, planes, H/stride, W/stride)  kernel=3, stride=stride, pad=1
        out = self.bn1(out)              # (B, planes, H/stride, W/stride)
        out = self.relu1(out)            # (B, planes, H/stride, W/stride)

        out = self.conv2(out)            # (B, planes, H/stride, W/stride)  kernel=3, stride=1, pad=1
        out = self.bn2(out)              # (B, planes, H/stride, W/stride)

        if self.downsample is not None:
            identity = self.downsample(x)  # (B, planes, H/stride, W/stride)  1x1 conv 對齊維度

        out += identity                  # (B, planes, H/stride, W/stride)
        out = self.relu2(out)            # (B, planes, H/stride, W/stride)
        return out


class QuantResNet(nn.Module):
    """
    Quantized ResNet with LSQ-aware training.

    Quantization scheme (consistent with QuantVisionTransformer):
      - conv1 (stem):  fixed 8-bit weight/activation
      - BasicBlock convs and downsample convs: wbits/abits
      - fc (head): fixed 8-bit weight/activation
    """

    def __init__(self, block, layers, num_classes=1000,
                 wbits=32, abits=32, offset=False,
                 input_noise_std=0, output_noise_std=0, enable_linear_noise=False,
                 **kwargs):  # absorb ViT-specific kwargs (headwise, phase_noise_std, etc.)
        super().__init__()

        if wbits > 16:
            print("Use float weights.")
        else:
            print(f"Use {wbits} bit weights.")
        if abits > 16:
            print("Use float activations.")
        else:
            print(f"Use {abits} bit activations.")

        self.inplanes = 64

        # Stem: fixed 8-bit (mirrors QuantPatchEmbed in ViT)
        if wbits <= 16:
            self.conv1 = QuantConv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False,
                nbits=8, nbits_a=8, mode=Qmodes.layer_wise, offset=offset,
                input_noise_std=input_noise_std, output_noise_std=output_noise_std,
                enable_linear_noise=enable_linear_noise)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        quant_kwargs = dict(
            wbits=wbits, abits=abits, offset=offset,
            input_noise_std=input_noise_std, output_noise_std=output_noise_std,
            enable_linear_noise=enable_linear_noise)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1, **quant_kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **quant_kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **quant_kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **quant_kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Head: fixed 8-bit (mirrors head in ViT)
        if wbits <= 16:
            self.fc = QuantLinear(
                512 * block.expansion, num_classes,
                nbits=8, nbits_a=8, mode=Qmodes.layer_wise, offset=offset,
                input_noise_std=input_noise_std, output_noise_std=output_noise_std,
                enable_linear_noise=enable_linear_noise)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride=1,
                    wbits=32, abits=32, offset=False,
                    input_noise_std=0, output_noise_std=0, enable_linear_noise=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if wbits <= 16:
                down_conv = QuantConv2d(
                    self.inplanes, planes * block.expansion, kernel_size=1,
                    stride=stride, bias=False,
                    nbits=wbits, nbits_a=abits, mode=Qmodes.layer_wise, offset=offset,
                    input_noise_std=input_noise_std, output_noise_std=output_noise_std,
                    enable_linear_noise=enable_linear_noise)
            else:
                down_conv = nn.Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=1,
                    stride=stride, bias=False)
            downsample = nn.Sequential(
                down_conv,
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample,
                        wbits=wbits, abits=abits, offset=offset,
                        input_noise_std=input_noise_std, output_noise_std=output_noise_std,
                        enable_linear_noise=enable_linear_noise)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes,
                                wbits=wbits, abits=abits, offset=offset,
                                input_noise_std=input_noise_std, output_noise_std=output_noise_std,
                                enable_linear_noise=enable_linear_noise))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@register_model
def resnet18_quant(pretrained=False, **kwargs):
    """LSQ-quantized ResNet-18 for ImageNet-1K."""
    model = QuantResNet(QuantBasicBlock, [2, 2, 2, 2], **kwargs)
    model.default_cfg = _cfg()
    return model

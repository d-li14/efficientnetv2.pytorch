"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021). 
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""

import torch
import torch.nn as nn
import math
from functools import partial

__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl',
           'effnetv2_base', 'effnetv2_b0', 'effnetv2_b1', 'effnetv2_b2', 'effnetv2_b3']


def _make_divisible(v, divisor, min_value=None, round_limit=.9):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None, round_limit=0.9):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    return _make_divisible(channels * multiplier, divisor, min_value=channel_min, round_limit=round_limit)


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., cfgs=None, stem_size=24, num_feature=1792):
        super(EffNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(stem_size * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(num_feature * width_mult, 8) if width_mult > 1.0 else num_feature
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    settings = {"cfgs": [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
        ]
    }
    kwargs.update(settings)
    return EffNetV2(**kwargs)

def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    settings = {"cfgs": [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    }
    kwargs.update(settings)
    return EffNetV2(**kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    settings = {"cfgs": [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    }
    kwargs.update(settings)
    return EffNetV2(**kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    settings = {"cfgs": [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    }
    kwargs.update(settings)
    return EffNetV2(**kwargs)


def effnetv2_base(**kwargs):
    """
    Constructs a EfficientNetV2-Base model
    """
    width_mult = kwargs.pop("width_mult", 1.0)
    round_chs_fn = partial(round_channels, multiplier=width_mult, round_limit=0.)
    num_feature = round_chs_fn(1280)

    depth_multiplier = kwargs.pop("depth_multiplier", 1.0)

    settings = {"cfgs": [
            # t, c, n, s, SE
            [1, 16, 1, 1, 0],
            [4, 32, 2, 2, 0],
            [4, 48, 2, 2, 0],
            [4, 96, 3, 2, 1],
            [6, 112, 5, 1, 1],
            [6, 192, 8, 2, 1],
        ],
        "stem_size": 32,
        "num_feature": num_feature,
        "width_mult": width_mult
    }
    # scale depth
    for i in range(len(settings["cfgs"])):
        settings["cfgs"][i][2] = int(math.ceil(depth_multiplier*settings["cfgs"][i][2]))

    kwargs.update(settings)
    return EffNetV2(**kwargs)


effnetv2_b0 = effnetv2_base


def effnetv2_b1(**kwargs):
    settings = {
        "depth_multiplier": 1.1,
        "width_mult": 1.0,
    }
    kwargs.update(settings)
    return effnetv2_base(**kwargs)


def effnetv2_b2(**kwargs):
    settings = {
        "depth_multiplier": 1.2,
        "width_mult": 1.1,
    }
    kwargs.update(settings)
    return effnetv2_base(**kwargs)


def effnetv2_b3(**kwargs):
    settings = {
        "depth_multiplier": 1.4,
        "width_mult": 1.2,
    }
    kwargs.update(settings)
    return effnetv2_base(**kwargs)

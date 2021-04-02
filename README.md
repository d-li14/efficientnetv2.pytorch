# PyTorch implementation of EfficientNet V2

Reproduction of EfficientNet V2 architecture as described in [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) by Mingxing Tan, Quoc V. Le with the [PyTorch](pytorch.org) framework.

# Requirements

PyTorch 1.8+ is required to support [nn.SiLU](https://pytorch.org/docs/master/generated/torch.nn.SiLU.html)

# Models

| Architecture      | # Parameters | FLOPs | Top-1 / Top-5 Accuracy (%) |
| ----------------- | ------------ | ------ | -------------------------- |
| EfficientNetV2-S    | 24M |  |  |

More model definitions are pending for architectural details.

Stay tuned for ImageNet pre-trained weights.

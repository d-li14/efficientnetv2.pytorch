**[NEW!]** Check out our latest work [involution](https://github.com/d-li14/involution) accepted to CVPR'21 that introduces a new neural operator, other than convolution and self-attention.

---

# PyTorch implementation of EfficientNet V2

Reproduction of EfficientNet V2 architecture as described in [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) by Mingxing Tan, Quoc V. Le with the [PyTorch](pytorch.org) framework.

# Requirements

PyTorch 1.7+ is required to support [nn.SiLU](https://pytorch.org/docs/master/generated/torch.nn.SiLU.html)

# Models

| Architecture      | # Parameters | FLOPs | Top-1 Acc. (%) |
| ----------------- | ------------ | ------ | -------------------------- |
| EfficientNetV2-S    | 24M | 8.8B |  |

More model definitions are pending for architectural details from the authors.

Stay tuned for ImageNet pre-trained weights.

# Acknowledgement

The implementation is heavily borrowed from [HBONet](https://github.com/d-li14/HBONet) or [MobileNetV2](https://github.com/d-li14/mobilenetv2.pytorch), please kindly consider citing the following

```
@InProceedings{Li_2019_ICCV,
author = {Li, Duo and Zhou, Aojun and Yao, Anbang},
title = {HBONet: Harmonious Bottleneck on Two Orthogonal Dimensions},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}
```
```
@InProceedings{Sandler_2018_CVPR,
author = {Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
title = {MobileNetV2: Inverted Residuals and Linear Bottlenecks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

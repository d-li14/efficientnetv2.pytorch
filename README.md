**[NEW!]** Check out our latest work [involution](https://github.com/d-li14/involution) accepted to CVPR'21 that introduces a new neural operator, other than convolution and self-attention.

---

# PyTorch implementation of EfficientNet V2

Reproduction of EfficientNet V2 architecture as described in [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) by Mingxing Tan, Quoc V. Le with the [PyTorch](pytorch.org) framework.

# Requirements

PyTorch 1.7+ is required to support [nn.SiLU](https://pytorch.org/docs/master/generated/torch.nn.SiLU.html)

# Models

| Architecture      | # Parameters | FLOPs | Top-1 Acc. (%) |
| ----------------- | ------------ | ------ | -------------------------- |
| EfficientNetV2-S  | 22.103832M   | 23202.270720M  |  |
| EfficientNetV2-M  | 55.300016M   | 43557.531136M |
| EfficientNetV2-L  | 119.355792M  | 98599.022080M|
| EfficientNetV2-XL | 208.960328M  | 144211.693568M|
| EfficientNetV2-B0 | 7.780248M | 5952.832768M | |
| EfficientNetV2-B1 | 9.009872M | 6685.624320M | |
| EfficientNetV2-B2 | 10.749136M | 9067.325440M | |
| EfficientNetV2-B3 | 14.461720M | 11929.994368M | |

* Flops are all measured on input (224, 224).

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

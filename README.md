**[NEW!]** Check out our latest work [involution](https://github.com/d-li14/involution) accepted to CVPR'21 that introduces a new neural operator, other than convolution and self-attention.

---

# PyTorch implementation of EfficientNet V2

Reproduction of EfficientNet V2 architecture as described in [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) by Mingxing Tan, Quoc V. Le with the [PyTorch](pytorch.org) framework.

## Models

| Architecture      | # Parameters | FLOPs | Top-1 Acc. (%) |
| ----------------- | ------------ | ------ | -------------------------- |
| EfficientNetV2-B0 | 7.17M | 0.79G @ 224 | |
| EfficientNetV2-B1 | 8.18M | 1.09G @ 224 | |
| EfficientNetV2-B2 | 10.37M | 1.25G @ 224 | |
| EfficientNetV2-B3 | 14.69M | 1.83G @ 224  | |
| EfficientNetV2-S    | 21.10M | 2.90G @ 224 |  |
| EfficientNetV2-M    | 55.30M | 5.44G @ 224 |  |
| EfficientNetV2-L    | 119.36M | 12.32G @ 224 |  |
| EfficientNetV2-XL    | 208.96M | 18.02G @ 224 |  |

Stay tuned for ImageNet pre-trained weights.

## Acknowledgement

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

The official [TensorFlow implementation](https://github.com/google/automl/tree/master/efficientnetv2) by [@mingxingtan](https://github.com/mingxingtan).

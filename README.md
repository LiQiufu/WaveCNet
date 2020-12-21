# WaveCNet [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Wavelet_Integrated_CNNs_for_Noise-Robust_Image_Classification_CVPR_2020_paper.pdf)

The WaveCNet is designed using DWT and the commonly used CNN networks in PyTorch: https://pytorch.org/docs/stable/torchvision/models.html#classification

The main.py is revised on the PyTorch image classification code: https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101

WaveUNet for image segmentation has been renamed as WaveSNet: https://github.com/LiQiufu/WaveSNet

## (0) Paper Title
Wavelet Integrated CNNs for Noise-Robust Image Classification

## (1) Training WaveCNet on ImageNet

```bash
 CUDA_VISIBLE_DEVICES=0 python main.py --data /PYTHON_TO_IMANGENET -a resnet18_dwt -b 256 -w bior3.3 --gpu 0 --lr 0.1
```

## (2) The trained weight files

The pretrained weight files have be uploaded on the website: https://pan.baidu.com/s/1RN_WW0dRrTHmLdKTTGTZgg  (passwords: sf1d)

## (3) The paper

If the code or method help you in the research, please cite the following paper:

```bash
@InProceedings(qiufu_2020_CVPR,
author = {Li, Qiufu and Shen, Linlin and Guo, Sheng and Lai, Zhihui},
title = {Wavelet Integrated CNNs for Noise-Robust Image Classification},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {june},
year = {2020}
}
```

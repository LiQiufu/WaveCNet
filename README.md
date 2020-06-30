# WaveCNet [[paper]]()

The WaveCNet is designed using DWT and the commonly used CNN networks in PyTorch: https://pytorch.org/docs/stable/torchvision/models.html#classification

The WaveCNet code is revised on the PyTorch image classification code: https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101

## (1) Training WaveCNet on ImageNet

```bash
 CUDA_VISIBLE_DEVICES=0 python main.py --data /PYTHON_TO_IMANGENET -a resnet18_dwt -b 256 -w bior3.3 --gpu 0 --lr 0.1
```

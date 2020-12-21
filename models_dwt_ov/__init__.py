"""
这个模块提供 WaveCNet 的其他版本：
WaveCNet的规则是 X --> X_ll
其他版本包括：
    1) X --> torch.cat(X_ll, X_lh, X_hl, X_hh)
    2) X --> 1/4*(X_ll + X_lh + X_hl + X_hh)
"""

import pywt
import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from DWT_IDWT.DWT_IDWT_layer import *

class Downsample_v1(nn.Module):
    """
        for ResNet_C
        X --> torch.cat(X_ll, X_lh, X_hl, X_hh)
    """
    def __init__(self, wavename = 'haar'):
        super(Downsample_v1, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return torch.cat((LL, LH, HL, HH), dim = 1)


class Downsample_v2(nn.Module):
    """
        for ResNet_A
        X --> 1/4*(X_ll + X_lh + X_hl + X_hh)
    """
    def __init__(self, wavename = 'haar'):
        super(Downsample_v2, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return (LL + LH + HL + HH) / 4

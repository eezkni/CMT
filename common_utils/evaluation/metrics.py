import torch
from torch import nn
import torch.utils.data

import pyiqa
import numpy as np
import lpips as lpips_

class PSNR(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.psnr = pyiqa.create_metric('psnr').cuda()

    def forward(self, input_tensor, target):
        return self.psnr(input_tensor.unsqueeze(0), target.unsqueeze(0)).item()

class SSIM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ssim = pyiqa.create_metric('ssim').cuda()

    def forward(self, input_tensor, target):
        return self.ssim(input_tensor.unsqueeze(0), target.unsqueeze(0)).item()

class LPIPS(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fn_alex = lpips_.LPIPS(net='alex').cuda()

    """
    Input of tensors are [0, 1]
    """
    def forward(self, input_tensor, target):
        # input_tensor -> (c, h, w)
        with torch.no_grad():
            # convert to [-1, 1]
            tensor_a = (input_tensor - 0.5) * 2
            tensor_b = (target - 0.5) * 2

            tensor_a = tensor_a.squeeze(0)
            tensor_b = tensor_b.squeeze(0)

            result = self.loss_fn_alex(tensor_a, tensor_b)

        return result.item()

class NIQE(nn.Module):
    def __init__(self):
        super().__init__()

        self.niqe = pyiqa.create_metric('niqe_matlab').cuda()

    def forward(self, input_tensor, target):
        return self.niqe(input_tensor.unsqueeze(0)).item()

class NIMA(nn.Module):
    def __init__(self):
        super().__init__()

        self.nima = pyiqa.create_metric('nima').cuda()

    def forward(self, input_tensor, target):
        return self.nima(input_tensor.unsqueeze(0)).item()

class UNIQUE(nn.Module):
    def __init__(self):
        super().__init__()

        self.unique = pyiqa.create_metric('unique').cuda()

    def forward(self, input_tensor, target):
        return self.unique(input_tensor.unsqueeze(0)).item()

class BRISQUE(nn.Module):
    def __init__(self):
        super().__init__()

        self.brisque = pyiqa.create_metric('brisque_matlab').cuda()

    def forward(self, input_tensor, target):
        return self.brisque(input_tensor.unsqueeze(0)).item()

class CLIPIQA(nn.Module):
    def __init__(self):
        super().__init__()

        self.brisque = pyiqa.create_metric('clipiqa').cuda()

    def forward(self, input_tensor, target):
        return self.brisque(input_tensor.unsqueeze(0)).item()

class MUSIQ(nn.Module):
    def __init__(self):
        super().__init__()

        self.brisque = pyiqa.create_metric('musiq').cuda()

    def forward(self, input_tensor, target):
        return self.brisque(input_tensor.unsqueeze(0)).item()

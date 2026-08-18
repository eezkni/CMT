import torch
from torch import nn
import torch.utils.data

import pyiqa
import numpy as np
import lpips
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

def mae(input_tensor, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input_tensor, target)
    return output

# input_tensor and target tensor should be 0 ~ 1
class psnr(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psnr_func = reference_pyiqa_metric_fun_generator('psnr')

    def forward(self, input_tensor, target):
        return self.psnr_func(input_tensor, target)


class ssim(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssim_func = reference_pyiqa_metric_fun_generator('ssim')

    def forward(self, input_tensor, target):
        return self.ssim_func(input_tensor, target)

# def psnr(input_tensor, target):
#     psnr_func = reference_pyiqa_metric_fun_generator('psnr')
#     return psnr_func(input_tensor, target)
#
# def ssim(input_tensor, target):
#     ssim_func = reference_pyiqa_metric_fun_generator('ssim')
#     return ssim_func(input_tensor, target)
# def psnr(input_tensor, target):
#     return peak_signal_noise_ratio(np.array(input_tensor.cpu()), np.array(target.cpu()))
#
#
# # ssim_func = reference_pyiqa_metric_fun_generator('ssim')
# def ssim(input_tensor, target):
#     return structural_similarity(np.array(torch.squeeze(input_tensor.cpu())), np.array(torch.squeeze(target.cpu())), channel_axis=0, data_range=1)


class packLPIPS(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn_alex = lpips.LPIPS(net='alex').cuda()

    """
    Input of tensors are [0, 1]
    """
    def forward(self, input_tensor, target):
        with torch.no_grad():
            # convert to [-1, 1]
            tensor_a = (input_tensor - 0.5) * 2
            tensor_b = (target - 0.5) * 2
            result = self.loss_fn_alex(tensor_a, tensor_b)

            result_sum = 0
            if len(result.shape) == 0:
                return float(result)

            for i in result:
                result_sum += i

        return float(result_sum / len(result))

# def lpips(input_tensor, target):
#     method_metric = pyiqa.create_metric('lpips').cuda()
#     if len(input_tensor.shape) <= 3:
#         reshape_input_tensor = reshape_tensor(input_tensor)
#         reshape_target = reshape_tensor(target)
#     else:
#         reshape_input_tensor = input_tensor
#         reshape_target = target
#
#     result = method_metric(reshape_input_tensor, reshape_target)
#
#     result_sum = 0
#     if len(result.shape) == 0:
#         return result
#
#     for i in result:
#         result_sum += i
#
#     return result_sum / len(result)
    # lpips_func = reference_pyiqa_metric_fun_generator('lpips')
    # return lpips_func(input_tensor, target)

def reverse_niqe(input_tensor, target):
    method_metric = pyiqa.create_metric('niqe').cuda()
    if len(input_tensor.shape) <= 3:
        reshape_input_tensor = reshape_tensor(input_tensor)
        # reshape_target = reshape_tensor(target)
    else:
        reshape_input_tensor = input_tensor
        # reshape_target = target

    result = method_metric(reshape_input_tensor)

    # result_sum = 0
    # if len(result.shape) == 1:
    #     result_sum = result
    # else:
    #     for i in result:
    #         result_sum += i
    #
    # result = result_sum / len(result)

    return (10 - result) * 4

def reference_pyiqa_metric_fun_generator(metric_name):
    def metric_fun(input_tensor, target):
        method_metric = pyiqa.create_metric(metric_name).cuda()
        if len(input_tensor.shape) <= 3:
            reshape_input_tensor = reshape_tensor(input_tensor)
            reshape_target = reshape_tensor(target)
        else:
            reshape_input_tensor = input_tensor
            reshape_target = target

        result = method_metric(reshape_input_tensor, reshape_target)

        result_sum = 0
        if len(result.shape) == 0:
            return result

        for i in result:
            result_sum += i

        return result_sum / len(result)

    return metric_fun

def reshape_tensor(input_tensor):
    with torch.no_grad():
        input_tensor = input_tensor.clamp_(0, 1).detach()
        # norm_input_tensor = input_tensor + torch.ones_like(input_tensor)
        # norm_input_tensor = norm_input_tensor / 2.0
    return input_tensor.unsqueeze(0)

def binary_class_acc(input_tensor, target):
    with torch.no_grad():
        num_count = target.shape[0]
        correct_count = 0

        for i in range(num_count):
            if (input_tensor[i][0] - input_tensor[i][1]) * (target[i][0] - target[i][1]) > 0:
                correct_count += 1

    return correct_count / num_count

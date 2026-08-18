import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torch.autograd import Variable
import torchvision.models as models
from abc import ABC, abstractmethod
import piq
from collections import OrderedDict
import numpy as np
from loguru import logger

from utils.LPIPS_loss import get_network, LinLayers


# all loss should inherit from LossWithWeights class
class LossWithWeights(nn.Module):
    def __init__(self):
        super(LossWithWeights, self).__init__()

    @abstractmethod
    def weight_from_epoch(self, epoch_index):
        pass


class L1Loss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.L1Loss = nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        return self.L1Loss(x, y)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class L2Loss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.L2Loss = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        return self.L2Loss(x, y)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class FocalLoss(LossWithWeights):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input_tensor, target):
        if input_tensor.dim() > 2:
            input_tensor = input_tensor.view(input_tensor.size(0), input_tensor.size(1), -1)  # N,C,H,W => N,C,H*W
            input_tensor = input_tensor.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input_tensor = input_tensor.contiguous().view(-1, input_tensor.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        log_pt = F.log_softmax(input_tensor)
        log_pt = log_pt.gather(1, target)
        log_pt = log_pt.view(-1)
        pt = Variable(log_pt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input_tensor.data.type():
                self.alpha = self.alpha.type_as(input_tensor.data)
            at = self.alpha.gather(0, target.data.view(-1))
            log_pt = log_pt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * log_pt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def weight_from_epoch(self, epoch_index):
        return 1.0


class VGG19_relu(torch.nn.Module):
    def __init__(self, device='cuda'):
        super(VGG19_relu, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # cnn = models.vgg19(pretrained=True)
        # cnn = models.vgg19()
        cnn = getattr(models, 'vgg19')(weights='VGG19_Weights.IMAGENET1K_V1')
        # cnn.load_state_dict(torch.load(os.path.join('./models/', 'vgg19-dcbb9e9d.pth')))
        cnn = cnn.to(device)
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class PerceptualLoss(LossWithWeights):
    def __init__(self, loss_weight, weights=[1.0, 1.0, 1.0, 1.0, 1.0], resize=False, criterion='l1', device='cuda'):
        super(PerceptualLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'sl1':
            self.criterion = torch.nn.SmoothL1Loss()
        elif criterion == 'l2':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(criterion))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.add_module('vgg', VGG19_relu(device))
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        self.weights = weights
        self.resize = resize
        self.transformer = torch.nn.functional.interpolate

        self.loss_weight = loss_weight

    # Input => [0, 1]
    # Output => [0, +inf)
    def __call__(self, x, y):
        if self.resize:
            x = self.transformer(x, mode='bicubic', size=(224, 224), align_corners=True)
            y = self.transformer(y, mode='bicubic', size=(224, 224), align_corners=True)

        # x = (x + 1) / 2
        # y = (y + 1) / 2
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean.to(x)) / self.std.to(x)
        y = (y - self.mean.to(y)) / self.std.to(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0.0
        loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return loss

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


# Input => [-1, 1]
# Output => [0, +inf)
# class CustomLoss(torch.nn.Module):
#     def __init__(self, perceptual_loss_weight):
#         super(CustomLoss, self).__init__()
#         self.perceptual_loss = PerceptualLoss()
#         self.L1_loss = nn.L1Loss()
#         self.perceptual_loss_weight = perceptual_loss_weight
#
#         self.suggest_ratio_sum = 0
#         self.call_count = 0
#
#     def __call__(self, x, y):
#         l1_result = self.L1_loss(x, y)
#         perceptual_result = self.perceptual_loss(x, y)
#         with torch.no_grad():
#             suggest_ratio = l1_result.item() / perceptual_result
#             self.suggest_ratio_sum += suggest_ratio
#             self.call_count += 1
#
#         # print('L1 Loss: {}, Perceptual Loss: {}'.format(l1_result.item(), perceptual_result))
#         return l1_result + self.perceptual_loss_weight * perceptual_result
#
#     def get_suggest_ratio(self):
#         suggest_ratio = self.suggest_ratio_sum / self.call_count
#         self.suggest_ratio_sum = 0
#         self.call_count = 0
#
#         return suggest_ratio


# loss = PerceptualLoss(weights=[1.0, 1.0, 1.0, 1.0, 1.0], resize=False, criterion='l1', device='cuda')

### TV Loss
def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def tv_loss(x):
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)


class TotalVariationLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super(TotalVariationLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        return tv_loss(x) / batch_size

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


# TODO NOTE: Data range not checked
class EASLoss(LossWithWeights):
    ''' edge aware smoothness loss '''

    def __init__(self, loss_weight=1):
        super(EASLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

        self.loss_weight = loss_weight

    def gradient_xy(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx, gy

    def forward(self, pred, gt):
        pred_grad_x, pred_grad_y = self.gradient_xy(pred)
        gt_grad_x, gt_grad_y = self.gradient_xy(gt)

        weights_x = torch.exp(-torch.mean(torch.abs(gt_grad_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(gt_grad_y), 1, keepdim=True))

        smoothness_x = torch.abs(pred_grad_x) * weights_x
        smoothness_y = torch.abs(pred_grad_y) * weights_y

        loss = (torch.mean(smoothness_x) + torch.mean(smoothness_y))

        return loss

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class ModifiedSSIMLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super(ModifiedSSIMLoss, self).__init__()
        self.loss = piq.SSIMLoss().to("cuda")

        self.loss_weight = loss_weight

    def forward(self, sr, hr):
        # sr = (sr + 1) / 2
        # hr = (hr + 1) / 2

        sr = torch.clamp(sr, 0, 1)
        hr = torch.clamp(hr, 0, 1)

        return self.loss(sr, hr)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class MultiScaleSSIMLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super(MultiScaleSSIMLoss, self).__init__()
        self.loss = piq.MultiScaleSSIMLoss()

        self.loss_weight = loss_weight

    def forward(self, sr, hr):
        # sr = (sr + 1) / 2
        # hr = (hr + 1) / 2

        sr = torch.clamp(sr, 0, 1)
        hr = torch.clamp(hr, 0, 1)

        return self.loss(sr, hr)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


# LPIPS Loss
class LPIPSLoss(LossWithWeights):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).
    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """

    def __init__(self, loss_weight=1, net_type: str = 'alex', version: str = '0.1'):
        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPSLoss, self).__init__()

        # pretrained network
        self.net = get_network(net_type).to("cuda")

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list).to("cuda")
        self.lin.load_state_dict(get_state_dict(net_type, version))

        self.loss_weight = loss_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x = (x + 1) / 2
        # y = (y + 1) / 2

        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / x.shape[0]

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
          + f'master/lpips/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict


def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity - ema_norm_similarity) ** 2
    return similarity_mse_loss


class FeatureSRCLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        x_extracted = x['extracted']
        y_extracted = y['extracted']

        loss_src = torch.sum(relation_mse_loss(x_extracted, y_extracted)) / x_extracted.shape[0]

        return loss_src

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class OutputSRCLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.feature_src_loss = FeatureSRCLoss(loss_weight=1)
        self.pretrained_feature_extract_module = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).cuda()

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.loss_weight = loss_weight

    def forward(self, x, y):
        feature_x = self.pretrained_feature_extract_module(self.normalize(x))
        feature_y = self.pretrained_feature_extract_module(self.normalize(y))

        return self.feature_src_loss(
            {'extracted': feature_x},
            {'extracted': feature_y}
        )

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class FeatureEncoderLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.mae_loss = nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        x_encoder_result = x['encoder_result']
        y_encoder_result = y['encoder_result']

        loss_encoder = 0
        for x_result, y_result in zip(x_encoder_result, y_encoder_result):
            loss_encoder += self.mae_loss(x_result, y_result)

        return loss_encoder / len(x_encoder_result)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class SmoothL1Loss(LossWithWeights):
    def __init__(self, beta, loss_weight=1):
        super().__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, x, y):
        # todo
        pass

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class CutMixLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.L1Loss = nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        if x[1] is None:
            return torch.zeros(1).cuda()

        loss_sum = 0
        for index in range(y.shape[0]):
            bbx1, bby1, bbx2, bby2 = x[1][index]

            loss_sum += self.L1Loss(x[0][index, :, bbx1:bbx2, bby1:bby2], y[index, :, bbx1:bbx2, bby1:bby2])

        return loss_sum / y.shape[0]

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class BatchMaskConsistencyLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        # logger.warning('Deprecated Class: {}'.format(self.__class__.__name__))
        self.L1LossSum = nn.L1Loss(reduction='sum')
        self.loss_weight = loss_weight

    def forward(self, x, y):
        loss_sum = torch.sum(torch.abs((x[0] - y) * x[1])) / torch.sum(x[2])
        # masked_x = x[0] * x[1]
        # masked_gt = y * x[1]
        # loss_sum = self.L1LossSum(masked_x, masked_gt)

        return loss_sum

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight

class MaskConsistencyLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.L1LossSum = nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, x, y):
        learn_img = x[0]
        mask = x[1]
        pixel_count = x[2]
        loss_patch = torch.sum(torch.abs(learn_img - y) * mask, dim=[1, 2, 3]) / pixel_count

        return torch.mean(loss_patch)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class ColorConsistLoss(LossWithWeights):
    """
    Code from Zero-DCE
    """

    def __init__(self, loss_weight=1):
        super(ColorConsistLoss, self).__init__()

        self.loss_weight = loss_weight

    def forward(self, x, y):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return torch.mean(k)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class L_bright_cosist(LossWithWeights):

    def __init__(self, loss_weight=1):
        super(L_bright_cosist, self).__init__()
        self.loss_weight = loss_weight

    def gradient_Consistency_loss_patch(self, x, y):
        # B*C*H*W
        min_x = torch.abs(x.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2, keepdim=True)[0].min(3, keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        # B*1*1,3
        product_separte_color = (x * y).mean([2, 3], keepdim=True)
        x_abs = (x ** 2).mean([2, 3], keepdim=True) ** 0.5
        y_abs = (y ** 2).mean([2, 3], keepdim=True) ** 0.5
        loss1 = (1 - product_separte_color / (x_abs * y_abs + 0.00001)).mean() + torch.mean(
            torch.acos(product_separte_color / (x_abs * y_abs + 0.00001)))

        product_combine_color = torch.mean(product_separte_color, 1, keepdim=True)
        x_abs2 = torch.mean(x_abs ** 2, 1, keepdim=True) ** 0.5
        y_abs2 = torch.mean(y_abs ** 2, 1, keepdim=True) ** 0.5
        loss2 = torch.mean(1 - product_combine_color / (x_abs2 * y_abs2 + 0.00001)) + torch.mean(
            torch.acos(product_combine_color / (x_abs2 * y_abs2 + 0.00001)))
        return loss1 + loss2

    def forward(self, x, y):
        B, C, H, W = x.shape
        loss = self.gradient_Consistency_loss_patch(x, y)
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, 0:W // 2], y[:, :, 0:H // 2, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, 0:W // 2], y[:, :, H // 2:, 0:W // 2])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, 0:H // 2, W // 2:], y[:, :, 0:H // 2, W // 2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:, :, H // 2:, W // 2:], y[:, :, H // 2:, W // 2:])

        return loss  # +loss1#+torch.mean(torch.abs(x-y))#+loss1

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


def L1toNeg1to1(x):
    # x is the result of L1, [0, +inf)
    return -1 + 1 / (x + 0.5)  # return (-1, 1]


def L1toNeg1to1Cosine(x):
    # x is the result of L1, [0, 1]
    return torch.cos(torch.pi * x)  # return [-1, 1]


# reference arxiv
class ContrastiveLoss(LossWithWeights):
    def __init__(self, gamma, alpha, m, range_mapping_method, loss_weight=1):
        super(ContrastiveLoss, self).__init__()
        self.loss_weight = loss_weight
        self.L1Loss = nn.L1Loss()
        self.gamma = gamma
        self.alpha = alpha
        self.m = m

        self.range_mapping = get_range_mapping_fun(range_mapping_method)

    def forward(self, x, y):
        neg_img = x[0]
        learn_img = x[1]
        mask = x[2]
        pixel_count = x[3]

        s_p = torch.sum(torch.abs(learn_img - y) * mask, dim=[1, 2, 3]) / pixel_count
        s_n = torch.sum(torch.abs(learn_img - neg_img) * mask, dim=[1, 2, 3]) / pixel_count

        s_p = self.range_mapping(s_p)
        s_n = self.range_mapping(s_n)

        return torch.mean(torch.log(1 + torch.exp(self.gamma * (s_n + self.m - self.alpha * s_p))))

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight

def get_range_mapping_fun(range_adjust_method):
    if range_adjust_method == 'linear':
        def linear_mapping(x):
            return 1 - 2 * x
        return linear_mapping

    elif range_adjust_method == 'cosine':
        def cosine_mapping(x):
            return torch.cos(torch.pi * x)
        return cosine_mapping

    elif range_adjust_method == 'sigmoid':
        def sigmoid_mapping(x):
            return 2 / (1 + torch.exp(8 * (x - 0.5))) - 1
        return sigmoid_mapping

    elif range_adjust_method == 'quadratic':
        def quadratic_mapping(x):
            return -1 + 2 * (x - 1) ** 2
        return quadratic_mapping

    elif range_adjust_method == 'ellipse_ne':
        def ellipse_ne_mapping(x):
            return 1 + 2 * torch.sqrt(1.0001 - x ** 2)
        return ellipse_ne_mapping

    elif range_adjust_method == 'ellipse_sw':
        def ellipse_sw_mapping(x):
            return 1 - 2 * torch.sqrt(1.0001 - (x - 1) ** 2)
        return ellipse_sw_mapping

    elif range_adjust_method == 'cubic':
        def cubic_mapping(x):
            return -8 * x ** 3 + 12 * x ** 2 - 6 * x + 1
        return cubic_mapping

    elif range_adjust_method == 'recp':
        def recp_mapping(x):
            return -1 + 1 / (x + 0.5)
        return recp_mapping

# class L1ContrastiveLoss(LossWithWeights):
#     def __init__(self, gamma, alpha, m, loss_weight=1):
#         super(L1ContrastiveLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.L1Loss = nn.L1Loss()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.m = m
#
#     def forward(self, x, y):
#         neg_img = x[0]
#         learn_img = x[1]
#         mask = x[2]
#         pixel_count = x[3]
#
#         l_p = torch.sum(torch.abs(learn_img - y) * mask, dim=[1, 2, 3]) / pixel_count
#         l_n = torch.sum(torch.abs(learn_img - neg_img) * mask, dim=[1, 2, 3]) / pixel_count
#
#         return torch.mean(torch.log(1 + torch.exp(self.gamma * (l_p + self.m - self.alpha * l_n))))
#
#     def weight_from_epoch(self, epoch_index):
#         return self.loss_weight

class ContrastiveLossCosine(LossWithWeights):
    def __init__(self, gamma, alpha, m, loss_weight=1):
        super(ContrastiveLossCosine, self).__init__()
        self.loss_weight = loss_weight
        self.L1Loss = nn.L1Loss()
        self.gamma = gamma
        self.alpha = alpha
        self.m = m

    def forward(self, x, y):
        neg_img = x[0]
        learn_img = x[1]
        mask = x[2]
        pixel_count = x[3]

        s_p = torch.sum(torch.abs(learn_img - y) * mask, dim=[1, 2, 3]) / pixel_count
        s_n = torch.sum(torch.abs(learn_img - neg_img) * mask, dim=[1, 2, 3]) / pixel_count

        s_p = L1toNeg1to1Cosine(s_p)
        s_n = L1toNeg1to1Cosine(s_n)

        return torch.mean(torch.log(1 + torch.exp(self.gamma * (s_n + self.m - self.alpha * s_p))))

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class ContrastiveLossCosineSim(LossWithWeights):
    def __init__(self, gamma, alpha, m, loss_weight=1):
        super(ContrastiveLossCosineSim, self).__init__()
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.alpha = alpha
        self.m = m

    def forward(self, x, y):
        neg_img = x[0]
        learn_img = x[1]
        mask = x[2]
        pixel_count = x[3]

        masked_neg_img = neg_img * mask
        masked_y = y * mask
        masked_learn_img = learn_img * mask
        length_masked_neg_img = torch.sqrt(torch.sum(masked_neg_img * masked_neg_img, dim=[1, 2, 3]))
        length_masked_y = torch.sqrt(torch.sum(masked_y * masked_y, dim=[1, 2, 3]))
        length_masked_learn_img = torch.sqrt(torch.sum(masked_learn_img * masked_learn_img, dim=[1, 2, 3]))
        s_p = torch.sum(masked_learn_img * masked_y, dim=[1, 2, 3]) / length_masked_learn_img / length_masked_y
        s_n = torch.sum(masked_learn_img * masked_neg_img, dim=[1, 2, 3]) / length_masked_learn_img / length_masked_neg_img

        return torch.mean(torch.log(1 + torch.exp(self.gamma * (s_n + self.m - self.alpha * s_p))))

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class ContrastiveLossTwo(LossWithWeights):
    def __init__(self, gamma, alpha, m, loss_weight=1):
        super(ContrastiveLossTwo, self).__init__()
        self.loss_weight = loss_weight
        self.L1Loss = nn.L1Loss()
        self.gamma = gamma
        self.alpha = alpha
        self.m = m

    def forward(self, x, y):
        low_img = x[0]
        learn_img = x[1]
        paired_mask = x[2]
        unpaired_mask = 1 - paired_mask
        paired_pixel_count = torch.sum(paired_mask, dim=[1, 2, 3])
        unpaired_pixel_count = torch.sum(unpaired_mask, dim=[1, 2, 3])

        s_p = torch.sum(torch.abs(learn_img - y) * paired_mask, dim=[1, 2, 3]) / paired_pixel_count
        s_n = torch.sum(torch.abs(learn_img - low_img) * unpaired_mask, dim=[1, 2, 3]) / unpaired_pixel_count

        s_p = L1toNeg1to1(s_p)
        s_n = L1toNeg1to1(s_n)

        return torch.mean(torch.log(1 + torch.exp(self.gamma * (s_n + self.m - self.alpha * s_p))))

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class CharbonnierLoss(LossWithWeights):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3, loss_weight=1):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class IlluAdjustLoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, x_illu, y_img):
        y_mean = torch.mean(y_img, dim=[1, 2, 3]).view(-1, 1)

        return self.l1_loss(x_illu, y_mean)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class PSNRLoss(LossWithWeights):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class FullMAELoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.L1Loss = nn.L1Loss(reduction='mean')
        self.loss_weight = loss_weight

    def forward(self, x, y):
        # loss_sum = torch.sum(torch.abs((x[0] - y) * x[1])) / torch.sum(x[2])
        # masked_x = x[0] * x[1]
        # masked_gt = y * x[1]
        # loss_sum = self.L1LossSum(masked_x, masked_gt)

        return self.L1Loss(x, y)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight

class PSNRContrastiveLoss(LossWithWeights):
    def __init__(self, gamma, m, loss_weight=1):
        super(PSNRContrastiveLoss, self).__init__()
        self.loss_weight = loss_weight
        self.L2LossPos = nn.MSELoss(reduction='sum')
        self.L2LossNeg = nn.MSELoss(reduction='sum')
        beta = 0.2

        self.pow = 10 * beta * gamma
        self.factor = math.exp(m * gamma)

    def forward(self, x, y):
        neg_img = x[0]
        learn_img = x[1]
        mask = x[2]

        masked_learn_img = learn_img * mask

        loss = torch.log(1 + self.factor * torch.pow((self.L2LossPos(masked_learn_img, y * mask) / self.L2LossNeg(masked_learn_img, neg_img * mask)), self.pow))

        return torch.mean(loss)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class OddFullMAELoss(LossWithWeights):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.L1Loss = nn.L1Loss(reduction='mean')
        self.loss_weight = loss_weight

    def forward(self, x, y):
        # loss_sum = torch.sum(torch.abs((x[0] - y) * x[1])) / torch.sum(x[2])
        # masked_x = x[0] * x[1]
        # masked_gt = y * x[1]
        # loss_sum = self.L1LossSum(masked_x, masked_gt)

        return self.L1Loss(x[0], y)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight


class L1ContrastiveLoss(LossWithWeights):
    def __init__(self, neg_factor, loss_weight=1):
        super().__init__()
        self.neg_factor = neg_factor
        self.loss_weight = loss_weight

    def forward(self, x, y):
        neg_img = x[0]
        anchor_img = x[1]
        mask = x[2]
        # pixel_count = x[3]
        pos_img = y
        batch_size = x[0].shape[0]

        pos_sum = 0
        neg_sum = 0
        for i in range(batch_size):
            for j in range(batch_size):
                mult_mask = mask[i] * mask[j]
                pixel_count = torch.sum(mult_mask)
                pos_sum += torch.sum(torch.abs(anchor_img[i] - pos_img[j] * mult_mask)) / pixel_count
                neg_sum += torch.sum(torch.abs(anchor_img[i] - neg_img[j] * mult_mask)) / pixel_count

        return (pos_sum - self.neg_factor * neg_sum) / (batch_size * batch_size)

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight

class SSIMContrastiveLoss(LossWithWeights):
    def __init__(self, gamma, alpha, m, range_mapping_method, loss_weight=1):
        super(SSIMContrastiveLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss = piq.SSIMLoss().to("cuda")
        self.gamma = gamma
        self.alpha = alpha
        self.m = m

        self.range_mapping = get_range_mapping_fun(range_mapping_method)

    def forward(self, x, y):
        """
        Args:
            x: [0] -> negative sample / full image, [1] -> anchor, [2] -> mask, [3] -> pixel count(not used here)
            y: positive sample
        """
        mask = x[2]
        # Replace the area in Anchor that is removed by CutMix with full image which wasn't being cutmixed
        anchor = x[1] * mask + x[0] * (1 - mask)

        # Replace the area in Positive Sample that is removed by CutMix
        pos_sample = y * mask + x[0] * (1 - mask)

        neg_sample = x[0]

        anchor = torch.clamp(anchor, 0, 1)
        pos_sample = torch.clamp(pos_sample, 0, 1)
        neg_sample = torch.clamp(neg_sample, 0, 1)

        s_p = 1 - self.loss(anchor, pos_sample)
        s_n = 1 - self.loss(anchor, neg_sample)

        # s_p = self.range_mapping(s_p)
        # s_n = self.range_mapping(s_n)

        return torch.mean(torch.log(1 + torch.exp(self.gamma * (s_n + self.m - self.alpha * s_p))))

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight

class SSIMContrastiveLoss2(LossWithWeights):
    def __init__(self, gamma, alpha, m, range_mapping_method, loss_weight=1):
        super(SSIMContrastiveLoss2, self).__init__()
        self.loss_weight = loss_weight
        self.loss = piq.SSIMLoss().to("cuda")
        self.gamma = gamma
        self.alpha = alpha
        self.m = m

        self.range_mapping = get_range_mapping_fun(range_mapping_method)

    def forward(self, x, y):
        """
        Args:
            x: [0] -> negative sample, [1] -> anchor, [2] -> mask, [3] -> pixel count(not used here)
            y: positive sample / full image
        """
        mask = x[2]
        # Replace the area in Anchor that is removed by CutMix with full image which wasn't being cutmixed
        anchor = x[1] * mask + y * (1 - mask)

        pos_sample = y

        neg_sample = x[0] * mask + y * (1 - mask)

        anchor = torch.clamp(anchor, 0, 1)
        pos_sample = torch.clamp(pos_sample, 0, 1)
        neg_sample = torch.clamp(neg_sample, 0, 1)

        s_p = 1 - self.loss(anchor, pos_sample)
        s_n = 1 - self.loss(anchor, neg_sample)

        # s_p = self.range_mapping(s_p)
        # s_n = self.range_mapping(s_n)

        return torch.mean(torch.log(1 + torch.exp(self.gamma * (s_n + self.m - self.alpha * s_p))))

    def weight_from_epoch(self, epoch_index):
        return self.loss_weight

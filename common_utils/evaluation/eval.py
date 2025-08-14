import os
import argparse
from loguru import logger
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from PIL import Image
import numpy as np
import sys
sys.path.append(os.getcwd())

from common_utils.evaluation.metrics import PSNR, SSIM, LPIPS, NIQE, NIMA, UNIQUE, BRISQUE

from data.general_dataset import PairedEvalDataset, PairedSMIDEvalDataset

from utils.util import set_seed, tensor2img
from utils.network_loader import load_network

SEED = 1951581
RESULT_DIR = r'THE PATH OF LOG & RESULT DIR'
USE_AMP = True
QUANT = False

def get_dataset(dataset_type):
    low_dir = None
    normal_dir = None
    if dataset_type == 'LOLv2':
        low_dir = r'THE DIR PATH OF LOLv2 LOW-LIGHT IMAGES'
        normal_dir = r'THE DIR PATH OF LOLv2 NORMAL-LIGHT IMAGES'

    elif dataset_type == 'SID':
        low_dir = r'THE DIR PATH OF SID LOW-LIGHT IMAGES'
        normal_dir = r'THE DIR PATH OF SID NORMAL-LIGHT IMAGES'

    elif dataset_type == 'SDSD':
        low_dir = r'THE DIR PATH OF SDSD LOW-LIGHT IMAGES'
        normal_dir = r'THE DIR PATH OF SDSD NORMAL-LIGHT IMAGES'

    elif dataset_type == 'SMID':
        return PairedSMIDEvalDataset(r'THE DIR PATH OF SMID IMAGES IN NPY FORMAT')

    elif dataset_type == 'LSRW-Huawei':
        low_dir = r'THE DIR PATH OF LSRW-Huawei LOW-LIGHT IMAGES'
        normal_dir = r'THE DIR PATH OF LSRW-Huawei NORMAL-LIGHT IMAGES'

    return PairedEvalDataset(low_dir, normal_dir)


def get_init_metric():
    metric_value_dict = {
        'PSNR': [],
        'SSIM': [],
        'LPIPS': [],
        'NIQE': [],
        'NIMA': [],
        # 'UNIQUE': [],
        # 'BRISQUE': [],
    }

    metric_func_dict = {
        'PSNR': PSNR(),
        'SSIM': SSIM(),
        'LPIPS': LPIPS(),
        'NIQE': NIQE(),
        'NIMA': NIMA(),
        # 'UNIQUE': UNIQUE(),
        # 'BRISQUE': BRISQUE()
    }

    return metric_value_dict, metric_func_dict

def average(values: list):
    return sum(values) / len(values)

def eval_one_dataset(dataset_name, network_name, metric_func_dict, metric_value_dict):
    for metric_name in metric_value_dict:
        metric_value_dict[metric_name] = []

    dataset = get_dataset(dataset_name)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    logger.success(f'Evaluation dataset {dataset_name} loaded.')

    logger.info('Start evaluation...')
    with torch.no_grad():
        with open(os.path.join(RESULT_DIR, f'metric_{dataset_name}_{network_name}.csv'), 'w') as csv_file:
            # write headline
            csv_file.write('image,')
            for metric_name in metric_value_dict:
                csv_file.write(f'{metric_name},')
            csv_file.write('\n')

            for data in tqdm(dataloader):
                file_name = data['file_name'][0]
                input_img = data['teacher_input'].cuda()
                ground_truth = data['ground_truth'][0].cuda()

                with autocast(USE_AMP):
                    output_img = network(input_img)
                    output_img = torch.clamp(output_img, 0, 1).float()
                    
                if QUANT:
                    output_img = torch.round(output_img * 255).float() / 255.0

                assert output_img.shape[0] == 1

                output_img = output_img[0]

                if args.save:
                    if args.dataset == 'SMID':
                        Image.fromarray(
                            tensor2img(output_img, min_max=(0, 1))
                        ).save(
                            os.path.join(
                                RESULT_DIR, 'imgs',
                                file_name[-13:-9] + '_' + file_name[-8:-4] + '.png'
                            )
                        )
                    else:
                        Image.fromarray(
                            tensor2img(output_img, min_max=(0, 1))
                        ).save(
                            os.path.join(
                                RESULT_DIR, 'imgs',
                                os.path.basename(file_name)
                            )
                        )

                csv_file.write(f'{file_name},')
                for metric_name, metric_func in metric_func_dict.items():
                    metric_value = metric_func(output_img, ground_truth)

                    metric_value_dict[metric_name].append(metric_value)

                    csv_file.write(f'{metric_value:.8f},')

                csv_file.write('\n')

            logger.success('Evaluation finish.')

            # write final result and report
            csv_file.write('Average,')

            average_str = ''

            for metric_name, metric_value in metric_value_dict.items():
                csv_file.write(f'{average(metric_value):.8f},')

                logger.info(f'{metric_name}:\t{average(metric_value):.8f}')
                average_str += f'{average(metric_value):.8f},'

    logger.info(average_str)

if __name__ == '__main__':
    set_seed(SEED)
    logger.info('Start Execution...')

    os.makedirs(RESULT_DIR, exist_ok=True)

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', type=str, help='Path of checkpoint weight')
    parser.add_argument('-n', '--network', type=str, help='The network type')
    parser.add_argument('-d', '--dataset', type=str, help='The evaluation dataset, all for all dataset')
    parser.add_argument('-s', '--save', action='store_true', help='Save the output image')

    args = parser.parse_args()

    logger.add(os.path.join(RESULT_DIR, f'eval_{args.dataset}_{args.network}.log'))
    logger.info(f'USE_AMP: {USE_AMP}\tQUANT: {QUANT}')

    if args.save:
        os.makedirs(os.path.join(RESULT_DIR, 'imgs'), exist_ok=True)

    network = load_network(args.network, args.weight).cuda()
    network.eval()
    logger.success(f'Model loaded from {os.path.abspath(args.weight)}.')

    metric_value_dict, metric_func_dict = get_init_metric()

    if args.dataset != 'all':
        eval_one_dataset(args.dataset, args.network, metric_func_dict, metric_value_dict)
    else:
        # eval all dataset
        for dataset_name in ['LOLv2', 'SID', 'SDSD', 'SMID', 'LSRW-Huawei']:
            eval_one_dataset(dataset_name, args.network, metric_func_dict, metric_value_dict)

    logger.info('Execution end.')

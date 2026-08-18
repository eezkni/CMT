from loguru import logger
import torch
import os
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor
import argparse
import sys
sys.path.append(os.getcwd())


from common_utils.evaluation.metrics import PSNR, SSIM, LPIPS, NIQE, NIMA
from data.general_dataset import PairedSMIDEvalDataset


all_dataset_image_dir = r'/path/to/all_dataset_output'

gt_dir_dict = {
    'LOLv2': r'/path/to/test_data/LOLv2/normal',
    'SID': r'/path/to/test_data/SID/normal',
    'SDSD': r'/path/to/test_data/SDSD/normal',
    'SMID': r'/path/to/datasets/SMID',
    'Huawei': r'/path/to/datasets/LSRW-Huawei/test/high'
}


def get_init_metric():
    metric_value_dict = {
        'PSNR': [],
        'SSIM': [],
        'LPIPS': [],
        'NIQE': [],
        'NIMA': [],
    }

    metric_func_dict = {
        'PSNR': PSNR(),
        'SSIM': SSIM(),
        'LPIPS': LPIPS(),
        'NIQE': NIQE(),
        'NIMA': NIMA(),
    }

    return metric_value_dict, metric_func_dict

def average(values: list):
    return sum(values) / len(values)


def calc_iqa_one_dataset_general(dataset_name, metric_func_dict, metric_value_dict):
    logger.success(f'Evaluation dataset {dataset_name} loaded.')
    for metric_name in metric_value_dict:
        metric_value_dict[metric_name] = []

    gt_dir = gt_dir_dict[dataset_name]
    gt_image_list = os.listdir(gt_dir)

    to_tensor = ToTensor()

    for image in tqdm(gt_image_list):
        gt_image_path = os.path.join(gt_dir, image)

        output_image_path = os.path.join(all_dataset_image_dir, f'{dataset_name}_{image}')
        if not os.path.exists(output_image_path):
            output_image_path = output_image_path[:-3] + 'png'

            assert os.path.exists(output_image_path)

        gt_image = Image.open(gt_image_path, 'r').convert('RGB')
        output_image = Image.open(output_image_path, 'r').convert('RGB')

        gt_image = to_tensor(gt_image).cuda()
        output_image = to_tensor(output_image).cuda()

        for metric_name, metric_func in metric_func_dict.items():
            metric_value = metric_func(output_image, gt_image)
            metric_value_dict[metric_name].append(metric_value)

    average_str = ''
    for metric_name, metric_value in metric_value_dict.items():
        logger.info(f'{metric_name}:\t{average(metric_value):.8f}')
        average_str += f'{average(metric_value):.8f},'

    logger.info(average_str)

def calc_iqa_one_dataset_SMID(dataset_name, metric_func_dict, metric_value_dict):
    logger.success(f'Evaluation dataset {dataset_name} loaded.')
    assert dataset_name == 'SMID'

    for metric_name in metric_value_dict:
        metric_value_dict[metric_name] = []

    gt_dir = gt_dir_dict[dataset_name]
    dataset = PairedSMIDEvalDataset(gt_dir)

    to_tensor = ToTensor()

    for index in tqdm(range(len(dataset))):
        data = dataset[index]
        gt_image = data["ground_truth"].cuda()
        
        file_name = data["file_name"]
        file_name = file_name[-13:].replace(os.path.sep, '_').replace('npy', 'png')
        # print(file_name)
        output_image_path = os.path.join(all_dataset_image_dir, f'{dataset_name}_{file_name}')
        # print(output_image_path)
        if not os.path.exists(output_image_path):
            output_image_path = output_image_path[:-3] + 'png'

            assert os.path.exists(output_image_path)

        output_image = Image.open(output_image_path, 'r').convert('RGB')

        output_image = to_tensor(output_image).cuda()

        for metric_name, metric_func in metric_func_dict.items():
            metric_value = metric_func(output_image, gt_image)
            metric_value_dict[metric_name].append(metric_value)

    average_str = ''
    
    for metric_name, metric_value in metric_value_dict.items():
        logger.info(f'{metric_name}:\t{average(metric_value):.8f}')
        average_str += f'{average(metric_value):.8f},'

    logger.info(average_str)


if __name__ in "__main__":
    logger.info('Start Execution')

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str, help='The network type')

    args = parser.parse_args()

    metric_value_dict, metric_func_dict = get_init_metric()

    all_dataset_image_dir = os.path.join(all_dataset_image_dir, args.network)

    with torch.no_grad():
        for dataset_name in ['LOLv2', 'SID', 'SDSD', 'SMID', 'Huawei']:
            if dataset_name == 'SMID':
                calc_iqa_one_dataset_SMID(dataset_name, metric_func_dict, metric_value_dict)
            else:
                calc_iqa_one_dataset_general(dataset_name, metric_func_dict, metric_value_dict)

    logger.info('Execution end.')

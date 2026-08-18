import os

# Lab GPU setting
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch


import argparse
from loguru import logger
import utils.util as RetinolUtil
import data as RetinolData
import models as RetinolModel
from torch.utils.data import DataLoader, Subset


def main(main_options):
    # set seed and cuDNN environment
    torch.backends.cudnn.enabled = True

    if main_options['speed_up']['ena_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info('TF32 enabled')
    logger.warning('CuDNN for acceleration enabled by setting torch.backends.cudnn.enabled as True')
    RetinolUtil.set_seed(main_options['seed'])
    if main_options['use_non_deterministic']:
        torch.use_deterministic_algorithms(False)
        logger.warning('Use non deterministic algorithm')

    # set log file
    logger.add(main_options['path']['log_file'], level=main_options['log_level'])

    # set dataloader
    paired_loader, unpaired_loader, val_loaders_dict = RetinolData.create_dataloader(main_options)  # unpaired_loader and val_loader are None if phase is test.

    weight_paths = []
    if main_options['phase'] == 'test':
        if main_options['path']['resume_state'] is not None:
            if os.path.exists(main_options['path']['resume_state']):
                # multi weights to be evaluated
                file_list = os.listdir(main_options['path']['resume_state'])
                weight_prefixes = []
                for file_name in file_list:
                    split_result = os.path.basename(file_name).split('_')
                    if len(split_result) > 0:
                        weight_prefixes.append(split_result[0])

                weight_prefixes = list(set(weight_prefixes))
                weight_prefixes.sort()

                for weight_prefix in weight_prefixes:
                    weight_paths.append(os.path.join(main_options['path']['resume_state'], weight_prefix))
            else:
                # single weight
                weight_paths.append(main_options['path']['resume_state'])

            assert len(weight_paths) > 0
            main_options['path']['resume_state'] = weight_paths[0]

        else:
            logger.critical('resume_path is null')
            return

    # set model
    model = RetinolModel.create_model(main_options, paired_loader, unpaired_loader, val_loaders_dict)

    try:
        if main_options['phase'] == 'train':
            model.train(main_options)
        else:
            logger.info('There will be {} weights to be evaluated'.format(len(weight_paths)))
            for weight_path in weight_paths:
                model.load_from_disk(weight_path, main_options)
                model.test(main_options)
    finally:
        logger.info('End Execution')


if __name__ == '__main__':
    logger.info('Start Execution...')
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='Run train or test', default='train')
    parser.add_argument('-v', '--visual', action='store_true', help='Visualize with Visdom')
    parser.add_argument('-r', '--reproduce', type=int, default=-1, help='Reproduce target (Epoch)')

    args = parser.parse_args()

    options = RetinolUtil.parse_config_json(args)
    main(options)

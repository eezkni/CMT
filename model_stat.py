from collections import OrderedDict
import torch
from torchsummary import summary
from ptflops import get_model_complexity_info

from loguru import logger
import argparse
import utils.util as RetinolUtil
from models import create_network
import json

if __name__ == '__main__':
    logger.info('Start Statistics...')
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    parser.add_argument('-s', '--size', type=list, help='Size of input data', default=[3, 192, 192])

    args = parser.parse_args()

    json_str = ''
    with open(args.config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    option = json.loads(json_str, object_pairs_hook=OrderedDict)

    network = create_network(option, option['model']['which_networks'][0], ema=False).cuda()

    print(args.size)
    input_size = tuple(args.size)
    # stat
    summary(network, input_size)

    macs, params = get_model_complexity_info(network, input_size, as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

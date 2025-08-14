import torch
import numpy as np
import random
import json
from collections import OrderedDict
import os
from datetime import datetime
from pathlib import Path
import shutil
from loguru import logger
import importlib
from types import FunctionType
from functools import partial

valid_code_dirs = ['config', 'models', 'utils', 'data']


def set_seed(seed, gl_seed=0):
    """  set random seed, gl_seed used in worker_init_fn function """
    if seed >= 0 and gl_seed >= 0:
        seed += gl_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    ''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
        speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
    if seed >= 0 and gl_seed >= 0:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # torch.use_deterministic_algorithms(False)
        # logger.warning('Use non deterministic algorithm')
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def init_obj(opt, *args, default_file_name='default file', given_module=None, init_type='Network',
             **modify_kwargs):
    """
    finds a function handle with the name given as 'name' in config,
    and returns the instance initialized with corresponding args.
    """
    if opt is None or len(opt) < 1:
        logger.error('Option is None when initialize {}'.format(init_type))
        return None

    ''' default format is dict with name key '''
    if isinstance(opt, str):
        opt = {'name': opt}
        logger.warning('Config is a str, converts to a dict {}'.format(opt))

    name = opt['name']
    ''' name can be list, indicates the file and class name of function '''
    if isinstance(name, list):
        file_name, class_name = name[0], name[1]
    else:
        file_name, class_name = default_file_name, name
    try:
        if given_module is not None:
            module = given_module
        else:
            module = importlib.import_module(file_name)

        attr = getattr(module, class_name)
        kwargs = opt.get('args', {})
        kwargs.update(modify_kwargs)
        ''' import class or function with args '''
        if isinstance(attr, type):
            return_instance = attr(*args, **kwargs)
            return_instance.__name__ = return_instance.__class__.__name__
        elif isinstance(attr, FunctionType):
            return_instance = partial(attr, *args, **kwargs)
            return_instance.__name__ = attr.__name__
        else:
            logger.error("Unrecognized attribute {}".format(attr))
            return None

        logger.success('{} [{:s}() from {:s}] is created.'.format(init_type, class_name, file_name))
    except Exception as e:
        logger.critical('{} [{:s}() from {:s}] not recognized.'.format(init_type, class_name, file_name))
        print(e)
        raise NotImplementedError('{} [{:s}() from {:s}] not recognized.'.format(init_type, class_name, file_name))
    return return_instance


# --------------- Parse Part --------------- #


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def make_dirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def write_json(content, file_name):
    file_name = Path(file_name)
    with file_name.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class NoneDict(dict):
    def __missing__(self, key):
        return None


def convert_dict_to_nonedict(opt):
    """ convert to NoneDict, which return None for missing key. """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = convert_dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [convert_dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def parse_config_json(console_args):
    json_str = ''
    with open(console_args.config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # if console_args.batch is not None:
    #     opt['datasets'][opt['phase']]['dataloader']['args']['batch_size'] = console_args.batch

    ''' set experiment directory '''
    if console_args.reproduce > 0:
        experiments_root = os.path.join(opt['path']['base_dir'], '{}_{}_reproduce{}_{}'.format(console_args.phase, opt['name'], console_args.reproduce, get_timestamp()))
    else:
        experiments_root = os.path.join(opt['path']['base_dir'], '{}_{}_{}'.format(console_args.phase, opt['name'], get_timestamp()))
    make_dirs(experiments_root)

    ''' save json '''
    write_json(opt, '{}/config.json'.format(experiments_root))

    ''' change folder relative hierarchy to absolute path'''
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if path is None:
            continue
        if 'resume' not in key and 'base' not in key and 'root' not in key:
            if not os.path.isabs(opt['path'][key]):
                opt['path'][key] = os.path.join(experiments_root, path)
            if key != 'log_file' and key != 'iqa_csv':
                make_dirs(opt['path'][key])
            else:
                with open(opt['path'][key], 'w'):
                    pass

    ''' code backup '''
    for name in os.listdir('.'):
        if name in valid_code_dirs:
            shutil.copytree(name, os.path.join(opt['path']['code'], name),
                            ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
        if '.py' in name or '.sh' in name:
            shutil.copy(name, opt['path']['code'])

    opt.update({
        'phase': console_args.phase,
        'visual': console_args.visual
    })

    if console_args.reproduce > 0:
        opt['train'].update({
            'reproduce_target': console_args.reproduce
        })
    return convert_dict_to_nonedict(opt)


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.cpu()
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np + min_max[0]) * (255.0 / (min_max[1] - min_max[0]))).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()

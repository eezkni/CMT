import random

from torch.utils.data import DataLoader, Subset
from utils.util import init_obj, set_seed
from loguru import logger
from torch.utils.data.dataloader import Sampler
import numpy as np
from functools import partial
import torch
from torchvision import transforms


def create_datasets(options):
    """ loading Dataset() class from given file's name """
    paired_dataset_opt = options['datasets'][options['phase']]['which_dataset']['paired_dataset']
    unpaired_dataset_opt = options['datasets'][options['phase']]['which_dataset']['unpaired_dataset']
    valid_dataset_opt = options['datasets'][options['phase']]['which_dataset']['valid_datasets']

    if paired_dataset_opt is not None:
        paired_dataset = init_obj(paired_dataset_opt, default_file_name='data.dataset', init_type='Dataset')
        logger.info("Paired dataset has {} samples.".format(len(paired_dataset)))
    else:
        paired_dataset = None

    if unpaired_dataset_opt is not None:
        unpaired_dataset = init_obj(unpaired_dataset_opt, default_file_name='data.dataset', init_type='Dataset')
        logger.info("Unpaired dataset has {} samples.".format(len(unpaired_dataset)))
    else:
        unpaired_dataset = None

    if valid_dataset_opt is not None:
        val_datasets_dict = {}
        for name, valid_dataset_opt in valid_dataset_opt.items():
            val_datasets_dict[name] = init_obj(valid_dataset_opt, default_file_name='data.dataset', init_type='Dataset')
            logger.info("Valid dataset {} has {} samples.".format(name, len(val_datasets_dict[name])))
    else:
        val_datasets_dict = None

    return paired_dataset, unpaired_dataset, val_datasets_dict


def create_dataloader(options):
    paired_dataloader_args = options['datasets'][options['phase']]['dataloader']['paired_args']
    unpaired_dataloader_args = options['datasets'][options['phase']]['dataloader']['unpaired_args']
    val_dataloader_args = options['datasets'][options['phase']]['dataloader']['val_args']

    generator = torch.Generator()
    generator.manual_seed(options['seed'])

    paired_dataset, unpaired_dataset, val_datasets_dict = create_datasets(options)
    worker_init_fn = partial(set_seed, gl_seed=options['seed'])

    ''' create dataloader and validation dataloader '''
    if paired_dataset is not None:
        # paired_sampler = ReproducibleSampler(paired_dataset)
        paired_dataloader = DataLoader(paired_dataset, worker_init_fn=worker_init_fn, generator=generator, **paired_dataloader_args)
        logger.success('Paired Loader Created.')

        # unpaired_sampler = ReproducibleSampler(unpaired_dataset)
        unpaired_dataloader = DataLoader(unpaired_dataset, worker_init_fn=worker_init_fn, generator=generator, **unpaired_dataloader_args)
        logger.success('Unpaired Loader Created.')

    else:
        paired_dataloader = None
        unpaired_dataloader = None

    val_dataloaders_dict = {}
    for val_dataset_name, val_dataset_inst in val_datasets_dict.items():
        # val_sampler = ReproducibleSampler(val_dataset_inst)
        val_dataloaders_dict[val_dataset_name] = DataLoader(val_dataset_inst, worker_init_fn=worker_init_fn, generator=generator,
                                                            **val_dataloader_args)
    logger.success('Valid Loader Created.')
    return paired_dataloader, unpaired_dataloader, val_dataloaders_dict


def subset_split(dataset, split_index):
    """
    split a dataset into non-overlapping new datasets of given lengths.
    """
    total_len = len(dataset)

    Subsets = [
        Subset(dataset, range(0, split_index)),
        Subset(dataset, range(split_index, total_len))
    ]

    return Subsets


# todo
def RandomCutMix(student_data_input, teacher_data_input, mix_data_input, mix_area_proportion_param):
    """
    Returns:
        data_input: the cutmix result input
        actual_box_list : the bbox of pixels from mix_data_input
    """
    assert student_data_input.size()[0] == mix_data_input.size()[0]
    # Make clone
    cutmix_student_input = student_data_input.clone()
    cutmix_teacher_input = teacher_data_input.clone()
    paired_mask = torch.zeros_like(student_data_input)
    random_engine = random.Random()
    for i in range(student_data_input.size()[0]):
        mix_area_proportion = min(0.9, max(0.1, random_engine.gauss(mix_area_proportion_param, 1)))

        while True:
            bbx1, bby1, bbx2, bby2 = random_bbox(student_data_input.size(), mix_area_proportion)
            actual_proportion = (bbx2 - bbx1) * (bby2 - bby1) / (student_data_input.size()[2] * student_data_input.size()[3])

            if actual_proportion > 0:
                break
            else:
                logger.warning('Invalid Cutmix: proj: {}, actual: {}'.format(mix_area_proportion, actual_proportion))
        # actual_box_list.append([bbx1, bby1, bbx2, bby2])

        cutmix_student_input[i, :, bbx1:bbx2, bby1:bby2] = mix_data_input[i, :, bbx1:bbx2, bby1:bby2]
        cutmix_teacher_input[i, :, bbx1:bbx2, bby1:bby2] = mix_data_input[i, :, bbx1:bbx2, bby1:bby2]
        paired_mask[i, :, bbx1:bbx2, bby1:bby2] = 1.0

    # I'm not sure about whether these tensors will calculate gradient
    assert not cutmix_student_input.requires_grad and not cutmix_teacher_input.requires_grad
    return cutmix_student_input, cutmix_teacher_input, paired_mask

def random_bbox(size, mix_area_proportion):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(mix_area_proportion)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def RandomCutMixUniform(student_data_input, teacher_data_input, mix_data_input):
    """
    Returns:
        data_input: the cutmix result input
        actual_box_list : the bbox of pixels from mix_data_input
    """
    assert student_data_input.size()[0] == mix_data_input.size()[0]
    # Make clone
    cutmix_student_input = student_data_input.clone()
    cutmix_teacher_input = teacher_data_input.clone()
    paired_mask = torch.zeros_like(student_data_input)
    random_engine = random.Random()
    for i in range(student_data_input.size()[0]):
        mix_area_proportion = random_engine.uniform(0.1, 0.9)

        while True:
            bbx1, bby1, bbx2, bby2 = random_bbox(student_data_input.size(), mix_area_proportion)
            actual_proportion = (bbx2 - bbx1) * (bby2 - bby1) / (student_data_input.size()[2] * student_data_input.size()[3])

            if actual_proportion > 0:
                break
            else:
                logger.warning('Invalid Cutmix: proj: {}, actual: {}'.format(mix_area_proportion, actual_proportion))
        # actual_box_list.append([bbx1, bby1, bbx2, bby2])

        cutmix_student_input[i, :, bbx1:bbx2, bby1:bby2] = mix_data_input[i, :, bbx1:bbx2, bby1:bby2]
        cutmix_teacher_input[i, :, bbx1:bbx2, bby1:bby2] = mix_data_input[i, :, bbx1:bbx2, bby1:bby2]
        paired_mask[i, :, bbx1:bbx2, bby1:bby2] = 1.0

    # I'm not sure about whether these tensors will calculate gradient
    assert not cutmix_student_input.requires_grad and not cutmix_teacher_input.requires_grad
    return cutmix_student_input, cutmix_teacher_input, paired_mask


class ReproducibleSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return ReproducibleIter(self.__len__())

    def __len__(self):
        return len(self.data_source)


class ReproducibleIter:
    def __init__(self, data_count):
        self.data_count = data_count
        self.current_idx = 0
        self.shuffle_list = list(range(self.data_count))
        random.shuffle(self.shuffle_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx < self.data_count:
            element = self.shuffle_list[self.current_idx]
            self.current_idx += 1
            return element
        else:
            raise StopIteration

class InfinityIterator:
    def __init__(self, collection):
        assert len(collection) > 0

        self.collection = collection
        self.iter = iter(collection)

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.collection)
            return next(self.iter)

    def __len__(self):
        return len(self.collection)


def crop_image_batch(image_tensor, image_size):
    if image_size is None:
        return image_tensor

    if len(image_tensor.size()) <= 3:
        image_tensor = torch.unsqueeze(image_tensor, 0)
    center_crop = transforms.CenterCrop(image_size)

    return center_crop(image_tensor)

import torch.utils.data as data
import os
import random
import torch.nn.functional as F

import torchvision.io
from PIL import Image
from utils.util import tensor2img
from torchvision import transforms
import torch
import timm
import numpy as np
from torchvision.io import read_image
import cv2


from utils.util import set_seed

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.npy', '.webp'
]


def get_data_transform(image_size):
    return transforms.Compose([
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor()
    ])

class ZeroOneNormalize:
    def __call__(self, image):
        return image.float().div(255)

class LOLBasedDatasetPaired(data.Dataset):
    def __init__(self, paired_path, image_size):
        self.paired_files = list_image_path(os.path.join(paired_path, 'low'),
                                            os.path.join(paired_path, 'normal'))

        self.all_files = self.paired_files
        # random.shuffle(self.all_files)

        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        low_image = Image.open(self.all_files[index][0]).convert('RGB')

        file_name = os.path.basename(self.all_files[index][0])

        if self.all_files[index][1] is None:
            low_image = self.transform(low_image)
            return file_name, low_image, low_image
        else:
            high_image = Image.open(self.all_files[index][1]).convert('RGB')

            torch_rng = torch.random.get_rng_state()
            low_image = self.transform(low_image)

            torch.random.set_rng_state(torch_rng)
            high_image = self.transform(high_image)
            return {
                "file_name": file_name,
                "teacher_input": low_image,
                "student_input": low_image,
                "ground_truth": high_image
            }

    def __len__(self):
        return len(self.all_files)


class LOLPairedWithGPUOptim(data.Dataset):
    def __init__(self, paired_path, image_size):
        self.paired_files = list_image_path(os.path.join(paired_path, 'low'),
                                            os.path.join(paired_path, 'normal'))

        self.all_files = self.paired_files
        # random.shuffle(self.all_files)

        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            ZeroOneNormalize()
        ])

    def __getitem__(self, index):
        low_image = read_image(self.all_files[index][0], torchvision.io.ImageReadMode.RGB)
        low_image = low_image.to('cuda')

        file_name = os.path.basename(self.all_files[index][0])

        high_image = read_image(self.all_files[index][1], torchvision.io.ImageReadMode.RGB)
        high_image = high_image.to('cuda')

        torch_rng = torch.random.get_rng_state()
        torch_cuda_rng = torch.cuda.random.get_rng_state()
        low_image = self.transform(low_image)

        torch.random.set_rng_state(torch_rng)
        torch.cuda.random.set_rng_state(torch_cuda_rng)
        high_image = self.transform(high_image)
        return {
            "file_name": file_name,
            "teacher_input": low_image,
            "student_input": low_image,
            "ground_truth": high_image
        }

    def __len__(self):
        return len(self.all_files)

class PairedDatasetWithNoise(data.Dataset):
    def __init__(self, paired_path, reference_path, reference_ratio, image_size, noise_scale):
        self.paired_files = list_image_path(os.path.join(paired_path, 'low'),
                                            os.path.join(paired_path, 'normal'))
        self.reference_files = list_image_path(reference_path, None)
        random.shuffle(self.reference_files)

        reference_length = int(len(self.paired_files) * reference_ratio)
        self.reference_files = self.reference_files[:reference_length]

        self.all_files = self.paired_files + self.reference_files
        random.shuffle(self.all_files)

        self.image_size = image_size

        self.transform = get_data_transform(image_size)

        self.noise_scale = noise_scale

    def __getitem__(self, index):
        seed = torch.random.seed()

        torch.random.manual_seed(seed)
        low_image = load_image(self.all_files[index][0], self.transform)
        file_name = os.path.basename(self.all_files[index][0])

        if self.all_files[index][1] is None:
            return file_name, low_image, low_image
        else:
            student_noise = torch.clamp(torch.randn_like(low_image), -1, 1) * self.noise_scale
            torch.random.manual_seed(seed)
            high_image = load_image(self.all_files[index][1], self.transform)
            return {
                "file_name": file_name,
                "teacher_input": low_image,
                "student_input": low_image + student_noise,
                "ground_truth": high_image
            }

    def __len__(self):
        return len(self.all_files)


class LOLBasedDatasetUnpaired(data.Dataset):
    def __init__(self, unpaired_low_path, image_size):
        self.low_files = list_image_path(unpaired_low_path, None)

        random.shuffle(self.low_files)

        self.image_size = image_size

        self.transform = get_data_transform(image_size)

        self.input_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        low_image = load_image(self.low_files[index][0], self.transform)
        file_name = os.path.basename(self.low_files[index][0])

        return {
            "file_name": file_name,
            "teacher_input": low_image,
            "student_input": low_image
        }

    def __len__(self):
        return len(self.low_files)


def list_image_path(low_light_path, high_light_path):
    files = os.listdir(low_light_path)
    files.sort()
    image_files = []
    for file in files:
        if os.path.splitext(file)[-1] in IMG_EXTENSIONS:
            image_files.append([os.path.join(low_light_path, file),
                                os.path.join(high_light_path, file) if high_light_path is not None else None])

    return image_files


class PngEvalLoader(data.Dataset):
    def __init__(self, paired_path, image_list):
        all_paired_files = list_image_path(os.path.join(paired_path, 'low/teacher'),
                                           os.path.join(paired_path, 'normal'))

        self.paired_files, self.image_count = split_select_file(all_paired_files, image_list)
        
        self.full_set = False

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.image_size = None

    def __getitem__(self, index):
        low_image = load_image(self.paired_files[index][0], self.transform)
        file_name = os.path.basename(self.paired_files[index][0])

        high_image = load_image(self.paired_files[index][1], self.transform)

        return {
            "file_name": file_name,
            "teacher_input": low_image,
            "student_input": low_image,
            "ground_truth": high_image
        }

    def __len__(self):
        if self.full_set:
            return len(self.paired_files)
        else:
            return self.image_count


def load_image(path, transform):
    image = Image.open(path).convert('RGB')
    image = transform(image)

    return image

def random_crop_flip_images(image_low_npy, image_normal_npy, patch_size):
    height = image_low_npy.shape[0]
    width = image_low_npy.shape[1]

    if height - patch_size == 0:
        r = 0
        c = 0
    else:
        r = np.random.randint(0, height - patch_size)
        c = np.random.randint(0, width - patch_size)

    flip_hor = (np.random.randint(0, 2) == 0)
    flip_ver = (np.random.randint(0, 2) == 0)

    # crop
    image_low_npy = image_low_npy[r:r + patch_size, c:c + patch_size, :]

    # flip
    if flip_hor:
        image_low_npy = np.flip(image_low_npy, axis=1)
    if flip_ver:
        image_low_npy = np.flip(image_low_npy, axis=0)

    if image_normal_npy is not None:
        image_normal_npy = image_normal_npy[r:r + patch_size, c:c + patch_size, :]

        if flip_hor:
            image_normal_npy = np.flip(image_normal_npy, axis=1)
        if flip_ver:
            image_normal_npy = np.flip(image_normal_npy, axis=0)

    return image_low_npy.copy(), image_normal_npy.copy()


def get_random_mask(image_size, patch_size, mask_ratio):
    ps_c, ps_r = image_size[0] // patch_size, image_size[1] // patch_size
    mask_length = int((ps_c * ps_r) * mask_ratio)
    mask_indices = list(range(ps_c * ps_r))
    random.shuffle(mask_indices)
    mask_indices = mask_indices[:mask_length]

    mask = torch.ones(image_size)

    for index in mask_indices:
        raw_start = (index // ps_c) * patch_size
        col_start = (index % ps_c) * patch_size
        for r in range(raw_start, raw_start + patch_size):
            for c in range(col_start, col_start + patch_size):
                mask[r][c] = 0.0

    return mask.bool()


mask_value = 0.0


class PairedDatasetWithMask(data.Dataset):
    def __init__(self, paired_path, image_size, patch_size, mask_ratio):
        self.paired_files = list_image_path(os.path.join(paired_path, 'low'),
                                            os.path.join(paired_path, 'normal'))

        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.transform = get_data_transform(image_size)

    def __getitem__(self, index):
        seed = abs(torch.random.seed()) % 2**32

        set_seed(seed)
        raw_low_image = load_image(self.paired_files[index][0], self.transform)
        file_name = os.path.basename(self.paired_files[index][0])
        mask = get_random_mask(self.image_size, self.patch_size, self.mask_ratio)
        masked_low_image = torch.masked_fill(raw_low_image, mask, mask_value)

        set_seed(seed)
        high_image = load_image(self.paired_files[index][1], self.transform)
        return {
            "file_name": file_name,
            "teacher_input": raw_low_image,
            "student_input": masked_low_image,
            "ground_truth": high_image
        }

    def __len__(self):
        return len(self.paired_files)


class NumpyEvalLoader(data.Dataset):
    def __init__(self, paired_path, image_list):
        all_paired_files = list_image_path(
            os.path.join(paired_path, 'low'),
            os.path.join(paired_path, 'normal')
        )

        self.paired_files, self.image_count = split_select_file(all_paired_files, image_list)

        self.full_set = False

        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        image_low = self.transform(np.load(self.paired_files[index][0]))
        image_normal = self.transform(np.load(self.paired_files[index][1]))

        file_name = os.path.basename(self.paired_files[index][0])

        return {
            "file_name": file_name,
            "teacher_input": image_low,
            "student_input": image_low,
            "ground_truth": image_normal
        }

    def __len__(self):
        if self.full_set:
            return len(self.paired_files)
        else:
            return self.image_count


# class WebpEvalLoader(data.Dataset):
#     def __init__(self, paired_path, image_list):
#         all_paired_files = list_image_path(
#             os.path.join(paired_path, 'low'),
#             os.path.join(paired_path, 'normal')
#         )
#
#         self.paired_files, self.image_count = split_select_file(all_paired_files, image_list)
#
#         self.full_set = False
#
#         self.transform = transforms.ToTensor()
#
#     def __getitem__(self, index):
#         image_low = self.transform(webp.load_image(self.paired_files[index][0], 'RGB'))
#         image_normal = self.transform(webp.load_image(self.paired_files[index][1], 'RGB'))
#
#         file_name = os.path.basename(self.paired_files[index][0])
#
#         return {
#             "file_name": file_name,
#             "teacher_input": image_low,
#             "student_input": image_low,
#             "ground_truth": image_normal
#         }
#
#     def __len__(self):
#         if self.full_set:
#             return len(self.paired_files)
#         else:
#             return self.image_count


class UnpairedDatasetFromDirs(data.Dataset):
    def __init__(self, dir_list, image_size):
        super(UnpairedDatasetFromDirs, self).__init__()

        self.image_size = image_size

        self.to_tensor = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()
        ])

        self.low_files = []
        for dir_path in dir_list:
            student_path = os.path.join(dir_path, './low/student')
            teacher_path = os.path.join(dir_path, './low/teacher')
            student_list = os.listdir(student_path)
            student_list.sort()
            self.low_files += [[
                os.path.join(student_path, file_name),
                os.path.join(teacher_path, file_name),
            ] for file_name in student_list]

    def __getitem__(self, index):
        index = index % len(self.low_files)
        student_file_path = self.low_files[index][0]

        student_image = Image.open(student_file_path).convert('RGB')
        # student_image = np.array(student_image)

        # teacher_image = np.array(teacher_image)

        # student_image, teacher_image = random_crop_flip_images(student_image, teacher_image, self.image_size[0])

        # student_image = Image.fromarray(student_image, mode='RGB')
        # teacher_image = Image.fromarray(teacher_image, mode='RGB')
        student_image = self.to_tensor(student_image)

        file_name = os.path.basename(student_file_path)

        return {
            "file_name": file_name,
            "student_input": student_image,
            "teacher_input": student_image
        }

    def __len__(self):
        return len(self.low_files) * 4

class UnpairedDatasetFromDirsV2(data.Dataset):
    def __init__(self, dir_list, image_size):
        super(UnpairedDatasetFromDirsV2, self).__init__()

        self.image_size = image_size

        self.to_tensor = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()
        ])
        # self.noise_on_teacher = noise_on_teacher

        self.low_files = []
        for dir_path in dir_list:
            student_path = os.path.join(dir_path, './low/student')
            teacher_path = os.path.join(dir_path, './low/teacher')
            student_list = os.listdir(student_path)
            student_list.sort()
            self.low_files += [[
                os.path.join(student_path, file_name),
                os.path.join(teacher_path, file_name),
            ] for file_name in student_list]

    def __getitem__(self, index):
        index = index % len(self.low_files)
        student_file_path = self.low_files[index][0]
        teacher_file_path = self.low_files[index][1]

        student_image = Image.open(student_file_path).convert('RGB')
        # student_image = np.array(student_image)

        teacher_image = Image.open(teacher_file_path).convert('RGB')
        # teacher_image = np.array(teacher_image)

        # student_image, teacher_image = random_crop_flip_images(student_image, teacher_image, self.image_size[0])

        # student_image = Image.fromarray(student_image, mode='RGB')
        # teacher_image = Image.fromarray(teacher_image, mode='RGB')
        rng_state = torch.get_rng_state()
        student_image = self.to_tensor(student_image)

        torch.set_rng_state(rng_state)
        teacher_image = self.to_tensor(teacher_image)

        file_name = os.path.basename(student_file_path)

        # teacher_noise = torch.clamp(torch.randn_like(teacher_image) * self.noise_on_teacher, -1 * self.noise_on_teacher, 1 * self.noise_on_teacher)

        return {
            "file_name": file_name,
            "student_input": student_image,
            "teacher_input": teacher_image
        }

    def __len__(self):
        return len(self.low_files) * 4

def split_select_file(all_paired_files, image_list):
    if image_list is None:
        paired_files = all_paired_files
        image_count = len(all_paired_files)

    else:
        selected_paired_files = []
        unselected_paired_files = []

        for image_pair in all_paired_files:
            if os.path.basename(image_pair[0]) in image_list:
                selected_paired_files.append(image_pair)
            else:
                unselected_paired_files.append(image_pair)

        paired_files = selected_paired_files + unselected_paired_files

        image_count = len(selected_paired_files)

    return paired_files, image_count

class UnpairedDatasetFromDirsExpand(data.Dataset):
    def __init__(self, dir_list, image_size, expand_factor=4):
        super(UnpairedDatasetFromDirsExpand, self).__init__()

        self.image_size = image_size
        self.expand_factor = expand_factor

        self.to_tensor = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()
        ])
        # self.noise_on_teacher = noise_on_teacher

        self.low_files = []
        for dir_path in dir_list:
            student_path = os.path.join(dir_path, './low/student')
            teacher_path = os.path.join(dir_path, './low/teacher')
            student_list = os.listdir(student_path)
            student_list.sort()
            self.low_files += [[
                os.path.join(student_path, file_name),
                os.path.join(teacher_path, file_name),
            ] for file_name in student_list]

    def __getitem__(self, index):
        index = index % len(self.low_files)
        student_file_path = self.low_files[index][0]
        teacher_file_path = self.low_files[index][1]

        student_image = Image.open(student_file_path).convert('RGB')
        # student_image = np.array(student_image)

        teacher_image = Image.open(teacher_file_path).convert('RGB')
        # teacher_image = np.array(teacher_image)

        # student_image, teacher_image = random_crop_flip_images(student_image, teacher_image, self.image_size[0])

        # student_image = Image.fromarray(student_image, mode='RGB')
        # teacher_image = Image.fromarray(teacher_image, mode='RGB')
        rng_state = torch.get_rng_state()
        student_image = self.to_tensor(student_image)

        torch.set_rng_state(rng_state)
        teacher_image = self.to_tensor(teacher_image)

        file_name = os.path.basename(student_file_path)

        # teacher_noise = torch.clamp(torch.randn_like(teacher_image) * self.noise_on_teacher, -1 * self.noise_on_teacher, 1 * self.noise_on_teacher)

        return {
            "file_name": file_name,
            "student_input": student_image,
            "teacher_input": teacher_image
        }

    def __len__(self):
        return len(self.low_files) * self.expand_factor

class LOLBasedDatasetPairedDiff(data.Dataset):
    def __init__(self, paired_path, reference_path, reference_ratio, image_size):
        self.paired_files = list_image_path(os.path.join(paired_path, 'low'),
                                            os.path.join(paired_path, 'normal'))
        self.reference_files = list_image_path(reference_path, None)
        random.shuffle(self.reference_files)

        reference_length = int(len(self.paired_files) * reference_ratio)
        self.reference_files = self.reference_files[:reference_length]

        self.all_files = self.paired_files + self.reference_files
        random.shuffle(self.all_files)

        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        self.grayscale = transforms.Grayscale(num_output_channels=3)

    def __getitem__(self, index):
        low_image = Image.open(self.all_files[index][0]).convert('RGB')
        second_index = random.randint(0, len(self) - 1)
        low_image_2 = Image.open(self.all_files[second_index][0]).convert('RGB')

        file_name = os.path.basename(self.all_files[index][0])

        if self.all_files[index][1] is None:
            low_image = self.transform(low_image)
            return file_name, low_image, low_image
        else:
            high_image = Image.open(self.all_files[index][1]).convert('RGB')
            high_image_2 = Image.open(self.all_files[second_index][1]).convert('RGB')

            torch_rng = torch.random.get_rng_state()
            low_image = self.transform(low_image)
            low_image_2 = self.transform(low_image_2)

            torch.random.set_rng_state(torch_rng)
            high_image = self.transform(high_image)
            high_image_2 = self.transform(high_image_2)
            return {
                "file_name": file_name,
                "teacher_input": low_image,
                "student_input": low_image,
                "ground_truth": high_image,
                "low_input_2": low_image_2,
                "high_image_2": high_image_2
            }

    def __len__(self):
        return len(self.all_files)


class PairedDatasetWithExclude(data.Dataset):
    def __init__(self, paired_path, image_size, exclude_list):
        raw_paired_files = list_image_path(os.path.join(paired_path, 'low/teacher'),
                                            os.path.join(paired_path, 'normal'))
        self.paired_files = []

        for pair_image_path in raw_paired_files:
            if os.path.basename(pair_image_path[0]) not in exclude_list:
                self.paired_files.append(pair_image_path)

        random.shuffle(self.paired_files)

        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor()
        ])

        self.grayscale = transforms.Grayscale(num_output_channels=3)

    def __getitem__(self, index):
        low_image = Image.open(self.paired_files[index][0]).convert('RGB')

        file_name = os.path.basename(self.paired_files[index][0])

        high_image = Image.open(self.paired_files[index][1]).convert('RGB')

        torch_rng = torch.random.get_rng_state()
        low_image = self.transform(low_image)

        torch.random.set_rng_state(torch_rng)
        high_image = self.transform(high_image)
        return {
            "file_name": file_name,
            "teacher_input": low_image,
            "student_input": low_image,
            "ground_truth": high_image
        }

    def __len__(self):
        return len(self.paired_files)


class MixedPairedDataset(data.Dataset):
    def __init__(self, paired_1_path, paired_2_path, image_size, mix_dataset_size, exclude_list):
        # paired 2 (SID)
        raw_paired_files_2 = list_image_path(os.path.join(paired_2_path, 'low/student'),
                                            os.path.join(paired_2_path, 'normal'))
        self.paired_files = []

        for pair_image_path in raw_paired_files_2:
            if os.path.basename(pair_image_path[0]) not in exclude_list:
                self.paired_files.append(pair_image_path)

        self.paired_files = self.paired_files[:mix_dataset_size]

        # paired 1 (LOLv2)
        paired_files_1 = list_image_path(os.path.join(paired_1_path, 'low'),
                                            os.path.join(paired_1_path, 'normal'))
        self.paired_files += paired_files_1

        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor()
        ])

        self.grayscale = transforms.Grayscale(num_output_channels=3)

    def __getitem__(self, index):
        low_image = Image.open(self.paired_files[index][0]).convert('RGB')

        file_name = os.path.basename(self.paired_files[index][0])

        high_image = Image.open(self.paired_files[index][1]).convert('RGB')

        torch_rng = torch.random.get_rng_state()
        low_image = self.transform(low_image)

        torch.random.set_rng_state(torch_rng)
        high_image = self.transform(high_image)
        return {
            "file_name": file_name,
            "teacher_input": low_image,
            "student_input": low_image,
            "ground_truth": high_image
        }

    def __len__(self):
        return len(self.paired_files)


class ImageEvalLoaderWithPad(data.Dataset):
    def __init__(self, paired_path, image_list, pad):
        all_paired_files = list_image_path(os.path.join(paired_path, 'low/teacher'),
                                           os.path.join(paired_path, 'normal'))

        self.paired_files, self.image_count = split_select_file(all_paired_files, image_list)

        self.full_set = False
        self.pad = pad

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        low_image = load_image(self.paired_files[0][0], self.transform)
        self.image_size = (low_image.shape[1], low_image.shape[2])

    def __getitem__(self, index):
        low_image = load_image(self.paired_files[index][0], self.transform)
        file_name = os.path.basename(self.paired_files[index][0])

        high_image = load_image(self.paired_files[index][1], self.transform)

        h, w = low_image.shape[1], low_image.shape[2]
        H, W = ((h + self.pad) // self.pad) * self.pad, ((w + self.pad) // self.pad) * self.pad
        padh = H - h if h % self.pad != 0 else 0
        padw = W - w if w % self.pad != 0 else 0
        low_image = F.pad(low_image, (0, padw, 0, padh), 'reflect')
        high_image = F.pad(high_image, (0, padw, 0, padh), 'reflect')

        return {
            "file_name": file_name,
            "teacher_input": low_image,
            "student_input": low_image,
            "ground_truth": high_image
        }

    def __len__(self):
        if self.full_set:
            return len(self.paired_files)
        else:
            return self.image_count


class LOLBasedDatasetUnpairedNoPerturb(data.Dataset):
    def __init__(self, unpaired_low_path, image_size):
        self.low_files = list_image_path(unpaired_low_path, None)

        random.shuffle(self.low_files)

        self.image_size = image_size

        self.transform = get_data_transform(image_size)

    def __getitem__(self, index):
        index = index % len(self.low_files)
        low_image = load_image(self.low_files[index][0], self.transform)
        file_name = os.path.basename(self.low_files[index][0])

        return {
            "file_name": file_name,
            "teacher_input": low_image,
            "student_input": low_image
        }

    def __len__(self):
        return len(self.low_files) * 4


class SinglePngEvalLoader(data.Dataset):
    def __init__(self, image_path, image_list):
        all_paired_files = list_image_path(image_path, None)

        self.all_files, self.image_count = split_select_file(all_paired_files, image_list)

        self.full_set = False

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.image_size = None

    def __getitem__(self, index):
        low_image = load_image(self.all_files[index][0], self.transform)
        file_name = os.path.basename(self.all_files[index][0])

        # high_image = load_image(self.all_files[index][1], self.transform)

        return {
            "file_name": file_name,
            "teacher_input": low_image,
            "student_input": low_image,
            "ground_truth": low_image
        }

    def __len__(self):
        if self.full_set:
            return len(self.all_files)
        else:
            return self.image_count


class UnpairedDatasetFromDirsNoPerturb(data.Dataset):
    def __init__(self, dir_list, image_size):
        super(UnpairedDatasetFromDirsNoPerturb, self).__init__()

        self.image_size = image_size

        self.to_tensor = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()
        ])

        self.low_files = []
        for dir_path in dir_list:
            student_path = os.path.join(dir_path, './low/student')
            teacher_path = os.path.join(dir_path, './low/teacher')
            student_list = os.listdir(student_path)
            student_list.sort()
            self.low_files += [[
                os.path.join(student_path, file_name),
                os.path.join(teacher_path, file_name),
            ] for file_name in student_list]

    def __getitem__(self, index):
        index = index % len(self.low_files)
        student_file_path = self.low_files[index][0]

        student_image = Image.open(student_file_path).convert('RGB')

        student_image = self.to_tensor(student_image)

        file_name = os.path.basename(student_file_path)

        return {
            "file_name": file_name,
            "student_input": student_image,
            "teacher_input": student_image
        }

    def __len__(self):
        return len(self.low_files) * 4


class SingleUnpairedDatasetFromDirsSaltPepperNoise(data.Dataset):
    def __init__(self, dir_list, image_size, noise_on_student):
        super(SingleUnpairedDatasetFromDirsSaltPepperNoise, self).__init__()

        self.image_size = image_size

        self.to_tensor = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()
        ])
        self.noise_on_student = noise_on_student

        self.low_files = []
        for dir_path in dir_list:
            student_path = dir_path
            student_list = os.listdir(student_path)
            student_list.sort()
            self.low_files += [os.path.join(student_path, file_name) for file_name in student_list]

    def __getitem__(self, index):
        index = index % len(self.low_files)
        student_file_path = self.low_files[index]

        student_image = Image.open(student_file_path).convert('RGB')
        # student_image = np.array(student_image)

        # teacher_image = Image.open(teacher_file_path).convert('RGB')
        # teacher_image = np.array(teacher_image)

        # student_image, teacher_image = random_crop_flip_images(student_image, teacher_image, self.image_size[0])

        # student_image = Image.fromarray(student_image, mode='RGB')
        # teacher_image = Image.fromarray(teacher_image, mode='RGB')
        # rng_state = torch.get_rng_state()
        student_image = self.to_tensor(student_image)

        # torch.set_rng_state(rng_state)
        # teacher_image = self.to_tensor(teacher_image)

        file_name = os.path.basename(student_file_path)

        student_noise = salt_pepper_noise_like(student_image, 0.05)

        return {
            "file_name": file_name,
            "student_input": student_image + student_noise * self.noise_on_student,
            "teacher_input": student_image
        }

    def __len__(self):
        return len(self.low_files) * 4


def salt_pepper_noise_like(template_tensor, threshold):
    noise = torch.zeros_like(template_tensor)
    mask = torch.rand_like(template_tensor)
    mask_pepper = mask.lt(threshold)
    noise[mask_pepper] = 1

    mask_salt = mask.gt(1 - threshold)
    noise[mask_salt] = 0

    return noise


class SingleUnpairedDatasetFromDirsNormalNoise(data.Dataset):
    def __init__(self, dir_list, image_size, noise_on_student):
        super(SingleUnpairedDatasetFromDirsNormalNoise, self).__init__()

        self.image_size = image_size

        self.to_tensor = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()
        ])
        self.noise_on_student = noise_on_student

        self.low_files = []
        for dir_path in dir_list:
            student_path = dir_path
            student_list = os.listdir(student_path)
            student_list.sort()
            self.low_files += [os.path.join(student_path, file_name) for file_name in student_list]

    def __getitem__(self, index):
        index = index % len(self.low_files)
        student_file_path = self.low_files[index]

        student_image = Image.open(student_file_path).convert('RGB')
        # student_image = np.array(student_image)

        # teacher_image = Image.open(teacher_file_path).convert('RGB')
        # teacher_image = np.array(teacher_image)

        # student_image, teacher_image = random_crop_flip_images(student_image, teacher_image, self.image_size[0])

        # student_image = Image.fromarray(student_image, mode='RGB')
        # teacher_image = Image.fromarray(teacher_image, mode='RGB')
        # rng_state = torch.get_rng_state()
        student_image = self.to_tensor(student_image)

        # torch.set_rng_state(rng_state)
        # teacher_image = self.to_tensor(teacher_image)

        file_name = os.path.basename(student_file_path)

        student_noise = torch.randn_like(student_image)

        return {
            "file_name": file_name,
            "student_input": student_image + student_noise * self.noise_on_student,
            "teacher_input": student_image
        }

    def __len__(self):
        return len(self.low_files) * 4


class UnpairedDatasetRandomBrightness(data.Dataset):
    def __init__(self, dir_list, image_size, random_factor=0.1):
        super(UnpairedDatasetRandomBrightness, self).__init__()

        self.image_size = image_size

        self.to_tensor = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()
        ])
        # self.noise_on_teacher = noise_on_teacher

        self.low_files = []
        for dir_path in dir_list:
            file_list = os.listdir(dir_path)
            file_list.sort()
            self.low_files += [os.path.join(dir_path, file_name) for file_name in file_list]

        self.random_factor = random_factor

    def __getitem__(self, index):
        index = index % len(self.low_files)
        file_path = self.low_files[index]

        raw_image = Image.open(file_path).convert('RGB')

        raw_image = self.to_tensor(raw_image)

        file_name = os.path.basename(file_path)

        bright_image_factor = random.uniform(-1 * self.random_factor, self.random_factor) + 1

        return {
            "file_name": file_name,
            "student_input": raw_image * bright_image_factor,
            "teacher_input": raw_image
        }

    def __len__(self):
        return len(self.low_files) * 4


class PairedDatasetRandomBrightness(data.Dataset):
    def __init__(self, paired_path, image_size, random_factor):
        self.paired_files = list_image_path(os.path.join(paired_path, 'low'),
                                            os.path.join(paired_path, 'normal'))

        self.all_files = self.paired_files
        # random.shuffle(self.all_files)

        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

        self.random_factor = random_factor

    def __getitem__(self, index):
        low_image = Image.open(self.all_files[index][0]).convert('RGB')

        file_name = os.path.basename(self.all_files[index][0])

        high_image = Image.open(self.all_files[index][1]).convert('RGB')

        torch_rng = torch.random.get_rng_state()
        low_image = self.transform(low_image)

        torch.random.set_rng_state(torch_rng)
        high_image = self.transform(high_image)
        return {
            "file_name": file_name,
            "teacher_input": low_image,
            "student_input": low_image * random.uniform(-1 * self.random_factor, self.random_factor) + 1,
            "ground_truth": high_image
        }

    def __len__(self):
        return len(self.all_files)


class NPYUnpairedDataset(data.Dataset):
    def __init__(self, dir_list, image_size):
        super(NPYUnpairedDataset, self).__init__()

        self.image_size = image_size

        self.to_tensor = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.ColorJitter(brightness=0.1, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()
        ])
        # self.noise_on_teacher = noise_on_teacher

        self.low_files = []
        for dir_path in dir_list:
            student_path = os.path.join(dir_path, './low')
            student_list = os.listdir(student_path)
            student_list.sort()
            self.low_files += [os.path.join(student_path, file_name) for file_name in student_list]

    def __getitem__(self, index):
        index = index % len(self.low_files)
        student_file_path = self.low_files[index]

        npy_data = np.load(student_file_path)

        npy_data = cv2.cvtColor(npy_data, cv2.COLOR_BGR2RGB)
        student_image = Image.fromarray(npy_data)

        student_image = self.to_tensor(student_image)

        file_name = os.path.basename(student_file_path)

        return {
            "file_name": file_name,
            "student_input": student_image,
            "teacher_input": student_image
        }

    def __len__(self):
        return len(self.low_files) * 4


class NPYEvalLoader(data.Dataset):
    def __init__(self, paired_path, image_list):
        all_paired_files = list_image_path(os.path.join(paired_path, 'low'),
                                           os.path.join(paired_path, 'normal'))

        self.paired_files, self.image_count = split_select_file(all_paired_files, image_list)

        self.full_set = False

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.image_size = None

    def __getitem__(self, index):
        npy_data = np.load(self.paired_files[index][0])

        npy_data = cv2.cvtColor(npy_data, cv2.COLOR_BGR2RGB)
        low_image = Image.fromarray(npy_data)

        low_image = self.transform(low_image)

        file_name = os.path.basename(self.paired_files[index][0])

        npy_data = np.load(self.paired_files[index][1])

        npy_data = cv2.cvtColor(npy_data, cv2.COLOR_BGR2RGB)
        high_image = Image.fromarray(npy_data)

        high_image = self.transform(high_image)

        return {
            "file_name": file_name,
            "teacher_input": low_image,
            "student_input": low_image,
            "ground_truth": high_image
        }

    def __len__(self):
        if self.full_set:
            return len(self.paired_files)
        else:
            return self.image_count


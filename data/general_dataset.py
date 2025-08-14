import torch.utils.data as data
import os
import random
import numpy as np
from torchvision import transforms
import torch
from PIL import Image

def scan_dir(prefix, path_list: list):
    file_list = os.listdir(prefix)
    file_list.sort()

    for f in file_list:
        current_path = os.path.join(prefix, f)
        if os.path.isdir(current_path):
            scan_dir(current_path, path_list)
        else:
            if current_path.lower().endswith(('.jpg', '.jpeg', '.png', '.npy')):
                path_list.append(current_path)

class PairedTrainDataset(data.Dataset):
    def __init__(self, low_dir, normal_dir, patch_size):
        self.low_image_list = []
        scan_dir(os.path.abspath(low_dir), self.low_image_list)

        self.normal_image_list = []
        scan_dir(os.path.abspath(normal_dir), self.normal_image_list)

        assert len(self.low_image_list) == len(self.normal_image_list)

        self.patch_size = patch_size

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        img_low = Image.open(self.low_image_list[index]).convert('RGB')
        img_normal = Image.open(self.normal_image_list[index]).convert('RGB')

        width, height = img_low.size

        left = random.randint(0, width - self.patch_size[0])
        top = random.randint(0, height - self.patch_size[1])

        box = (left, top, left + self.patch_size[0], top + self.patch_size[1])

        img_low = img_low.crop(box)
        img_normal = img_normal.crop(box)

        img_low = self.to_tensor(img_low)
        img_normal = self.to_tensor(img_normal)

        ver_flip = random.choice([True, False])
        hor_flip = random.choice([True, False])

        flip_dim_list = []
        if ver_flip:
            flip_dim_list.append(2)

        if hor_flip:
            flip_dim_list.append(1)

        if len(flip_dim_list) != 0:
            img_low = torch.flip(img_low, flip_dim_list)
            img_normal = torch.flip(img_normal, flip_dim_list)

        return {
            "teacher_input": img_low,
            "student_input": img_low,
            "ground_truth": img_normal
        }

    def __len__(self):
        return len(self.low_image_list)


class PairedEvalDataset(data.Dataset):
    def __init__(self, low_dir, normal_dir):
        self.low_image_list = []
        scan_dir(os.path.abspath(low_dir), self.low_image_list)

        self.normal_image_list = []
        scan_dir(os.path.abspath(normal_dir), self.normal_image_list)

        assert len(self.low_image_list) == len(self.normal_image_list)

        self.to_tensor = transforms.ToTensor()
        self.image_size = None

    def __getitem__(self, index):
        img_low = Image.open(self.low_image_list[index]).convert('RGB')
        img_normal = Image.open(self.normal_image_list[index]).convert('RGB')

        img_low = self.to_tensor(img_low)
        img_normal = self.to_tensor(img_normal)

        return {
            "file_name": self.low_image_list[index],
            "teacher_input": img_low,
            "student_input": img_low,
            "ground_truth": img_normal
        }

    def __len__(self):
        return len(self.low_image_list)


class MultiDirUnpairedTrainDataset(data.Dataset):
    def __init__(self, dir_list, patch_size):
        super(MultiDirUnpairedTrainDataset, self).__init__()

        self.patch_size = patch_size

        self.trans = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor()
        ])

        self.low_files = []
        for dir_path in dir_list:
            scan_dir(dir_path, self.low_files)

    def __getitem__(self, index):
        index = index % len(self.low_files)
        file_path = self.low_files[index]

        img_low = Image.open(file_path).convert('RGB')

        img_low = self.trans(img_low)

        file_name = os.path.basename(file_path)

        return {
            "file_name": file_name,
            "teacher_input": img_low,
            "student_input": img_low,
        }

    def __len__(self):
        return len(self.low_files) * 10


class PairedSMIDEvalDataset(data.Dataset):
    def __init__(self, root_dir, full=False):
        self.low_image_list = []
        self.normal_image_list = []

        with open(os.path.join(root_dir, 'test_list.txt'), 'r') as f:
            test_img_list = f.readlines()

        # remove last \n
        for i in range(len(test_img_list)):
            if test_img_list[i][-1] == '\n':
                test_img_list[i] = test_img_list[i][:-1]

        cnt = 0
        for img_name in test_img_list:
            normal_img_path = os.path.join(root_dir, 'high_np', img_name)
            normal_imgs = os.listdir(normal_img_path)
            assert len(normal_imgs) == 1
            normal_img_path = os.path.join(normal_img_path, normal_imgs[0])

            low_imgs_path = os.path.join(root_dir, 'low_np', img_name)
            if full:
                for low_img_name in os.listdir(low_imgs_path):
                    low_img_path = os.path.join(low_imgs_path, low_img_name)
                    self.low_image_list.append(low_img_path)
                    self.normal_image_list.append(normal_img_path)
            else:
                # only use one image
                # low_img_list = os.listdir(low_imgs_path)
                # img_idx = cnt % len(low_img_list)
                mapping_dict = {
                    "0007": "0001",
                    "0009": "0066",
                    "0010": "0002",
                    "0012": "0003",
                    "0013": "0035",
                    "0020": "0089",
                    "0036": "0005",
                    "0037": "0006",
                    "0039": "0007",
                    "0047": "0008",
                    "0048": "0009",
                    "0049": "0010",
                    "0050": "0011",
                    "0051": "0012",
                    "0052": "0013",
                    "0065": "0014",
                    "0066": "0015",
                    "0075": "0016",
                    "0076": "0017",
                    "0078": "0018",
                    "0083": "0067",
                    "0088": "0019",
                    "0091": "0022",
                    "0092": "0020",
                    "0099": "0021",
                    "0103": "0023",
                    "0104": "0024",
                    "0105": "0025",
                    "0106": "0026",
                    "0107": "0027",
                    "0129": "0032",
                    "0139": "0031",
                    "0145": "0029",
                    "0147": "0054",
                    "0151": "0028",
                    "0153": "0068",
                    "0154": "0004",
                    "0157": "0069",
                    "0166": "0062",
                    "0167": "0047",
                    "0169": "0048",
                    "0170": "0043",
                    "0172": "0066",
                    "0175": "0071",
                    "0177": "0046",
                    "0180": "0057",
                    "0181": "0059",
                    "0191": "0027",
                    "0196": "0059",
                }
                low_img_path = os.path.join(low_imgs_path, mapping_dict[img_name] + '.npy')
                self.low_image_list.append(low_img_path)
                self.normal_image_list.append(normal_img_path)

                cnt += 1

        self.to_tensor = transforms.ToTensor()
        self.image_size = None

    def __getitem__(self, index):
        img_low = Image.fromarray(np.load(self.low_image_list[index])).convert('RGB')
        img_normal = Image.fromarray(np.load(self.normal_image_list[index])).convert('RGB')

        img_low = self.to_tensor(img_low)
        img_normal = self.to_tensor(img_normal)

        # produce green channel
        img_low[1] = img_low[1] / 1.5

        return {
            "file_name": self.low_image_list[index],
            "teacher_input": img_low,
            "student_input": img_low,
            "ground_truth": img_normal
        }

    def __len__(self):
        return len(self.low_image_list)

class PairedSMIDTrainDataset(data.Dataset):
    def __init__(self, root_dir, patch_size):
        self.low_image_list = []
        self.normal_image_list = []
        self.patch_size = patch_size

        with open(os.path.join(root_dir, 'test_list.txt'), 'r') as f:
            test_img_list = f.readlines()

        # remove last \n
        for i in range(len(test_img_list)):
            if test_img_list[i][-1] == '\n':
                test_img_list[i] = test_img_list[i][:-1]

        for img_name in os.listdir(os.path.join(root_dir, 'high_np')):
            if img_name not in test_img_list:
                normal_img_path = os.path.join(root_dir, 'high_np', img_name)
                normal_imgs = os.listdir(normal_img_path)
                assert len(normal_imgs) == 1
                normal_img_path = os.path.join(normal_img_path, normal_imgs[0])

                low_imgs_path = os.path.join(root_dir, 'low_np', img_name)
                for low_img_name in os.listdir(low_imgs_path):
                    low_img_path = os.path.join(low_imgs_path, low_img_name)
                    self.low_image_list.append(low_img_path)
                    self.normal_image_list.append(normal_img_path)

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        img_low = Image.fromarray(np.load(self.low_image_list[index])).convert('RGB')
        img_normal = Image.fromarray(np.load(self.normal_image_list[index])).convert('RGB')

        width, height = img_low.size

        left = random.randint(0, width - self.patch_size[0])
        top = random.randint(0, height - self.patch_size[1])

        box = (left, top, left + self.patch_size[0], top + self.patch_size[1])

        img_low = img_low.crop(box)
        img_normal = img_normal.crop(box)

        img_low = self.to_tensor(img_low)
        img_normal = self.to_tensor(img_normal)

        ver_flip = random.choice([True, False])
        hor_flip = random.choice([True, False])

        flip_dim_list = []
        if ver_flip:
            flip_dim_list.append(2)

        if hor_flip:
            flip_dim_list.append(1)

        if len(flip_dim_list) != 0:
            img_low = torch.flip(img_low, flip_dim_list)
            img_normal = torch.flip(img_normal, flip_dim_list)

        # produce green channel
        img_low[1] = img_low[1] / 1.5

        return {
            "teacher_input": img_low,
            "student_input": img_low,
            "ground_truth": img_normal
        }

    def __len__(self):
        return len(self.low_image_list)

class UnpairedSMIDTrainDataset(data.Dataset):
    def __init__(self, dir_list, patch_size):
        super(UnpairedSMIDTrainDataset, self).__init__()

        self.patch_size = patch_size

        self.trans = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor()
        ])

        self.low_files = []
        for dir_path in dir_list:
            scan_dir(dir_path, self.low_files)

    def __getitem__(self, index):
        index = index % len(self.low_files)
        file_path = self.low_files[index]

        img_low = Image.fromarray(np.load(file_path)).convert('RGB')

        img_low = self.trans(img_low)
        # produce green channel
        img_low[1] = img_low[1] / 1.5

        file_name = os.path.basename(file_path)

        return {
            "file_name": file_name,
            "teacher_input": img_low,
            "student_input": img_low,
        }

    def __len__(self):
        return len(self.low_files)

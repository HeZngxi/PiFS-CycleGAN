import os
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import re


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'
        self.dir_A_label = os.path.join(opt.dataroot, opt.phase + "ALabel")  # create a path for label images

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_label_paths = sorted(make_dataset(self.dir_A_label, opt.max_dataset_size))  # load label images
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.A_label_size = len(self.A_label_paths)  # get the size of label dataset
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.input_nc = input_nc
        self.output_nc = output_nc
        
        # 创建基础变换（不包括随机裁剪）
        transform_list = []
        if 'resize' in opt.preprocess:
            # 修改为支持不同长宽的resize
            transform_list.append(transforms.Resize([opt.load_size_h, opt.load_size_w], interpolation=transforms.InterpolationMode.BICUBIC))
        self.base_transform = transforms.Compose(transform_list)
            
        # 创建标签图像的变换（使用与CT图像相同的bicubic插值）
        transform_list_label = []
        if 'resize' in opt.preprocess:
            transform_list_label.append(transforms.Resize([opt.load_size_h, opt.load_size_w], interpolation=transforms.InterpolationMode.BICUBIC))
        self.base_transform_label = transforms.Compose(transform_list_label)
            
        # 创建共享的随机变换参数
        self.use_random_crop = 'crop' in opt.preprocess
        # 修改为支持不同长宽的随机裁剪
        self.crop_h = opt.crop_size_h
        self.crop_w = opt.crop_size_w
        self.use_flip = not opt.no_flip

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        # 加载图像并确保是灰度图
        A_img = Image.open(A_path).convert("L")  # 直接转换为灰度图
        B_img = Image.open(B_path).convert("L")  # 直接转换为灰度图
        
        # Load corresponding label image for A
        # Generate label path from A_path with new naming convention
        # CT image: patientxxx-trainCT-x.png
        # Label image: patientxxx-sim-global-x-simulated.png
        dir_name = os.path.dirname(A_path)
        parent_dir = os.path.dirname(dir_name)
        base_name = os.path.basename(A_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Extract patient ID and x from CT image filename
        ct_match = re.match(r'patient(\d+)-trainCT-(\d+)', name_without_ext)
        if ct_match:
            patient_id = ct_match.group(1)
            x_value = ct_match.group(2)
            label_name = f'patient{patient_id}-sim-global-{x_value}-simulated.png'
            label_dir_name = "trainALabel" if "trainA" in dir_name else "trainBLabel"
            A_label_path = os.path.join(parent_dir, label_dir_name, label_name)
        else:
            # fallback to old naming convention if regex doesn't match
            label_name = name_without_ext + "-label.png"
            label_dir_name = "trainALabel" if "trainA" in dir_name else "trainBLabel"
            A_label_path = os.path.join(parent_dir, label_dir_name, label_name)
        
        # 应用基础变换（resize）
        A_img = self.base_transform(A_img)
        B_img = self.base_transform(B_img)
        if os.path.exists(A_label_path):
            A_label_img = Image.open(A_label_path).convert("L")  # Load as grayscale
            A_label_img = self.base_transform_label(A_label_img)
        else:
            A_label_img = None
        
        # 生成共享的随机变换参数
        if self.use_random_crop:
            # 生成随机裁剪参数，支持不同长宽
            w, h = A_img.size  # 获取图像尺寸
            # 确保裁剪尺寸不超过图像尺寸
            crop_h = min(self.crop_h, h)
            crop_w = min(self.crop_w, w)
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            # 对所有图像应用相同的随机裁剪
            A_img = F.crop(A_img, top, left, crop_h, crop_w)
            B_img = F.crop(B_img, top, left, crop_h, crop_w)
            if A_label_img is not None:
                A_label_img = F.crop(A_label_img, top, left, crop_h, crop_w)
        
        # 应用随机水平翻转
        if self.use_flip and random.random() > 0.5:
            A_img = F.hflip(A_img)
            B_img = F.hflip(B_img)
            if A_label_img is not None:
                A_label_img = F.hflip(A_label_img)
        
        # 应用最终的变换（转换为张量并归一化）
        A = F.to_tensor(A_img)
        B = F.to_tensor(B_img)
        A = F.normalize(A, (0.5,), (0.5,))
        B = F.normalize(B, (0.5,), (0.5,))
        
        if A_label_img is not None:
            A_label = F.to_tensor(A_label_img)
            A_label = F.normalize(A_label, (0.5,), (0.5,))
        else:
            # If label doesn't exist, create a zero tensor with the same shape as A after crop
            A_label = torch.zeros((1, self.crop_h if self.use_random_crop else A.size[1], self.crop_w if self.use_random_crop else A.size[2]), dtype=torch.float32)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path, "A_label": A_label}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
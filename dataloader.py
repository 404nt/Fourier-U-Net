from tkinter import image_names
import cv2
import matplotlib.pyplot as plt
import glob, os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class DatasetGenerate(Dataset):
    def __init__(self, img_folder, gt_folder, phase: str = 'train', size: int = 224, test_size = 0.0, seed=None):
        if test_size == 0.0:
            self.images = sorted(glob.glob(img_folder + '/*'))
            self.gts = sorted(glob.glob(gt_folder + '/*'))
        else:
            self.images = sorted(glob.glob(img_folder + '/*'))
            self.gts = sorted(glob.glob(gt_folder + '/*'))
            train_images, val_images, train_gts, val_gts = train_test_split(self.images, self.gts,
                                                                            test_size=test_size,
                                                                            random_state=seed)
            if phase == 'train':
                self.images = train_images
                self.gts = train_gts
            elif phase == 'val':
                self.images = val_images
                self.gts = val_gts
            else:
                pass
        self.size = size
        print(len(self.images))
    def __len__(self):
        return len(self.images)
    @staticmethod
    def preprocess(pil_img, size, is_mask):
        # resize = transforms.Compose([
        #     transforms.Resize((size, size))
        # ])
        pil_img = pil_img.resize((size, size), Image.BILINEAR)
        # pil_img = pil_img.convert('L')
        # pil_img = resize(pil_img)
        img_ndarray = np.asarray(pil_img)
        if not is_mask: # 不是mask
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

        if is_mask:
            if img_ndarray.ndim != 3:
                img_ndarray = np.repeat(img_ndarray[np.newaxis, :, :], 3, axis=0)
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)


    def __getitem__(self, idx):
        img = self.load(self.images[idx])
        mask = self.load(self.gts[idx])
        
        image = self.preprocess(img, self.size, is_mask=False)
        mask = self.preprocess(mask, self.size, is_mask=True)
        
        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }



class Test_DatasetGenerate(Dataset):
    def __init__(self, img_folder, gt_folder=None, size: int = 224):
        self.images = sorted(glob.glob(img_folder + '/*'))
        self.gts = sorted(glob.glob(gt_folder + '/*')) if gt_folder is not None else None
        self.size = size

    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def preprocess(pil_img, size, is_mask):
        w, h = pil_img.size
        size = 224
        resize = transforms.Compose([
            transforms.Resize((size, size))
        ])
        pil_img = resize(pil_img)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)


    def __getitem__(self, idx):
        img = self.load(self.images[idx])
        mask = self.load(self.gts[idx])
        
        image = self.preprocess(img, self.size, is_mask=False)
        mask = self.preprocess(mask, self.size, is_mask=True)
        
        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


def get_loader(img_folder, gt_folder, phase: str, batch_size, shuffle, 
               num_workers, prefetch_factor, size=None, seed=None):
    if phase == 'test':
        dataset = Test_DatasetGenerate(img_folder, gt_folder, size)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=True)
    else:
        dataset = DatasetGenerate(img_folder, gt_folder, phase, size, seed)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, prefetch_factor=prefetch_factor,
                                 drop_last=True, pin_memory=True)

    print(f'{phase} length : {len(dataset)}')
    return data_loader
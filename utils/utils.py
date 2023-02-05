import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def coe_to_spatial(img):
    img = img.to(torch.cfloat)
    iimage = torch.fft.ifftshift(img)
    iimage = torch.fft.ifft2(iimage)
    iimage = torch.abs(iimage)
    
    iimage -= torch.min(iimage)
    iimage /= torch.max(iimage)
    return iimage

def binary(img, n_classes):
    # BCE unet, BCE complexunet v2, v3 is OK!
    img -= torch.min(img)
    img /= torch.max(img)
    img = F.softmax(img, dim=0)
    # img = torch.sigmoid(img)
    full_mask = img.cpu().squeeze()
    mask = F.one_hot(full_mask.argmax(dim=0), n_classes).permute(2, 0, 1).numpy()
    result = (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)
    result = (result > np.mean(result))
    return Image.fromarray(result)

def objectmap(img):
    img -= torch.min(img)
    img /= torch.max(img)
    transform = transforms.Grayscale()
    img = transform(img)
    img = img.cpu().squeeze().numpy()
    result = (img > np.mean(img))
    return Image.fromarray(result)

def norm(img):
    # img = img**2
    img -= img.min(1, keepdim=True)[0]
    img /= img.max(1, keepdim=True)[0]
    img = torch.tensor(img*255, dtype=torch.uint8)
    return img

import cv2
import numpy as np
from collections import Counter
import time
from numba import jit

@jit(nopython=True)
def calcIJ(img_patch):
    total_p = img_patch.shape[0] * img_patch.shape[1]
    if total_p % 2 != 0:
        center_p = img_patch[int(img_patch.shape[0] / 2), int(img_patch.shape[1] / 2)]
        mean_p = (np.sum(img_patch) - center_p) / (total_p - 1)
        return (center_p, mean_p)
    else:
        pass


def calcEntropy2dSpeedUp(img, win_w=3, win_h=3):
    height = img.shape[0]

    ext_x = int(win_w / 2)
    ext_y = int(win_h / 2)

    ext_h_part = np.zeros([height, ext_x], img.dtype)
    tem_img = np.hstack((ext_h_part, img, ext_h_part))
    ext_v_part = np.zeros([ext_y, tem_img.shape[1]], img.dtype)
    final_img = np.vstack((ext_v_part, tem_img, ext_v_part))

    new_width = final_img.shape[1]
    new_height = final_img.shape[0]

    # 最耗时的步骤，遍历计算二元组
    IJ = []
    for i in range(ext_x, new_width - ext_x):
        for j in range(ext_y, new_height - ext_y):
            patch = final_img[j - ext_y:j + ext_y + 1, i - ext_x:i + ext_x + 1]
            ij = calcIJ(patch)
            IJ.append(ij)

    Fij = Counter(IJ).items()

    # 第二耗时的步骤，计算各二元组出现的概率
    Pij = []
    for item in Fij:
        Pij.append(item[1] * 1.0 / (new_height * new_width))

    H_tem = []
    for item in Pij:
        h_tem = -item * (np.log(item) / np.log(2))
        H_tem.append(h_tem)

    H = np.sum(H_tem)
    return H
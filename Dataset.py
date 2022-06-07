from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from PIL import Image


VOC_COLORMAP = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]

VOC_CLASSES = ['1', '2', '3', '4', '5', '6']


def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射。"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                       colormap[2]] = i
    return colormap2label


def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引。"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    k = colormap2label[idx]
    return k


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1.0, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.size = 512

        self.ids = [splitext(file)[0] for file in listdir(masks_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_trans = voc_label_indices(pil_img, voc_colormap2label())
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        img = Image.open(img_file[0])
        mask = Image.open(mask_file[0])
        mask = mask.convert("RGB")
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1) / 255
        mask = torch.from_numpy(np.array(mask)).permute(2, 0, 1)
        mask = self.preprocess(mask)

        return {
            'image': img.type(torch.FloatTensor),
            'mask': mask.type(torch.FloatTensor)
        }

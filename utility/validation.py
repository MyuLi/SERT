import torch
import torchvision
import random
import cv2
import shutil
try: 
    from .util import *
except:
    from util import *

from torchvision.transforms import Compose, ToPILImage, ToTensor, RandomHorizontalFlip, RandomChoice
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import TransformDataset, SplitDataset, TensorDataset, ResampleDataset

from PIL import Image
from skimage.util import random_noise
from scipy.ndimage.filters import gaussian_filter


def show_validation_cadi():
    all_datadir = '/data/HSI_Data/icvl201'
    train_dir = '/data/HSI_Data/icvl_train_gaussian/'
    test_dir = '/data/HSI_Data/icvl_validation_5'
    all_fns = os.listdir(all_datadir)
    test_fns = os.listdir(test_dir)
    train_fns = os.listdir(train_dir)
    rest_fns = []
    for fn in all_fns:
        if fn not in test_fns:
            if fn not in train_fns:
                rest_fns.append(fn)
    print(rest_fns)


if __name__ == '__main__':
    show_validation_cadi()
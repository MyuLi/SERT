import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import random
import scipy.stats as stats
from torch.utils.data import DataLoader

from skimage import io
import cv2

####################i##############################################################################

class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

def load_tif_img(filepath):
    img = io.imread(filepath)
    img = img.astype(np.float32)
    #if type == 'gt':
    img = img/4096.

    return img

def is_tif_file(filename):
    return any(filename.endswith(extension) for extension in [".tif"])   

class DataLoaderTrain(Dataset):
    def __init__(self, data_dir, ratio=50, img_options=None, target_transform=None,use2d=True,repeat=20):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))
        
        self.clean_filenames = [os.path.join(data_dir, 'gt', x)          for x in clean_files if is_tif_file(x)]
        self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x)     for x in noisy_files if is_tif_file(x)]

        self.clean = [torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        self.ratio = ratio
        self.use2d=use2d
        self.repeat =repeat

    def __len__(self):
        return self.tar_size*self.repeat

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = self.clean[tar_index]
        noisy = self.noisy[tar_index]
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)
        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * self.ratio

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]

        return  noisy,clean#, clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, data_dir, ratio=50, target_transform=None,use2d=True):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))

        self.clean_filenames = [os.path.join(data_dir, 'gt', x)      for x in clean_files if is_tif_file(x)]
        self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x) for x in noisy_files if is_tif_file(x)]

        self.clean = [torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        

        self.tar_size = len(self.clean_filenames)  
        self.ratio = ratio
        self.use2d = use2d

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        clean = self.clean[tar_index]
        noisy = self.noisy[tar_index]
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
       
        ps = 512
        r = clean.shape[1]//2-ps//2
        c = clean.shape[2]//2-ps//2
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * self.ratio
        
        if not self.use2d:
            clean = clean[None,...]
            noisy = noisy[None,...]
        clean = torch.clamp(clean, 0, 1)
        noisy = torch.clamp(noisy, 0, 1)
        return  noisy,clean#, clean_filename, noisy_filename


if __name__ == '__main__':
    rgb_dir = '/media/lmy/LMY/aaai/real_dataset'
    ratio = 50
    train_dir = '/media/lmy/LMY/aaai/train_real/'
    img_options ={}
    img_options['patch_size'] = 128
    #train_dataset = DataLoaderTrain(train_dir,50,img_options=img_options)
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=1, shuffle=True,
    #                           num_workers=1)
    test_dir= '/media/lmy/LMY/aaai/test_real/'
    dataset = DataLoaderVal(test_dir, ratio, None)
    
    # print(len(dataset))
   
    train_loader = DataLoader(dataset, batch_size=1, num_workers=1)
    #print(iter(train_loader).next())
    for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(batch_idx,inputs.shape)
            band =20
            inputs = inputs.numpy()
            targets = targets.numpy()
            cv2.imwrite('tnoisy_'+'_band'+str(band)+'.png',inputs[0,band]*255)
            cv2.imwrite('tgt_'+'_band'+str(band)+'.png',targets[0,band]*255)
            break
    

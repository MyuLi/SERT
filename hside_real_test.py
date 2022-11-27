import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import *
from hsi_setup import Engine, train_options, make_dataset


if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)
    

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.net.use_2dconv)


    """Test-Dev"""
    basefolder = '/test_real'
  
    mat_datasets = DataLoaderVal(basefolder, 50, None,use2d=engine.get_net().use_2dconv)
    print(len(mat_datasets))
    print('loading finished')

 
    mat_loader = DataLoader(
        mat_datasets,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda     )  
      
    strart_time = time.time()
    engine.test(mat_loader, basefolder)
    end_time = time.time()
    test_time = end_time-strart_time
    print('cost-time: ',(test_time/15))
        

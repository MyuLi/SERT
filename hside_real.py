import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from utility import *
import datetime
import time
from hsi_setup import Engine, train_options, make_dataset
#os.environ["WANDB_MODE"] ='offline'

if __name__ == '__main__':
    """Training settings"""
    

       
    

    parser = argparse.ArgumentParser(
    description='Hyperspectral Image Denoising (Complex noise)')
    opt = train_options(parser)
    print(opt)
    
    img_options={}
    img_options['patch_size'] = 128

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    
    train_dir = '/train_real/'

    train_dataset = DataLoaderTrain(train_dir,50,img_options=img_options,use2d=engine.get_net().use_2dconv)
    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
   
    print('==> Preparing data..')


    """Test-Dev"""
    
    basefolder = '/test_real'
    
    mat_datasets = DataLoaderVal(basefolder, 50, None,use2d=engine.get_net().use_2dconv)
    
    mat_loader = DataLoader(
        mat_datasets,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda
    )      

    base_lr = opt.lr
    epoch_per_save = 20
    adjust_learning_rate(engine.optimizer, opt.lr)
    print('loading finished')
    # from epoch 50 to 100
    engine.epoch  = 0
    while engine.epoch < 1000:
        np.random.seed()

        if engine.epoch == 200:
        
            adjust_learning_rate(engine.optimizer, base_lr*0.5)
          
        if engine.epoch == 400:
            
            adjust_learning_rate(engine.optimizer, base_lr*0.1)

        
        engine.train(train_loader,mat_loader)
        
        engine.validate(mat_loader, 'real')
        

        display_learning_rate(engine.optimizer)
        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()
    wandb.finish()

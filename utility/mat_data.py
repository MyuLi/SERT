"""generate testing mat dataset"""
import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat

from util import crop_center, Visualize3D, minmax_normalize
from PIL import Image

def create_mat_dataset(datadir, fnames, newdir, matkey, func=None, load=h5py.File):
    if not exists(newdir):
        os.mkdir(newdir)
    

    for i, fn in enumerate(fnames):
        print('generate data(%d/%d)' %(i+1, len(fnames)))
        filepath = join(datadir, fn)
        try:
            mat = load(filepath)
            
            data = func(mat[matkey][...])
            data_hwc = data.transpose((2,1,0))
            savemat(join(newdir, fn), {'data': data_hwc})
            try:
                Image.fromarray(np.array(data_hwc*255,np.uint8)[:,:,20]).save('/data/HSI_Data/icvl_test_512_png/{}.png'.format(os.path.splitext(fn)[0]))
            except Exception as e:
                print(e)
        except:
            print('open error for {}'.format(fn))
            continue
            
        

def create_icvl_sr():
    basedir = '/data/HSI_Data/'
    datadir = join(basedir, '/data/HSI_Data/icvl201/')
    newdir = join(basedir, '/media/lmy/LMY/cvpr2023/test_96/')

    #fnames = os.listdir(datadir)
    f = open("test_list.txt")
    fnames = f.readlines()
    for i in range(len(fnames)):
        fnames[i] = fnames[i].split('\n')[0]
    f.close()
    print(fnames)
    
    def func(data):
        data = np.rot90(data, k=-1, axes=(1,2))
        
        data = crop_center(data, 512, 512)
        
        data = minmax_normalize(data)
        return data
    
    create_mat_dataset(datadir, fnames, newdir, 'rad', func=func)

def generate_icvl_png():
    basedir = '/data/HSI_Data/laisdata'
    #/data/HSI_Data/laisdata/sig10_size256
    datadir = join(basedir, 'sig10_size256')
    newdir = join(basedir, 'size256_png')
    fnames = os.listdir(datadir)
    # def func(data):
    #     data = np.rot90(data, k=-1, axes=(1,2))
    #     data = minmax_normalize(data)
    #     return data

    if not exists(newdir):
        os.mkdir(newdir)
    
    for i, fn in enumerate(fnames):
        print('generate data(%d/%d)' %(i+1, len(fnames)))
        filepath = join(datadir, fn)
        try:
            #mat = h5py.File(filepath)
            mat = loadmat(filepath)
            #data = func(mat['rad'][...])
            data = mat['gt']
            # data = mat['data']
            #data_hwc = data.transpose((2,1,0))
            data_hwc = data
            Image.fromarray(np.array(data_hwc*255,np.uint8)[:,:,20]).save(os.path.join(newdir, '{}.png'.format(os.path.splitext(fn)[0])))
        
        except:
            print('open error for {}'.format(fn))
            continue


def copydata():
    basedir = '/data/HSI_Data/laisdata'
    datadir = join(basedir, 'sig10')
    newdir = join(basedir, 'gt')
    fnames = os.listdir(datadir)
    for i, fn in enumerate(fnames):
        print('generate data(%d/%d)' %(i+1, len(fnames)))
        filepath = join(datadir, fn)
        mat = loadmat(filepath)
        data = mat['gt']
        savemat(join(newdir, fn), {'data': data})



if __name__ == '__main__':
    create_icvl_sr()
    #generate_icvl_png()
    #copydata()
    pass

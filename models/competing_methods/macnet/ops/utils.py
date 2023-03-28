import torch
import torch.functional as F
from random import randint
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import compare_ssim, compare_psnr
from .gauss import fspecial_gauss
from scipy import signal
def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))
def gen_bayer_mask(h,w):
    x = torch.zeros(1, 3, h, w)

    x[:, 0, 1::2, 1::2] = 1  # r
    x[:, 1, ::2, 1::2] = 1
    x[:, 1, 1::2, ::2] = 1  # g
    x[:, 2, ::2, ::2] = 1  # b

    return x

def togray(tensor):
    b, c, h, w = tensor.shape
    tensor = tensor.view(b, 3, -1, h, w)
    tensor = tensor.sum(1)
    return tensor

def torch_to_np(img_var):
    return img_var.detach().cpu().numpy()

def plot_tensor(img, **kwargs):
    inp_shape = tuple(img.shape)
    print(inp_shape)
    img_np = torch_to_np(img)
    if inp_shape[1]==3:
        img_np_ = img_np.transpose([1,2,0])
        plt.imshow(img_np_)

    elif inp_shape[1]==1:
        img_np_ = np.squeeze(img_np)
        plt.imshow(img_np_, **kwargs)

    else:
        # raise NotImplementedError
        plt.imshow(img_np, **kwargs)
    plt.axis('off')


def get_mask(A):
    mask = A.clone().detach()
    mask[A != 0] = 1
    return mask.byte()

def sparsity(A):
    return get_mask(A).sum().item()/A.numel()

def soft_threshold(x, lambd):
    return nn.functional.relu(x - lambd,inplace=True) - nn.functional.relu(-x - lambd,inplace=True)
def nn_threshold(x, lambd):
    return nn.functional.relu(x - lambd)

def fastSoftThrs(x, lmbda):
    return x + 0.5 * (torch.abs(x-torch.abs(lmbda))-torch.abs(x+torch.abs(lmbda)))

def save_checkpoint(state,ckpt_path):
    torch.save(state, ckpt_path)

def generate_key():
    return '{}'.format(randint(0, 100000))

def show_mem():
    mem = torch.cuda.memory_allocated() * 1e-6
    max_mem = torch.cuda.max_memory_allocated() * 1e-6
    return mem, max_mem

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def step_lr(optimizer, lr_decay):
    lr = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = lr * lr_decay
def set_lr(optimizer, lr):
    # lr = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = lr

def step_lr_als(optimizer, lr_decay):
    lr = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = lr * lr_decay
    optimizer.param_groups[1]['lr'] *= lr_decay

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def gen_mask_windows(h, w):
    '''
    return mask for block window
    :param h:
    :param w:
    :return: (h,w,h,w)
    '''
    mask = torch.zeros(2 * h, 2 * w, h, w)
    for i in range(h):
        for j in range(w):
            mask[i:i + h, j:j + w, i, j] = 1

    return mask[h // 2:-h // 2, w // 2:-w // 2, :, :]


def gen_linear_mask_windows(h, w, h_,w_):
    '''
    return mask for block window
    :param h:
    :param w:
    :return: (h,w,h,w)
    '''

    x = torch.ones(1, 1, h - h_ + 1, w - w_ + 1)
    k = torch.ones(1, 1, h_, w_)
    kernel = F.conv_transpose2d(x, k)
    kernel /= kernel.max()
    mask = torch.zeros(2 * h, 2 * w, h, w)
    for i in range(h):
        for j in range(w):
            mask[i:i + h, j:j + w, i, j] = kernel

    return mask[h // 2:-h // 2, w // 2:-w // 2, :, :]

def gen_quadra_mask_windows(h, w, h_,w_):
    '''
    return mask for block window
    :param h:
    :param w:
    :return: (h,w,h,w)
    '''

    x = torch.ones(1, 1, h - h_ + 1, w - w_ + 1)
    k = torch.ones(1, 1, h_, w_)
    kernel = F.conv_transpose2d(x, k) **2
    kernel /= kernel.max()
    mask = torch.zeros(2 * h, 2 * w, h, w)
    for i in range(h):
        for j in range(w):
            mask[i:i + h, j:j + w, i, j] = kernel

    return mask[h // 2:-h // 2, w // 2:-w // 2, :, :]

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)
def Init_DCT(n, m):
    """ Compute the Overcomplete Discrete Cosinus Transform. """
    n=int(n)
    m=int(m)
    Dictionary = np.zeros((n,m))
    for k in range(m):
        V = np.cos(np.arange(0, n) * k * np.pi / m)
        if k > 0:
            V = V - np.mean(V)
        Dictionary[:, k] = V / np.linalg.norm(V)
    # Dictionary = np.kron(Dictionary, Dictionary)
    # Dictionary = Dictionary.dot(np.diag(1 / np.sqrt(np.sum(Dictionary ** 2, axis=0))))
    # idx = np.arange(0, n ** 2)
    # idx = idx.reshape(n, n, order="F")
    # idx = idx.reshape(n ** 2, order="C")
    # Dictionary = Dictionary[idx, :]
    Dictionary = torch.from_numpy(Dictionary).float()
    return Dictionary

def est_noise(y, noise_type='additive'):
    """
    This function infers the noise in a
    hyperspectral data set, by assuming that the
    reflectance at a given band is well modelled
    by a linear regression on the remaining bands.

    Parameters:
        y: `numpy array`
            a HSI cube ((m*n) x p)

       noise_type: `string [optional 'additive'|'poisson']`

    Returns: `tuple numpy array, numpy array`
        * the noise estimates for every pixel (N x p)
        * the noise correlation matrix estimates (p x p)

    Copyright:
        Jose Nascimento (zen@isel.pt) and Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    # def est_additive_noise(r):
    #     small = 1e-6
    #     L, N = r.shape
    #     w=np.zeros((L,N), dtype=np.float)
    #     RR=np.dot(r,r.T)
    #     RRi = np.linalg.pinv(RR+small*np.eye(L))
    #     RRi = np.matrix(RRi)
    #     for i in range(L):
    #         XX = RRi - (RRi[:,i]*RRi[i,:]) / RRi[i,i]
    #         RRa = RR[:,i]
    #         RRa[i] = 0
    #         beta = np.dot(XX, RRa)
    #         beta[0,i]=0;
    #         w[i,:] = r[i,:] - np.dot(beta,r)
    #     Rw = np.diag(np.diag(np.dot(w,w.T) / N))
    #     return w, Rw
    def est_additive_noise(r):
        small = 1e-6
        L, N = r.shape
        w=torch.zeros((L,N), dtype=torch.float,device=r.device)
        RR=r@r.T
        # print((small*torch.eye(L,device=r.device)).device)
        temp=RR+small*torch.eye(L,device=r.device)
        # print(temp.device)
        RRi = torch.inverse(temp)

        # RRi = np.matrix(RRi)
        for i in range(L):
            XX = RRi - (RRi[:,i].unsqueeze(1)*RRi[i,:].unsqueeze(0)) / RRi[i,i]
            RRa = RR[:,i]
            RRa[i] = 0
            beta =XX@RRa
            beta[i]=0;
            w[i,:] = r[i,:] - beta@r
        Rw = torch.diag(torch.diag((w@w.T) / N))
        return w, Rw

    h, w, numBands = y.shape
    y = torch.reshape(y, (w * h, numBands))
    # y = np.reshape(y, (w * h, numBands))
    y = y.T
    L, N = y.shape
    # verb = 'poisson'
    if noise_type == 'poisson':
        sqy = torch.sqrt(y * (y > 0))
        u, Ru = est_additive_noise(sqy)
        x = (sqy - u) ** 2
        w = torch.sqrt(x) * u * 2
        Rw = (w@w.T) / N
    # additive
    else:
        w, Rw = est_additive_noise(y)
    return w.T, Rw.T

    # y = y.T
    # L, N = y.shape
    # #verb = 'poisson'
    # if noise_type == 'poisson':
    #     sqy = np.sqrt(y * (y > 0))
    #     u, Ru = est_additive_noise(sqy)
    #     x = (sqy - u)**2
    #     w = np.sqrt(x)*u*2
    #     Rw = np.dot(w,w.T) / N
    # # additive
    # else:
    #     w, Rw = est_additive_noise(y)
    # return w.T, Rw.T


def hysime(y, n, Rn):
    """
    Hyperspectral signal subspace estimation

    Parameters:
        y: `numpy array`
            hyperspectral data set (each row is a pixel)
            with ((m*n) x p), where p is the number of bands
            and (m*n) the number of pixels.

        n: `numpy array`
            ((m*n) x p) matrix with the noise in each pixel.

        Rn: `numpy array`
            noise correlation matrix (p x p)

    Returns: `tuple integer, numpy array`
        * kf signal subspace dimension
        * Ek matrix which columns are the eigenvectors that span
          the signal subspace.

    Copyright:
        Jose Nascimento (zen@isel.pt) & Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    h, w, numBands = y.shape
    y = torch.reshape(y, (w * h, numBands))
    y=y.T
    n=n.T
    Rn=Rn.T
    L, N = y.shape
    Ln, Nn = n.shape
    d1, d2 = Rn.shape

    x = y - n;

    Ry = y@y.T / N
    Rx = x@x.T/ N
    E, dx, V =torch.svd(Rx.cpu())
    E=E.to(device=y.device)
    # print(V)
    Rn = Rn+torch.sum(torch.diag(Rx))/L/10**5 * torch.eye(L,device=y.device)
    Py = torch.diag(E.T@(Ry@E))
    Pn = torch.diag(E.T@(Rn@E))
    cost_F = -Py + 2 * Pn
    kf = torch.sum(cost_F < 0)
    ind_asc = torch.argsort(cost_F)
    Ek = E[:, ind_asc[0:kf]]
    # h, w, numBands = y.shape
    # y = np.reshape(y, (w * h, numBands))
    # y = y.T
    # n = n.T
    # Rn = Rn.T
    # L, N = y.shape
    # Ln, Nn = n.shape
    # d1, d2 = Rn.shape
    #
    # x = y - n;
    #
    # Ry = np.dot(y, y.T) / N
    # Rx = np.dot(x, x.T) / N
    # E, dx, V = np.linalg.svd(Rx)
    #
    # Rn = Rn + np.sum(np.diag(Rx)) / L / 10 ** 5 * np.eye(L)
    # Py = np.diag(np.dot(E.T, np.dot(Ry, E)))
    # Pn = np.diag(np.dot(E.T, np.dot(Rn, E)))
    # cost_F = -Py + 2 * Pn
    # kf = np.sum(cost_F < 0)
    # ind_asc = np.argsort(cost_F)
    # Ek = E[:, ind_asc[0:kf]]
    return kf, E # Ek.T ?
def count(M):
    w, Rw = est_noise(M)
    kf, Ek = hysime(M, w, Rw)
    return kf, Ek, w, Rw

def cal_sam(X, Y, eps=1e-8):
    # X = torch.squeeze(X.data).cpu().numpy()
    # Y = torch.squeeze(Y.data).cpu().numpy()
    tmp = (np.sum(X*Y, axis=0) + eps) / ((np.sqrt(np.sum(X**2, axis=0)) + eps) * (np.sqrt(np.sum(Y**2, axis=0)) + eps)+eps)
    return np.mean(np.real(np.arccos(tmp)))
def cal_psnr(im_true,im_test,eps=13-8):
    c,_,_=im_true.shape
    bwindex = []
    for i in range(c):
        bwindex.append(compare_psnr(im_true[i,:,:], im_test[i,:,:]))
    return  np.mean(bwindex)
def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))

def cal_ssim(im_true,im_test,eps=13-8):
    # print(im_true.shape)
    # print(im_true.shape)
    # print(im_test.shape)
    # im_true=im_true.cpu().numpy()
    # im_test = im_test.cpu().numpy()
    c,_,_=im_true.shape
    bwindex = []
    for i in range(c):
        bwindex.append(ssim(im_true[i,:,:]*255, im_test[i,:,:,]*255))
    return np.mean(bwindex)
# def cal_ssim(im_true,im_test,eps=13-8):
#     c,_,_=im_true.shape
#     bwindex = []
#     for i in range(c):
#         bwindex.append(compare_ssim(im_true[i,:,:], im_test[i,:,:,]))
#     return np.mean(bwindex)

# class Bandwise(object):
#     def __init__(self, index_fn):
#         self.index_fn = index_fn
#
#     def __call__(self, X, Y):
#         C = X.shape[-3]
#         bwindex = []
#         for ch in range(C):
#             x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
#             y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
#             index = self.index_fn(x, y)
#             bwindex.append(index)
#         return bwindex


def MSIQA(X, Y):
    # print(X.shape)
    # print(Y.shape)
    psnr = cal_psnr(X, Y)
    ssim = cal_ssim(X, Y)
    sam = cal_sam(X, Y)
    return psnr, ssim, sam
if __name__ == '__main__':
    hsi = torch.rand(200,200, 198)
    w, Rw=est_noise(hsi)
    kf, E= hysime(hsi, w, Rw)
    print(kf)





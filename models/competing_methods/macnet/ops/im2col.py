from torch.nn import functional as F
import torch
from torch.nn.modules.utils import _pair
import math


def Im2Col(input_tensor, kernel_size, stride, padding,dilation=1,tensorized=False,):
    batch = input_tensor.shape[0]
    out = F.unfold(input_tensor, kernel_size=kernel_size, padding=padding, stride=stride,dilation=dilation)

    if tensorized:
        lh,lw = im2col_shape(input_tensor.shape[1:],kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)[-2:]
        out = out.view(batch,-1,lh,lw)
    return out
def Cube2Col(input_tensor, kernel_size, stride, padding,dilation=1,tensorized=False,):
    input_sz=input_tensor.shape
    if input_sz[1]<kernel_size:
        input_tensor = F.pad(input_tensor, (0, 0, 0, 0, kernel_size-input_sz[1], 0), 'constant', 0)
        # input_tensor=F.pad(input_tensor,(0,1), mode='replicate')
    input_sz=input_tensor.shape
    _t=input_sz[1]-kernel_size+1
    out=torch.zeros(input_sz[0],kernel_size**3,input_sz[1]-kernel_size+1,input_sz[2]-kernel_size+1,input_sz[3]-kernel_size+1)

    for i in range(_t):
        ind1=i
        ind2=i+kernel_size
        out[:,:,i,:,:]=Im2Col(input_tensor[:,ind1:ind2,:,:], kernel_size, stride, padding, dilation, tensorized)
    return out
def Col2Cube(input_tensor,output_size, kernel_size, stride, padding, dilation=1, avg=False,input_tensorized=False):
    batch = input_tensor.shape[0]
    _t = output_size[0] - kernel_size + 1
    out = torch.zeros([batch,output_size[0],output_size[1],output_size[2]]).to(input_tensor.device)
    me=torch.zeros_like(out).to(input_tensor.device)
    for i in range(_t):
        ind1 = i
        ind2 = i + kernel_size
        if input_tensorized:
            temp_tensor = input_tensor[:,:,i,:,:].flatten(2,3)
            out[:,ind1:ind2,:,:] += F.fold(temp_tensor, output_size=output_size[1:], kernel_size=kernel_size, padding=padding, stride=stride,dilation=dilation)
            me[:,ind1:ind2,:,:] += F.fold(torch.ones_like(temp_tensor), output_size=output_size[1:], kernel_size=kernel_size,
                        padding=padding, stride=stride, dilation=dilation)


    if avg:
        # me[me==0]=1 # !!!!!!!
        out = out / me

        # me_ = F.conv_transpose2d(torch.ones_like(input_tensor),torch.ones(1,1,kernel_size,kernel_size))

    return out


def Col2Im(input_tensor,output_size, kernel_size, stride, padding, dilation=1, avg=False,input_tensorized=False):
    batch = input_tensor.shape[0]

    if input_tensorized:
        input_tensor = input_tensor.flatten(2,3)
    out = F.fold(input_tensor, output_size=output_size, kernel_size=kernel_size, padding=padding, stride=stride,dilation=dilation)

    if avg:
        me = F.fold(torch.ones_like(input_tensor), output_size=output_size, kernel_size=kernel_size, padding=padding, stride=stride,dilation=dilation)
        # me[me==0]=1 # !!!!!!!
        out = out / me

        # me_ = F.conv_transpose2d(torch.ones_like(input_tensor),torch.ones(1,1,kernel_size,kernel_size))

    return out


class Col2Im_(torch.nn.Module):

    def __init__(self,input_shape, output_size, kernel_size, stride, padding, dilation=1, avg=False,input_tensorized=False):
        super(Col2Im_,self).__init__()

        xshape = tuple(input_shape)

        if input_tensorized:
            xshape = xshape[0:2]+(xshape[2]*xshape[3],)

        if avg:
            me = F.fold(torch.ones(xshape), output_size=output_size, kernel_size=kernel_size,
                        padding=padding, stride=stride, dilation=dilation)
            me[me == 0] = 1
            self.me = me

    def forward(self, input_tensor,output_size, kernel_size, stride, padding, dilation=1, avg=False,input_tensorized=False):
        if input_tensorized:
            input_tensor = input_tensor.flatten(2, 3)
        out = F.fold(input_tensor, output_size=output_size, kernel_size=kernel_size, padding=padding, stride=stride,
                     dilation=dilation)
        if avg:
            out /= self.me
        return out

# def im2col_shape(size, kernel_size, stride, padding):
#     ksize_h, ksize_w = _pair(kernel_size)
#     stride_h, stride_w = _pair(stride)
#     pad_h, pad_w = _pair(padding)
#     n_input_plane, height, width = size
#     height_col = (height + 2 * pad_h - ksize_h) // stride_h + 1
#     width_col = (width + 2 * pad_w - ksize_w) // stride_w + 1
#     return n_input_plane, ksize_h, ksize_w, height_col, width_col

def im2col_shape(size, kernel_size, stride, padding, dilation):
    ksize_h, ksize_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    dil_h, dil_w = _pair(dilation)
    pad_h, pad_w = _pair(padding)
    n_input_plane, height, width = size
    height_col = (height + 2 * pad_h - dil_h * (ksize_h-1)-1) / stride_h + 1
    width_col = (width + 2 * pad_w - dil_w * (ksize_w-1)-1) / stride_w + 1
    return n_input_plane, ksize_h, ksize_w, math.floor(height_col), math.floor(width_col)


def col2im_shape(size, kernel_size, stride, padding, input_size=None):
    ksize_h, ksize_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    n_input_plane, ksize_h, ksize_w, height_col, width_col = size
    if input_size is not None:
        height, width = input_size
    else:
        height = (height_col - 1) * stride_h - 2 * pad_h + ksize_h
        width = (width_col - 1) * stride_w - 2 * pad_w + ksize_w
    return n_input_plane, height, width
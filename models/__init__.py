from .sert import SERT
from .competing_methods import *

def sert_base():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32,32] ,        depths=[ 6,6,6],         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=128,down_rank=8)     #16,32,32

    net.use_2dconv = True     
    net.bandwise = False          
    return net


def sert_tiny():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32] ,        depths=[ 4,4],         num_heads=[ 6,6],split_sizes=[2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=128,down_rank=8)     #16,32,32
    net.use_2dconv = True     
    net.bandwise = False          
    return net

def sert_small():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32,32] ,        depths=[ 4,4,4],         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=128,down_rank=8)     #16,32,32
    net.use_2dconv = True     
    net.bandwise = False          
    return net

def sert_urban():
    net = SERT(inp_channels=210,dim = 96*2,         window_sizes=[8,16,16] ,        depths=[ 6,6,6],         num_heads=[ 6,6,6],split_sizes=[2,4,4],mlp_ratio=2,down_rank=8,memory_blocks=128)  
    net.use_2dconv = True     
    net.bandwise = False          
    return net


def sert_real():
    net = SERT(inp_channels=34,dim = 96,         window_sizes=[16,32,32] ,        depths=[6,6,6],down_rank=8,         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,memory_blocks=64)

    net.use_2dconv = True     
    net.bandwise = False          
    return net

def qrnn3d():
    net = QRNNREDC3D(1, 16, 5, [1, 3], has_ad=True)
    net.use_2dconv = False
    net.bandwise = False
    return net

def grn_net():
    net = U_Net_GR(in_ch=31,out_ch=31)
    net.use_2dconv = True
    net.bandwise = False
    return net


def grn_net_real():
    net = U_Net_GR(in_ch=34,out_ch=34)
    net.use_2dconv = True
    net.bandwise = False
    return net

def grn_net_urban():
    net = U_Net_GR(in_ch=210,out_ch=210)
    net.use_2dconv = True
    net.bandwise = False
    return net


def t3sc():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('models/competing_methods/T3SC/layers/t3sc.yaml')
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net

def t3sc_real():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('models/competing_methods/T3SC/layers/t3sc_real.yaml')
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net

def t3sc_urban():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('models/competing_methods/T3SC/layers/t3sc_urban.yaml')
    net = MultilayerModel(**cfg.params)
    net.use_2dconv = True
    net.bandwise = False
    return net

def macnet():
    net = MACNet(in_channels=1,channels=16,num_half_layer=5)
    net.use_2dconv = True
    net.bandwise = False          
    return net 
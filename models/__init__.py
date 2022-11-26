from .sert import SERT

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

def sert_large():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32,32,32] ,        depths=[ 6,6,6,6],         num_heads=[ 6,6,6,6],split_sizes=[1,2,4,8],mlp_ratio=2,weight_factor=0.1,memory_blocks=128,down_rank=8)     #16,32,32
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

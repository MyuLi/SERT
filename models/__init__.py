from .sert import SERT

def sert():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32,32] ,        depths=[ 6,6,6],         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=64,down_rank=4)     #16,32,32

    net.use_2dconv = True     
    net.bandwise = False          
    return net


def sert_tiny():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32] ,        depths=[ 4,4],         num_heads=[ 6,6],split_sizes=[2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=64,down_rank=4)     #16,32,32
    net.use_2dconv = True     
    net.bandwise = False          
    return net

def sert_small():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32,32] ,        depths=[ 4,4,4],         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=64,down_rank=4)     #16,32,32
    net.use_2dconv = True     
    net.bandwise = False          
    return net

def sert_large():
    net = SERT(inp_channels=31,dim = 96,         window_sizes=[16,32,32,32] ,        depths=[ 6,6,6,6],         num_heads=[ 6,6,6,6],split_sizes=[1,2,4,8],mlp_ratio=2,weight_factor=0.1,memory_blocks=64,down_rank=4)     #16,32,32
    net.use_2dconv = True     
    net.bandwise = False          
    return net

def sert_wdc():
    net = SERT(inp_channels=191,dim = 192,         window_sizes=[16,32,32] ,        depths=[ 6,6,6],         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,weight_factor=0.1,memory_blocks=64,down_rank=4)     #16,32,32

    net.use_2dconv = True     
    net.bandwise = False          
    return net


def sert_real():
    net = SERT(inp_channels=34,dim = 96,         window_sizes=[16,32,32] ,        depths=[6,6,6],down_rank=30,         num_heads=[ 6,6,6],split_sizes=[1,2,4],mlp_ratio=2,me_block_num=64,weight_factor=0.1,memory_blocks=64,down_rank=4)

    net.use_2dconv = True     
    net.bandwise = False          
    return net

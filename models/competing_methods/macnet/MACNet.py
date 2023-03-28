from collections import namedtuple
from .ops.utils import est_noise,count
# from model.qrnn.combinations import *
from .non_local import NLBlockND,EfficientNL
from .combinations import *
Params = namedtuple('Params', ['in_channels', 'channels', 'num_half_layer','rs'])
from skimage.restoration import  denoise_nl_means,estimate_sigma
class MACNet(nn.Module):
    '''
    Tied lista with coupling
    '''

    def __init__(self, in_channels=1,channels =16, num_half_layer =5 ):
        super(MACNet, self).__init__()
        self.rs = 2
        self.net=REDC3DBNRES_NL(in_channels=in_channels,channels=channels,num_half_layer=num_half_layer)

    def forward(self, I, writer=None, epoch=None, return_patches=False):

        return self.pro_sub(I)

    def pro_sub(self, I):
        R = list()
        Ek = list()
        Rw = list()
        I_iid = list()
        sigma_est = 0
        I_size = I.shape
        for _I in I:
            _I = _I.permute([1, 2, 0])
            _, _, w, _Rw = count(_I)  # count subspace

            _I = torch.matmul(_I, torch.inverse(_Rw).sqrt())  # spectral iid
            I_nlm = _I.cpu().numpy()
            sigma_est = estimate_sigma(I_nlm, multichannel=True, average_sigmas=True)
            I_nlm = denoise_nl_means(I_nlm, patch_size=7, patch_distance=9, h=0.08, multichannel=True,
                                     fast_mode=True, sigma=sigma_est)
            I_nlm = torch.FloatTensor(I_nlm).to(device=_I.device)
            _R, _Ek, _, _ = count(I_nlm)
            if self.rs:
                _R = _R // 3
            # _R = max(_R, torch.FloatTensor(3).to(I.device))
            R.append(_R)
            Ek.append(_Ek)
            Rw.append(_Rw)
            I_iid.append(_I)
        dim = max(torch.stack(R).max(), 3)
        Ek = torch.stack(Ek, dim=0)
        I_iid = torch.stack(I_iid, dim=0)
        Ek = Ek[:, :, 0:dim]
        Rw = torch.stack(Rw, dim=0)
        I_sub = torch.bmm(I_iid.view(I_size[0], -1, I_size[1]), Ek)
        I_sub = I_sub.view(I_size[0], I_size[2], I_size[3], -1).permute([0, 3, 1, 2])
        CNN_sub = self.net(I_sub.unsqueeze(1)).squeeze(1)
        CNN_sub = CNN_sub.view(I_size[0], dim, -1)
        output = torch.bmm(Rw.sqrt(), torch.bmm(Ek, CNN_sub))
        output = output.view(I_size)
        return output
class REDC3DBNRES_NL(torch.nn.Module):
    """Residual Encoder-Decoder Convolution 3D
    Args:
        downsample: downsample times, None denotes no downsample"""

    def __init__(self, in_channels, channels, num_half_layer, downsample=None):
        super(REDC3DBNRES_NL, self).__init__()
        # Encoder
        # assert downsample is None or 0 < downsample <= num_half_layer
        interval = 2

        self.feature_extractor = BNReLUConv3d(in_channels, channels)
        self.encoder = nn.ModuleList()
        for i in range(1, num_half_layer + 1):
            if i % interval:
                encoder_layer = BNReLUConv3d(channels, channels)
            else:
                encoder_layer = BNReLUConv3d(channels, 2 * channels, k=3, s=(1, 2, 2), p=1)
                channels *= 2
            self.encoder.append(encoder_layer)
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(1, num_half_layer + 1):
            if i % interval:
                decoder_layer = BNReLUDeConv3d(channels, channels)
            else:
                decoder_layer = BNReLUUpsampleConv3d(channels, channels // 2)
                channels //= 2
            self.decoder.append(decoder_layer)
        self.reconstructor = BNReLUDeConv3d(channels, in_channels)
        # self.enl_1 = EfficientNL(in_channels=channels)
        self.enl_2 = EfficientNL(in_channels=channels)
        self.enl_3 = EfficientNL(in_channels=1,key_channels=1,value_channels=1,head_count=1)

     # = None, head_count = None,  = None
    def forward(self, x):
        num_half_layer = len(self.encoder)
        xs = [x]
        out = self.feature_extractor(xs[0])
        xs.append(out)
        for i in range(num_half_layer - 1):
            out = self.encoder[i](out)
            xs.append(out)
        out = self.encoder[-1](out)
        # out = self.nl_1(out)
        out = self.decoder[0](out)
        for i in range(1, num_half_layer):
            out = out + xs.pop()
            out = self.decoder[i](out)
        out = self.enl_2(out) + xs.pop()
        out = self.reconstructor(out)
        out = self.enl_3(out) + xs.pop()
        return out


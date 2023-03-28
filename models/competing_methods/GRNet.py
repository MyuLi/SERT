from re import S
from turtle import forward
from matplotlib.pyplot import sca
from numpy import True_, pad
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_relu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True):
        super(conv_relu, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class GSM(nn.Module):
    def __init__(self, in_ch):
        super(GSM, self).__init__()
        self.channel = in_ch
        self.conv1 = nn.Conv2d(self.channel, self.channel//2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.channel, self.channel//2, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.channel, self.channel//2, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(self.channel//2, self.channel, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        theta = self.conv1(x)
        theta = torch.reshape(theta, (-1, theta.shape[1], theta.shape[2]*theta.shape[3]))

        phi = self.conv2(x)
        phi = torch.reshape(phi, (-1, phi.shape[1], phi.shape[2]*phi.shape[3]))

        g = self.conv3(x)
        g = torch.reshape(g, (-1, g.shape[1], g.shape[2]*g.shape[3]))

        phi1 = torch.reshape(phi, (-1, phi.shape[1]*phi.shape[2]))
        phi1 = F.softmax(phi1, dim=-1)
        phi1 = torch.reshape(phi1, phi.shape)

        g1 = torch.reshape(g, (-1, g.shape[1]*g.shape[2]))
        g1 = F.softmax(g1, dim=-1)
        g1 = torch.reshape(g1, g.shape)

        phi1 = phi1.transpose(1,2)
        y = torch.bmm(theta, phi1)
       # print(theta.shape[1]*phi1.shape[1]*phi1.shape[2])
        y = torch.bmm(y, g1)
        #print(y.shape[1]*g1.shape[1]*g1.shape[2])

        # y = torch.bmm(phi1, g1)
        # y = torch.bmm(theta, y)
        # y = torch.matmul(theta, y)

        F_s = torch.reshape(y, (-1, self.channel//2, x.shape[2], x.shape[3]))

        res_F = self.conv4(F_s)

        return res_F+x

class GCM(nn.Module):
    def __init__(self, in_ch):
        super(GCM ,self).__init__()
        self.channel = in_ch

        self.conv1 = nn.Conv2d(self.channel, self.channel//4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.channel, self.channel//2, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.channel//4, self.channel//4, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(self.channel//2, self.channel//2, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(self.channel//2, self.channel, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #x shape: [B, C, H, W]

        x1 = self.conv1(x) # [B, C/4, H, W]
        x1 = torch.reshape(x1, [x1.shape[0], x1.shape[1], -1]) # [B, C/4, H*W]

        x2 = self.conv2(x) # [B, C/2, H, W]
        x2 = torch.reshape(x2, [x2.shape[0], x2.shape[1], -1]) # [B, C/2, H*W]
        x2 = x2.permute((0, 2, 1)) # [B, H*W, C/2]
        
        v = torch.bmm(x1, x2)
       # print(x1.shape[1]*x2.shape[1]*x2.shape[2])
        # v = torch.matmul(x1, x2) # [B, C/4, C/2]
        tmp = torch.reshape(v, (-1, v.shape[1]*v.shape[2]))
        tmp = F.softmax(tmp, dim=-1)
        v = torch.reshape(tmp, v.shape)
        v = torch.unsqueeze(v, dim=3) # [B, C/4, C/2, 1]

        n = self.conv3(v) # [B, C/4, C/2, 1]
        n = v + n # [B, C/4, C/2, 1]

        n = self.relu(n)
        n = n.permute((0, 2, 1, 3)) # [B, C/2, C/4, 1]

        n = self.conv4(n) # [B, C/2, C/4, 1]

        z = torch.squeeze(n, dim=3) # [B, C/2, C/4]
        
        y = torch.bmm(z, x1)
        #print(z.shape[1]*x1.shape[1]*x1.shape[2])
        # y = torch.matmul(z, x1) # [B, C/2, H*W] 
        y = torch.unsqueeze(y, dim=3) # [B, C/2, H*W, 1]
        y = torch.reshape(y, (y.shape[0], y.shape[1], x.shape[2], x.shape[3])) # [B, C/2, H, W]
        x_res = self.conv5(y) # [B, C, H, W]

        return x + x_res
        


class DCM(nn.Module):
    def __init__(self, channel, out_channel=None):
        super(DCM, self).__init__()

        if out_channel == None:
            out_channel = channel

        self.conv1 = conv_relu(channel, channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = conv_relu(channel, channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.conv3 = conv_relu(channel, channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv2d(channel, out_channel, kernel_size=1, stride=1, padding=0, padding_mode='replicate')
    def forward(self, x):
        c1 = self.conv1(x)
        tmp1 = c1 + x
        c2 = self.conv2(tmp1)
        tmp2 = tmp1 + c2
        c3 = self.conv3(tmp2)
        tmp3 = tmp2 + c3
        c4 = self.conv4(tmp3)

        return c4


class BlockEncoder(nn.Module):
    def __init__(self, in_ch):
        super(BlockEncoder, self).__init__()
        self.DCM = DCM(in_ch)
        self.GCM = GCM(in_ch)
    def forward(self, x):
        dcm_x = self.DCM(x)
        gcm_x = self.GCM(dcm_x)
        return x + gcm_x

class BlockDecoder(nn.Module):
    def __init__(self, in_ch):
        super(BlockDecoder, self).__init__()
        self.GSM = GSM(in_ch)
        self.DCM = DCM(in_ch)
    def forward(self, x):
        gsm_x = self.GSM(x)
        dcm_x = self.DCM(gsm_x)
        return x + dcm_x

class GRNet(nn.Module):
    def __init__(self, in_ch=25):
        super(GRNet, self).__init__()

        n1 = 64
        # filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        filters = [64, 64, 64, 64, 64]

        self.down0 = conv_relu(filters[0], filters[0], kernel_size=3, padding=1, stride=2, bias=True, padding_mode='replicate')
        self.down1 = conv_relu(filters[0], filters[0], kernel_size=3, padding=1, stride=2, bias=True, padding_mode='replicate')
        self.down2 = conv_relu(filters[0], filters[0], kernel_size=3, padding=1, stride=2, bias=True, padding_mode='replicate')
        # self.Down4 = conv_relu(filters[0], filters[0], kernel_size=3, padding=1, stride=2, bias=True, padding_mode='replicate')
        # self.Down4 = nn.Conv2d()

        self.conv0 = nn.Conv2d(in_ch, filters[0], kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True)
        self.conv1 = nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=True)
        self.encoder0 = BlockEncoder(filters[0])
        self.encoder1 = BlockEncoder(filters[1])
        self.encoder2 = BlockEncoder(filters[2])
        self.middle = BlockEncoder(filters[3])
        # self.Conv5 = BlockEncoder(filters[4])

        # self.Up5 = nn.Conv2d(filters[4]*2, filters[3], kernel_size=3, stride=1, padding=1, bias=True)
        self.up_conv2 = conv_relu(filters[2]*2, filters[2], kernel_size=1, padding=0, stride=1, bias=True)
        self.decoder2 = BlockDecoder(filters[4])

        # self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2, padding=0, bias=True)
        # self.Up4 = nn.Conv2d(filters[3]*2, filters[2], kernel_size=3, stride=1, padding=1, bias=True)
        self.up_conv1 = conv_relu(filters[1]*2, filters[1], kernel_size=1, padding=0, stride=1, bias=True_)
        self.decoder1 = BlockDecoder(filters[3])

        # self.Up3 = nn.Conv2d(filters[2]*2, filters[1], kernel_size=3, stride=1, padding=1, bias=True)
        self.up_conv0 = conv_relu(filters[0]*2, filters[0], kernel_size=1, padding=0, stride=1, bias=True)
        self.decoder0 = BlockDecoder(filters[2])

        # self.Up2 = nn.Conv2d(filters[1]*2, filters[0], kernel_size=3, stride=1, padding=1, bias=True)
        # self.Up_conv2 = BlockDecoder(filters[1])

        self.Conv = nn.Conv2d(filters[0], in_ch, kernel_size=3, padding=1, stride=1, padding_mode='replicate')

    def forward(self, x):


        basic = self.conv0(x)
        basic1 = self.conv1(basic)

        encode0 = self.encoder0(basic1)
        down0 = self.down0(encode0)

        encode1 = self.encoder1(down0)
        down1 = self.down1(encode1)

        encode2 = self.encoder2(down1)
        down2 = self.down2(encode2)

        media_end = self.middle(down2)

        deblock2 = F.upsample_bilinear(media_end, scale_factor=2)
        deblock2 = torch.cat((deblock2, encode2), dim=1)
        deblock2 = self.up_conv2(deblock2)
        deblock2 = self.decoder2(deblock2)

        deblock1 = F.upsample_bilinear(deblock2, scale_factor=2)
        deblock1 = torch.cat((deblock1, encode1), dim=1)
        deblock1 = self.up_conv1(deblock1)
        deblock1 = self.decoder1(deblock1)

        deblock0 = F.upsample_bilinear(deblock1, scale_factor=2)
        deblock0 = torch.cat((deblock0, encode0), dim=1)
        deblock0 = self.up_conv0(deblock0)
        deblock0 = self.decoder0(deblock0)


        decoding_end = deblock0 + basic
        res = self.Conv(decoding_end)
        out = x + res
        
        return out



class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))
        
        self.conv_residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, x):

        x = self.conv(x) + self.conv_residual(x)
        return x

class U_Net_GR(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=34, out_ch=34):
        super(U_Net_GR, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Down1 = nn.Conv2d(filters[0], filters[0], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down2 = nn.Conv2d(filters[1], filters[1], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down3 = nn.Conv2d(filters[2], filters[2], kernel_size=4, stride=2, padding=1, bias=True)
        self.Down4 = nn.Conv2d(filters[3], filters[3], kernel_size=4, stride=2, padding=1, bias=True)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.skip1 = nn.Conv2d(in_ch, filters[0], kernel_size=1, stride=1, padding=0)
        self.Conv2 = conv_block(filters[0], filters[1])
        self.skip2 = nn.Conv2d(filters[0], filters[1], kernel_size=1, stride=1, padding=0)
        self.Conv3 = conv_block(filters[1], filters[2])
        self.skip3 = nn.Conv2d(filters[1], filters[2], kernel_size=1, stride=1, padding=0)
        self.Conv4 = conv_block(filters[2], filters[3])
        self.skip4 = nn.Conv2d(filters[2], filters[3], kernel_size=1, stride=1, padding=0)
        self.Conv5 = conv_block(filters[3], filters[4])
        self.skip5 = nn.Conv2d(filters[3], filters[4], kernel_size=1, stride=1, padding=0)
        self.Up_conv5 = conv_block(filters[4], filters[3])
        self.skip_up5 = nn.Conv2d(filters[4], filters[3], kernel_size=1, stride=1, padding=0)
        self.Up_conv4 = conv_block(filters[3], filters[2])
        self.skip_up4 = nn.Conv2d(filters[3], filters[2], kernel_size=1, stride=1, padding=0)
        self.Up_conv3 = conv_block(filters[2], filters[1])
        self.skip_up3 = nn.Conv2d(filters[2], filters[1], kernel_size=1, stride=1, padding=0)
        self.Up_conv2 = conv_block(filters[1], filters[0])
        self.skip_up2 = nn.Conv2d(filters[1], filters[0], kernel_size=1, stride=1, padding=0)

        # self.Conv1 = DCM(in_ch, filters[0])
        # self.Conv2 = DCM(filters[0], filters[1])
        # self.Conv3 = DCM(filters[1], filters[2])
        # self.Conv4 = DCM(filters[2], filters[3])
        # self.Conv5 = DCM(filters[3], filters[4])

        # self.Up_conv5 = DCM(filters[4], filters[3])
        # self.Up_conv4 = DCM(filters[3], filters[2])
        # self.Up_conv3 = DCM(filters[2], filters[1])
        # self.Up_conv2 = DCM(filters[1], filters[0])


        self.GCM1 = GCM(filters[0])
        self.GCM2 = GCM(filters[1])
        self.GCM3 = GCM(filters[2])
        self.GCM4 = GCM(filters[3])
        self.GCM5 = GCM(filters[4])

        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2, padding=0, bias=True)
        self.GSM5 = GSM(filters[4])

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2, padding=0, bias=True)
        self.GSM4 = GSM(filters[3])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2, padding=0, bias=True)
        self.GSM3 = GSM(filters[2])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2, padding=0, bias=True)
        self.GSM2 = GSM(filters[1])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.GCM1(self.Conv1(x)) + self.skip1(x)

        e2 = self.Down1(e1)
        e2 = self.GCM2(self.Conv2(e2)) + self.skip2(e2)

        e3 = self.Down2(e2)
        e3 = self.GCM3(self.Conv3(e3)) + self.skip3(e3)

        e4 = self.Down3(e3)
        e4 = self.GCM4(self.Conv4(e4)) + self.skip4(e4)

        e5 = self.Down4(e4)
        e5 = self.GCM5(self.Conv5(e5)) + self.skip5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(self.GSM5(d5)) + self.skip_up5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(self.GSM4(d4)) + self.skip_up4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(self.GSM3(d3)) + self.skip_up3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(self.GSM2(d2)) + self.skip_up2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out+x

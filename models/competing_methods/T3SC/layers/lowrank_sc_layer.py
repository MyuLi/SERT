import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import logging

from models.competing_methods.T3SC.layers.encoding_layer import EncodingLayer
from models.competing_methods.T3SC.layers.soft_thresholding import SoftThresholding

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LowRankSCLayer(EncodingLayer):
    def __init__(
        self,
        patch_side,
        stride,
        K,
        rank,
        patch_centering,
        lbda_init,
        lbda_mode,
        beta=0,
        ssl=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.in_channels is not None
        assert self.code_size is not None
        self.patch_side = patch_side
        self.stride = stride
        self.K = K
        self.rank = rank
        self.patch_centering = patch_centering
        self.lbda_init = lbda_init
        self.lbda_mode = lbda_mode
        self.patch_size = self.in_channels * self.patch_side ** 2
        self.spat_dim = self.patch_side ** 2
        self.spec_dim = self.in_channels
        self.beta = beta
        self.ssl = ssl

        # first is spectral, second is spatial
        self.init_weights(
            [
                (self.code_size, self.spec_dim, self.rank),
                (self.code_size, self.rank, self.spat_dim),
            ]
        )

        self.thresholds = SoftThresholding(
            mode=self.lbda_mode,
            lbda_init=self.lbda_init,
            code_size=self.code_size,
            K=self.K,
        )
        if self.patch_centering and self.patch_side == 1:
            raise ValueError(
                "Patch centering and 1x1 kernel will result in null patches"
            )

        if self.patch_centering:
            ones = torch.ones(
                self.in_channels, 1, self.patch_side, self.patch_side
            )
            self.ker_mean = (ones / self.patch_side ** 2).to(device)
        self.ker_divider = torch.ones(
            1, 1, self.patch_side, self.patch_side
        ).to(device)
        self.divider = None

        if self.beta:
            self.beta_estimator = nn.Sequential(
                # layer1
                nn.Conv2d(
                    in_channels=1, out_channels=64, kernel_size=5, stride=2
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                # layer2
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=3, stride=2
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                # layer3
                nn.Conv2d(
                    in_channels=128, out_channels=1, kernel_size=3, stride=1
                ),
                nn.Sigmoid(),
            )

    def init_weights(self, shape):
        for w in ["C", "D", "W"]:
            setattr(self, w, self.init_param(shape))

    def init_param(self, shape):
        def init_tensor(shape):
            tensor = torch.empty(*shape)
            torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
            return tensor

        if isinstance(shape, list):
            return torch.nn.ParameterList([self.init_param(s) for s in shape])
        return torch.nn.Parameter(init_tensor(shape))

    def _encode(self, x, sigmas=None, ssl_idx=None, **kwargs):
        self.shape_in = x.shape
        bs, c, h, w = self.shape_in

        if self.beta:
            block = min(56, h)
            c_w = (w - block) // 2
            c_h = (h - block) // 2
            to_estimate = x[:, :, c_h : c_h + block, c_w : c_w + block].view(
                bs * c, 1, block, block
            )
            beta = 1 - self.beta_estimator(to_estimate)
            # (bs * c, 1)
            beta = beta.view(bs, c, 1, 1)
        else:
            beta = torch.ones((bs, c, 1, 1), device=x.device)

        if self.ssl:
            # discard error on bands we want to predict
            with torch.no_grad():
                mask = torch.ones_like(beta)
                mask[:, ssl_idx.long()] = 0.0

            beta = beta * mask

        if self.beta or self.ssl:
            # applying beta before or after centering is equivalent
            x = x * beta

        CT = (self.C[0] @ self.C[1]).view(
            self.code_size,
            self.in_channels,
            self.patch_side,
            self.patch_side,
        )

        if self.patch_centering:
            A = F.conv2d(x, CT - CT.mean(dim=[2, 3], keepdim=True))
            self.means = F.conv2d(x, self.ker_mean, groups=self.in_channels)
        else:
            A = F.conv2d(x, CT)

        alpha = self.thresholds(A, 0)

        D = (self.D[0] @ self.D[1]).view(
            self.code_size,
            self.in_channels,
            self.patch_side,
            self.patch_side,
        )

        for k in range(1, self.K):
            D_alpha = F.conv_transpose2d(alpha, D)
            D_alpha = D_alpha * beta
            alpha = self.thresholds(A + alpha - F.conv2d(D_alpha, CT), k)

        return alpha

    def _decode(self, alpha, **kwargs):
        W = ((self.W[0]) @ self.W[1]).view(
            self.code_size,
            self.in_channels,
            self.patch_side,
            self.patch_side,
        )

        x = F.conv_transpose2d(alpha, W)

        if self.patch_centering:
            x += F.conv_transpose2d(
                self.means,
                self.ker_mean * self.patch_side ** 2,
                groups=self.in_channels,
            )
        if self.divider is None or self.divider.shape[-2:] != (x.shape[-2:]):
            ones = torch.ones(
                1, 1, alpha.shape[2], alpha.shape[3], device=alpha.device
            ).to(alpha.device)
            self.divider = F.conv_transpose2d(ones, self.ker_divider)

        x = x / self.divider

        return x

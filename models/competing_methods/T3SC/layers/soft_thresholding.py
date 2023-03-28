import torch
import torch.nn as nn
import torch.nn.functional as F


MODES = ["SG", "SC", "MG", "MC"]


class SoftThresholding(nn.Module):
    def __init__(self, mode, lbda_init, code_size=None, K=None):
        super().__init__()
        assert mode in MODES, f"Mode {mode!r} not recognized"
        self.mode = mode

        if self.mode[1] == "C":
            # 1 lambda per channel
            lbda_shape = (1, code_size, 1, 1)
        else:
            # 1 lambda for all channels
            lbda_shape = (1, 1, 1, 1)

        if self.mode[0] == "M":
            # 1 set of lambdas per unfolding
            self.lbda = nn.ParameterList(
                [
                    nn.Parameter(lbda_init * torch.ones(*lbda_shape))
                    for _ in range(K)
                ]
            )
        else:
            # 1 set of lambdas for all unfoldings
            self.lbda = nn.Parameter(lbda_init * torch.ones(*lbda_shape))

    def forward(self, x, k=None):
        if self.mode[0] == "M":
            return self._forward(x, self.lbda[k])
        else:
            return self._forward(x, self.lbda)

    def _forward(self, x, lbda):
        return F.relu(x - lbda) - F.relu(-x - lbda)

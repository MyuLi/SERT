import logging

import torch.nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EncodingLayer(nn.Module):
    def __init__(
        self,
        in_channels=None,
        code_size=None,
        input_centering=False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.code_size = code_size
        self.input_centering = input_centering

    def forward(self, x, mode=None, **kwargs):
        assert mode in ["encode", "decode", None], f"Mode {mode!r} unknown"

        if mode in ["encode", None]:
            x = self.encode(x, **kwargs)
        if mode in ["decode", None]:
            x = self.decode(x, **kwargs)
        return x

    def encode(self, x, **kwargs):
        if self.input_centering:
            self.input_means = x.mean(dim=[2, 3], keepdim=True)
            x -= self.input_means

        x = self._encode(x, **kwargs)

        return x

    def decode(self, x, **kwargs):
        x = self._decode(x, **kwargs)

        if self.input_centering:
            x += self.input_means

        return x

    def _encode(self, x, **kwargs):
        raise NotImplementedError

    def _decode(self, x, **kwargs):
        raise NotImplementedError

import logging

import torch
import torch.nn as nn

from  models.competing_methods.T3SC import layers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MultilayerModel(nn.Module):
    def __init__(
        self,
        channels,
        layers,
        ssl=0,
        n_ssl=0,
        ckpt=None,
    ):
        super().__init__()
        self.channels = channels
        self.layers_params = layers
        self.ssl = ssl
        self.n_ssl = n_ssl
        logger.debug(f"ssl : {self.ssl}, n_ssl : {self.n_ssl}")

        self.init_layers()
        self.normalized_dict = False

        logger.info(f"Using SSL : {self.ssl}")
        self.ckpt = ckpt
        if self.ckpt is not None:
            logger.info(f"Loading ckpt {self.ckpt!r}")
            d = torch.load(self.ckpt)
            self.load_state_dict(d["state_dict"])

    def init_layers(self):
        list_layers = []
        in_channels = self.channels

        for i in range(len(self.layers_params)):
            logger.debug(f"Initializing layer {i}")
            name = self.layers_params[f"l{i}"]["name"]
            params = self.layers_params[f"l{i}"]["params"]
            layer_cls = layers.__dict__[name]
            layer = layer_cls(
                in_channels=in_channels,
                **params,
            )
            in_channels = layer.code_size

            list_layers.append(layer)
        self.layers = nn.ModuleList(list_layers)

    def forward(
        self, x, mode=None, img_id=None, sigmas=None, ssl_idx=None, **kwargs
    ):
        assert mode in ["encode", "decode", None], f"Mode {mode!r} unknown"
        x = x.float().clone()

        if mode in ["encode", None]:
            x = self.encode(x, img_id, sigmas=sigmas, ssl_idx=ssl_idx)
        if mode in ["decode", None]:
            x = self.decode(x, img_id)
        return x

    def encode(self, x, img_id, sigmas, ssl_idx):

        for layer in self.layers:
            x = layer(
                x,
                mode="encode",
                img_id=img_id,
                sigmas=sigmas,
                ssl_idx=ssl_idx,
            )
        return x

    def decode(self, x, img_id):
        for layer in self.layers[::-1]:
            x = layer(x, mode="decode", img_id=img_id)

        return x

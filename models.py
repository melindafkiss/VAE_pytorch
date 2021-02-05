# This code part is copied from 1Konny/WAE-pytorch.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import spectral_norm

import gin

import math
import numpy as np

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class mu_model(nn.Module):
    def __init__(self):
        super(mu_model, self).__init__()
    
    def forward(self, tensor):
        dim = tensor.shape[-1] // 2 
        return tensor[..., :dim]

class std_model(nn.Module):
    def __init__(self):
        super(std_model, self).__init__()

    def forward(self, tensor):
        dim = tensor.shape[-1] // 2 
        return tensor[..., dim:]

"""
class sampling_model(nn.Module):
    def __init__(self, mu, std):
        super(sampling_model, self).__init__()
        self.mu = mu
        self.std = std

    def forward(self, tensor):
        dist = torch.distributions.normal.Normal(self.mu, self.std)
        
        return torch.add(torch.mul(tensor, torch.exp(torch.mul(self.std, 0.5))), self.mu)
"""
"""
class sampling_model(nn.Module):
    def __init__(self):
        super(sampling_model, self).__init__()

    def forward(self, mu, std):
        dist = torch.distributions.normal.Normal(torch.zeros(mu, torch.exp(torch.mul(std, 0.5)))
        s = dist.rsample()
        return s
"""


class sampling_model(nn.Module):
    def __init__(self):
        super(sampling_model, self).__init__()

    def forward(self, tensor):
        mu = tensor[..., 0]
        std = tensor[..., 1]        
        base_dist = torch.distributions.normal.Normal(mu, torch.exp(torch.mul(std, 0.5)))
        dist = torch.distributions.independent.Independent(base_dist, 1)
        s = dist.rsample()
        return s
    
@gin.configurable(blacklist=["input_normalize_sym"])
class WAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=64, nc=3, distribution = 'sphere', input_normalize_sym=False):
        super(WAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.distribution = distribution
        self.input_normalize_sym = input_normalize_sym
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*4*4)),                                 # B, 1024*4*4
            nn.Linear(1024*4*4, z_dim)                            # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024*8*8),                           # B, 1024*8*8
            View((-1, 1024, 8, 8)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon, z

    def _encode(self, x):
        if self.distribution == "sphere":
            return F.normalize(self.encoder(x), dim=1, p=2)
        else:
            return self.encoder(x)

    def _decode(self, z):
        xd = self.decoder(z)
        if self.input_normalize_sym:
            return F.tanh(xd)
        else:
            return F.sigmoid(xd)


@gin.configurable(blacklist=["input_normalize_sym"])
class MlpModel(nn.Module):
    """Encoder-Decoder architecture for MNIST-like datasets."""
    def __init__(self, z_dim=10, nc=1, input_dims=(28, 28, 1),
                 input_normalize_sym=False,
                 e_num_layers=gin.REQUIRED,
                 g_num_layers=gin.REQUIRED,
                 e_num_filters=gin.REQUIRED,
                 g_num_filters=gin.REQUIRED,
                 batch_norm=gin.REQUIRED):
        super(MlpModel, self).__init__()
        self.input_dims = input_dims
        self.e_num_filters = e_num_filters
        self.g_num_filters = g_num_filters
        self.e_num_layers = e_num_layers
        self.g_num_layers = g_num_layers

        self.batch_norm = batch_norm
        self.z_dim = z_dim
        self.nc = nc
        self.input_normalize_sym = input_normalize_sym

        self.encoder = self.build_encoder_layers()
        self.decoder = self.build_decoder_layers()

        self.weight_init()

    def build_encoder_layers(self):
        self.encoder_layers = []

        #channels = self.input_dims[2]
        input_dim = np.prod(self.input_dims)
        output_dim = self.e_num_filters

        self.encoder_layers.append(nn.Flatten())

        for i in range(self.e_num_layers):
            self.encoder_layers.append(nn.Linear(input_dim, output_dim))
            if self.batch_norm:
                self.encoder_layers.append(nn.BatchNorm1d(output_dim))
            self.encoder_layers.append(nn.ReLU(True))
            input_dim = output_dim

        self.encoder_layers.append(nn.Linear(input_dim, self.z_dim * 2))
        self.encoder_layers.append(View((-1, self.z_dim, 2)))

        return nn.Sequential(*self.encoder_layers)

    def build_decoder_layers(self):

        self.decoder_layers = []

        input_dim = self.z_dim
        output_dim = self.g_num_filters

        for i in range(self.g_num_layers):
            self.decoder_layers.append(nn.Linear(input_dim, output_dim))
            if self.batch_norm:
                self.decoder_layers.append(nn.BatchNorm1d(output_dim))
            self.decoder_layers.append(nn.ReLU(True))
            input_dim = output_dim

        self.decoder_layers.append(nn.Linear(input_dim, np.prod(self.input_dims)))
        if len(self.input_dims) < 3:
            self.decoder_layers.append(View((-1, *self.input_dims)))
        else:
            channels = self.input_dims[2]
            size = self.input_dims[:2]
            self.decoder_layers.append(View((-1, channels, *size)))

        return nn.Sequential(*self.decoder_layers)


    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)
        #mu = mu_model()(z)
        #std = std_model()(z)
        #eps = Variable(torch.FloatTensor(self.z_dim).normal_(), requires_grad = True).cuda()
        #s = sampling_model(mu, std)(eps)
        s = sampling_model()(z)
        x_recon = self._decode(s)
        return x_recon

    def _encode(self, x):
        z = self.encoder(x)
        return z


    def _decode(self, z):
        xd = z
        xd = self.decoder(xd)

        if self.input_normalize_sym:
            return F.tanh(xd)
        else:
            return F.sigmoid(xd)

        
@gin.configurable(blacklist=["input_normalize_sym"])
class MnistModel(nn.Module):
    """Encoder-Decoder architecture for MINST-like datasets."""
    def __init__(self, z_dim=10, nc=1, input_dims=(28,28,1), distribution = gin.REQUIRED, input_normalize_sym=False):
        super(MnistModel, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.distribution = distribution
        self.input_normalize_sym = input_normalize_sym
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1, bias=False),              # B,  128, 14, 14
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),              # B,  256, 7, 7
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            View((-1, 64*7*7)),                                   # B, 64*4*4
            nn.Linear(64*7*7, z_dim)                            # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 64*7*7),                           # B, 64*7*7
            View((-1, 64, 7, 7)),                               # B, 64,  7,  7
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),   # B,  64, 14, 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 128, 4, 2, 1, bias=False),    # B,  256, 28, 28
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon, z

    def _encode(self, x):
        #distribution = gin.REQUIRED
        if self.distribution == "sphere":
            return F.normalize(self.encoder(x), dim=1, p=2)
        else:
            return self.encoder(x)

    def _decode(self, z):
        xd = self.decoder(z)
        if self.input_normalize_sym:
            return F.tanh(xd)
        else:
            return F.sigmoid(xd)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass

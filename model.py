import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

from networks import SpectralNorm, Self_Attn, ResBlock
import numpy as np


class GeneratorResnet(nn.Module):
    def __init__(self, opt, conv_dim=64):
        super(GeneratorResnet, self).__init__()

        layers = []
        repeat_num = int(np.log2(opt.img_size)) - 3
        mult = 2 ** repeat_num

        layers += [
            SpectralNorm(nn.ConvTranspose2d(opt.latent_dim, conv_dim * mult, 4)),
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(inplace=True)
        ]

        curr_dim = int(conv_dim * mult)

        while curr_dim > (conv_dim * 2):
            layers += [ResBlock(curr_dim, curr_dim//2)]
            curr_dim = curr_dim // 2

        # add attention to the last 2 layers
        layers += [
            Self_Attn(curr_dim),
            ResBlock(curr_dim, conv_dim),
            Self_Attn(conv_dim),
            nn.ConvTranspose2d(conv_dim, opt.channels, 4, 2, 1),
            nn.Tanh()
        ]

        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.conv_blocks(z)
        return out


class Generator(nn.Module):
    def __init__(self, opt, conv_dim=64):
        super(Generator, self).__init__()

        layers = []
        repeat_num = int(np.log2(opt.img_size)) - 3
        mult = 2 ** repeat_num

        layers += [
            SpectralNorm(nn.ConvTranspose2d(opt.latent_dim, conv_dim * mult, 4)),
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU(inplace=True),
        ]

        curr_dim = int(conv_dim * mult)

        while curr_dim > (conv_dim * 2):
            layers += [
                SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim//2, 4, 2, 1)),
                nn.BatchNorm2d(curr_dim//2),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2

        # add attention to the last 2 layers
        layers += [
            Self_Attn(curr_dim),
            SpectralNorm(nn.ConvTranspose2d(curr_dim, conv_dim, 4, 2, 1)),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(inplace=True),
            Self_Attn(conv_dim),
            nn.ConvTranspose2d(conv_dim, opt.channels, 4, 2, 1),
            nn.Tanh()
        ]

        self.conv_blocks1 = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.conv_blocks1(z)
        return out


class DiscriminatorResnet(nn.Module):
    def __init__(self, opt, conv_dim=64):
        super(DiscriminatorResnet, self).__init__()

        def resblock(in_channel, out_channel, downsample=True):
            return ResBlock(in_channel, out_channel,
                             bn=False,
                             upsample=False, downsample=downsample)

        repeat_num = int(np.log2(opt.img_size)) - 3
        mult = 2 ** repeat_num

        curr_dim = conv_dim

        self.pre_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(opt.channels, curr_dim, 3, padding=1)),
            nn.ReLU(inplace=True),
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim, 3, padding=1)),
            nn.AvgPool2d(2)
        )

        self.pre_skip = SpectralNorm(nn.Conv2d(opt.channels, curr_dim, 1))

        layers = []

        while curr_dim < ((conv_dim * mult) // 2):
            layers += [
                resblock(curr_dim, curr_dim * 2)
            ]
            curr_dim *= 2

        layers += [
            Self_Attn(curr_dim),
            resblock(curr_dim, curr_dim * 2)
        ]
        curr_dim *= 2
        layers += [
            Self_Attn(curr_dim),
            nn.Conv2d(curr_dim, 1, 4)
        ]
        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre_conv(x)
        out = out + self.pre_skip(F.avg_pool2d(x, 2))
        out = self.conv_blocks(out)
        return out.squeeze()


class Discriminator(nn.Module):
    def __init__(self, opt, conv_dim=64):
        super(Discriminator, self).__init__()

        repeat_num = int(np.log2(opt.img_size)) - 3
        mult = 2 ** repeat_num

        curr_dim = conv_dim
        layers = [
            SpectralNorm(nn.Conv2d(opt.channels, curr_dim, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True)
        ]

        while curr_dim < ((conv_dim * mult) // 2):
            layers += [
                SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
                nn.LeakyReLU(0.1, inplace=True)
            ]
            curr_dim *= 2

        layers += [
            Self_Attn(curr_dim),
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        curr_dim *= 2
        
        layers += [
            Self_Attn(curr_dim),
            nn.Conv2d(curr_dim, 1, 4)
        ]

        self.conv_blocks = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_blocks(x)
        return out.squeeze()

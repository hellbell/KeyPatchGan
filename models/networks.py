import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
This code was implemented based on Star-GAN pytorch implementation.
"""


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class PartEncoderR(nn.Module):
    def __init__(self, opts, repeat_num=6, num_downsample=2):
        super(PartEncoderR, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, opts.conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(opts.conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling
        curr_dim = opts.conv_dim
        for i in range(num_downsample):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MaskGeneratorR(nn.Module):
    def __init__(self, opts, num_updample=2):
        super(MaskGeneratorR, self).__init__()

        layers = []
        curr_dim = opts.conv_dim * np.power(2,num_updample)
        # Up-Sampling
        for i in range(num_updample):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ImageGeneratorR(nn.Module):
    def __init__(self, opts, num_updample=2):
        super(ImageGeneratorR, self).__init__()

        layers = []
        curr_dim = opts.conv_dim * np.power(2, num_updample)
        # Up-Sampling
        for i in range(num_updample):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class DiscriminatorR(nn.Module):
    def __init__(self, opts, repeat_num=6):
        super(DiscriminatorR, self).__init__()

        conv_dim = opts.conv_dim
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))

        self.model = nn.Sequential(*layers)
        # self.conv = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        # h = self.model(x)
        # out_real = self.conv(h)
        out_real = self.model(x)
        return out_real.squeeze()





#################################################################
# U-Net structure
#################################################################

class PartEncoderU(nn.Module):
    def __init__(self, opts):
        super(PartEncoderU, self).__init__()

        self.opts = opts
        self.num_conv_layers = opts.num_conv_layers

        conv_dims_in = [self.opts.c_dim]
        conv_dims_out = []

        for i in range(self.opts.num_conv_layers):
            powers = min(3, i)
            conv_dims_in.append(opts.conv_dim * np.power(2, powers))
            conv_dims_out.append(opts.conv_dim * np.power(2, powers))
        conv_dims_out.append(self.opts.part_embed_dim)

        layer = []

        for i in range(self.opts.num_conv_layers + 1):
            if i == self.opts.num_conv_layers:
                _kernel_size = np.int(self.opts.output_size / np.power(2, self.opts.num_conv_layers))
                _stride = 1
                _padding = 0
            else:
                _kernel_size = 5
                _stride = 2
                _padding = 2

            if i == 0 or i == self.opts.num_conv_layers:
                actv = nn.LeakyReLU(0.2)
            else:
                actv = nn.Sequential(nn.BatchNorm2d(conv_dims_out[i]), nn.LeakyReLU(0.2))

            conv = nn.Conv2d(conv_dims_in[i], conv_dims_out[i],
                                       kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=True)
            layer.append(nn.Sequential(conv, actv))

        model = [layer[i] for i in range(len(layer))]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        e = []
        out = x
        for i in range(len(self.model)):
            out = self.model[i](out)
            e.append(out)
        return e

class MaskGeneratorU(nn.Module):
    def __init__(self, opts):
        super(MaskGeneratorU, self).__init__()

        self.opts = opts
        conv_dims_in = [opts.part_embed_dim]
        conv_dims_out = []

        for i in range(self.opts.num_conv_layers):
            powers = min(3, self.opts.num_conv_layers - 1 - i)
            conv_dims_in.append(opts.conv_dim * np.power(2, powers) * 2)
            conv_dims_out.append(opts.conv_dim * np.power(2, powers))
        conv_dims_out.append(1)

        layer = []

        for i in range(self.opts.num_conv_layers + 1):
            if i == 0:
                _kernel_size = np.int(self.opts.output_size / np.power(2, self.opts.num_conv_layers))
                _stride = 1
                _padding = 0
            else:
                _kernel_size = 5
                _stride = 2
                _padding = 2

            if i == self.opts.num_conv_layers:
                actv = nn.Sigmoid()
            else:
                actv = nn.Sequential(nn.BatchNorm2d(conv_dims_out[i]), nn.ReLU())

            convT = nn.ConvTranspose2d(conv_dims_in[i], conv_dims_out[i],
                                    kernel_size=_kernel_size, stride=_stride, padding=_padding,
                                    bias=True)

            layer.append(convT)
            layer.append(actv)

        model = [layer[i] for i in range(len(layer))]
        self.model = nn.Sequential(*model)

    def forward(self, parts_enc):

        len_parts_enc = len(parts_enc)
        _output_size = []
        for i in range(len_parts_enc):
            if i == 0:
                _output_size.append(4)
            else:
                _output_size.append(_output_size[i - 1] * 2)

        m = []
        out = parts_enc[-1]

        for i, layer in enumerate(self.model):
            if i % 2 == 0:
                # convTranspose layer
                out = layer(out, output_size=[_output_size[i/2], _output_size[i/2]])
            else:
                # activation layer
                out = layer(out)
                if i < (len(self.model)-1):
                    # concatenate
                    out = torch.cat([out, parts_enc[-2 - (i-1)/2]], 1)
                m.append(out)
        return m

class ImageGeneratorU(nn.Module):
    def __init__(self, opts):
        super(ImageGeneratorU, self).__init__()

        self.opts = opts
        conv_dims_in = [opts.part_embed_dim + opts.z_dim]
        conv_dims_out = []

        for i in range(self.opts.num_conv_layers):
            powers = min(3, self.opts.num_conv_layers - 1 - i)
            conv_dims_in.append(opts.conv_dim * np.power(2, powers) * 3)
            conv_dims_out.append(opts.conv_dim * np.power(2, powers))
        conv_dims_out.append(opts.c_dim)

        layer = []
        for i in range(self.opts.num_conv_layers + 1):
            if i == 0:
                _kernel_size = np.int(self.opts.output_size / np.power(2, self.opts.num_conv_layers))
                _stride = 1
                _padding = 0
            else:
                _kernel_size = 5
                _stride = 2
                _padding = 2

            if i == self.opts.num_conv_layers:
                actv = nn.Tanh()
            else:
                actv = nn.Sequential(nn.BatchNorm2d(conv_dims_out[i]), nn.ReLU())

            convT = nn.ConvTranspose2d(conv_dims_in[i], conv_dims_out[i],
                                                 kernel_size=_kernel_size, stride=_stride, padding=_padding,
                                                 bias=True)

            layer.append(convT)
            layer.append(actv)

        model = [layer[i] for i in range(len(layer))]
        self.model = nn.Sequential(*model)

    def forward(self, embed, z, m):
        len_m = len(m)
        _output_size = []
        for i in range(len_m):
            if i == 0:
                _output_size.append(4)
            else:
                _output_size.append(_output_size[i - 1] * 2)

        g = []
        out = torch.cat([embed, z], 1)
        for i, layer in enumerate(self.model):
            if i % 2 == 0:
                # convTranspose layer
                out = layer(out, output_size=[_output_size[i/2], _output_size[i/2]])
            else:
                # activation layer
                out = layer(out)
                if i < (len(self.model)-1):
                    # concatenate
                    out = torch.cat([out, m[(i-1)/2]], 1)
                g.append(out)

        return g


class DiscriminatorU(nn.Module):
    def __init__(self, opts):
        super(DiscriminatorU, self).__init__()

        self.opts = opts
        self.num_conv_layers = opts.num_conv_layers

        conv_dims_in = [self.opts.c_dim]
        conv_dims_out = []

        for i in range(self.opts.num_conv_layers):
            powers = min(3, i)
            conv_dims_in.append(opts.conv_dim * np.power(2, powers))
            conv_dims_out.append(opts.conv_dim * np.power(2, powers))
        conv_dims_out.append(1)

        layer = []

        for i in range(self.opts.num_conv_layers + 1):

            if i == self.opts.num_conv_layers:
                _kernel_size = np.int(self.opts.output_size / np.power(2, self.opts.num_conv_layers))
                _stride = 1
                _padding = 0
            else:
                _kernel_size = 5
                _stride = 2
                _padding = 2

            if i == 0:
                actv = nn.LeakyReLU(0.2)
            elif i == self.opts.num_conv_layers:
                actv = nn.Sigmoid()
            else:
                actv = nn.Sequential(nn.BatchNorm2d(conv_dims_out[i]), nn.LeakyReLU(0.2))

            conv = nn.Conv2d(conv_dims_in[i], conv_dims_out[i],
                                       kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=True)
            layer.append(conv)
            layer.append(actv)

        model = [layer[i] for i in range(len(layer))]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
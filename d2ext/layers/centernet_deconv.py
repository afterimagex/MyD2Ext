import torch
import torch.nn as nn

from detectron2.layers import get_norm, ModulatedDeformConv, Conv2d, DeformConv


class DeformConvWithOffset(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            deformable_groups=1,
            norm=None,
            activation=None,
    ):
        super(DeformConvWithOffset, self).__init__()
        self.dcn = DeformConv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            deformable_groups,
            norm=norm,
            activation=activation,
        )
        self.offset = Conv2d(
            in_channels,
            deformable_groups * 2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        offset = self.offset(x)
        output = self.dcn(x, offset)
        return output


class ModulatedDeformConvWithOffset(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            deformable_groups=1,
            norm=None,
            activation=None,
    ):
        super(ModulatedDeformConvWithOffset, self).__init__()
        self.dcn = ModulatedDeformConv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            deformable_groups,
            norm=norm,
            activation=activation,
        )
        self.mask = Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        mask = self.mask(x)
        o1, o2, mask = torch.chunk(mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.dcn(x, offset, mask)
        return output


class DeconvLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            padding=1,
            output_padding=0,
            deform=False,
            deform_modulate=False,
            norm=None,
    ):
        super(DeconvLayer, self).__init__()

        if deform:
            if deform_modulate:
                self.conv = ModulatedDeformConvWithOffset(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    norm=get_norm(norm, out_channels),
                    activation=nn.ReLU(),
                )
            else:
                self.conv = DeformConvWithOffset(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    norm=get_norm(norm, out_channels),
                    activation=nn.ReLU(),
                )
        else:
            self.conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, out_channels),
                activation=nn.ReLU(),
            )
        self.up_sample = torch.nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=not norm,
        )
        self.norm = get_norm(norm, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.relu(x)
        x = self.up_sample(x)
        return x

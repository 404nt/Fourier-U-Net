import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

''''
complex unet blocks etc.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexConv2d, ComplexMaxPool2d, ComplexReLU, ComplexDropout, ComplexLinear, ComplexBatchNorm2d, NaiveComplexBatchNorm2d, ComplexConvTranspose2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d, complex_upsample, complex_dropout
'''
complex activation funciton
ReLU --> ComplexReLU
LeakyReLU --> ComplexLeakyReLU
RReLU --> ComplexRReLU
PReLU --> ComplexPReLU
'''
class ComplexReLU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input.real) + 1.j * F.relu(input.imag)

class ComplexLeakyReLU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(input.real).type(torch.complex64) + 1j*F.leaky_relu(input.imag).type(torch.complex64)

class ComplexRReLU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.rrelu(input.real) + 1.j*F.rrelu(input.imag)

class ComplexPReLU(nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super(ComplexPReLU, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.prelu(input.real, self.weight) + 1j*F.prelu(input.imag, self.weight)

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


def complex_double_conv(in_channels, out_channels):
    return nn.Sequential(
        ComplexConv2d(in_channels, out_channels, 3, padding=1),
        NaiveComplexBatchNorm2d(out_channels),
        ComplexLeakyReLU(),
        ComplexConv2d(out_channels, out_channels, 3, padding=1),
        NaiveComplexBatchNorm2d(out_channels),
        ComplexLeakyReLU()
    )

class complexconv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=1):
            super(complexconv, self).__init__()
            self.conv = ComplexConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups)
            self.bn = ComplexBatchNorm2d(out_channels)
            self.relu = ComplexReLU()
            self.leakyrelu = ComplexLeakyReLU()
            self.rrelu = ComplexRReLU()
            self.prelu = ComplexPReLU()
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            # x = self.leakyrelu(x)
            x = self.prelu(x)
            return x


def strided_conv(in_channels):
    return nn.Sequential(
        ComplexConv2d(in_channels, in_channels, 2, padding=0, stride=2),
        NaiveComplexBatchNorm2d(in_channels),
        ComplexLeakyReLU()
    )

def strided_deconv(in_channels):
    return nn.Sequential(
        ComplexConvTranspose2d(in_channels, in_channels, 4, padding=1, stride=2),
        NaiveComplexBatchNorm2d(in_channels),
        ComplexLeakyReLU()
    )

class FourierTransform_Coe(torch.nn.Module):
    def __init__(self):
        super(FourierTransform_Coe, self).__init__()
    def transform(self, input_data):
        fourier = torch.fft.fft2(input_data)
        return fourier

    def inverse(self, fourier):
        inverse_transform = torch.fft.ifft2(fourier)
        inverse_transform = torch.abs(inverse_transform)
        return inverse_transform.to(torch.float)

    def forward(self, input_data):
        self.fourier = self.transform(input_data)
        reconstruction = self.inverse(self.fourier)
        return reconstruction

class FourierTransform_AP(torch.nn.Module):
    def __init__(self):
        super(FourierTransform_AP, self).__init__()

    def transform(self, input_data):
        f = torch.fft.fft2(input_data)
        fshift = torch.fft.fftshift(f)
        amplitude = torch.absolute(fshift)
        phase = torch.angle(fshift)
        return amplitude, phase

    def inverse(self, amplitude, phase):
        ishift = torch.fft.ifftshift(amplitude * torch.exp(1j*phase))
        iimg = torch.fft.ifft2(ishift)
        iimg = torch.absolute(iimg)

        return iimg.to(torch.float)

    def forward(self, input_data):
        self.amplitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.amplitude, self.phase)
        return reconstruction
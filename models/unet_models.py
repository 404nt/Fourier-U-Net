from complexPyTorch.complexLayers import *
from complexPyTorch.complexFunctions import *
from models.unet_parts import *
import torch.nn as nn
# class fcn(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(fcn, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes

#         self.conv0 = nn.Conv2d(in_channels=3, out_channels=n_channels,
#                                     kernel_size=3, stride=1, padding=1)
#         self.bn0 = nn.BatchNorm2d(self.n_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.leakyrelu = ComplexLeakyReLU()

#         self.fft = FourierTransform_Coe()

#         self.conv1 = ComplexConv2d(3, 16, 3, padding=1, stride=1)
#         self.conv2 = ComplexConv2d(16, 32, 3, padding=1, stride=1)
#         self.conv3 = ComplexConv2d(32, 16, 3, padding=1, stride=1)
#         # self.conv4 = ComplexConv2d(16, 3, 3, padding=1, stride=1)
#         self.out = nn.Conv2d(3, n_classes, 1)
#     def forward(self, x):
#         if self.n_channels != 3:
#             x = self.conv0(x)
#             x = self.bn0(x)
#             x = self.relu(x)
#         x = self.fft.transform(x)
#         x = self.conv1(x)
#         x = self.leakyrelu(x)
#         x = self.conv2(x)
#         x = self.leakyrelu(x)
#         x = self.conv3(x)
#         x = self.leakyrelu(x)
#         # x = self.conv4(x)
#         # x = self.leakyrelu(x)
#         x = self.fft.inverse(x)
#         logits = self.out(x)
#         return logits
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class test_unet(nn.Module):
    def __init__(self, n_channels, n_classes, n_depth):
        super(test_unet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_depth = n_depth

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=n_channels,
                                    kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fft = FourierTransform_Coe()

        self.dconv_down1 = complex_double_conv(n_channels, n_depth[0])
        self.pool1 = strided_conv(n_depth[0])

        self.dconv_down2 = complex_double_conv(n_depth[0], n_depth[1])
        self.pool2 = strided_conv(n_depth[1])

        self.dconv_down3 = complex_double_conv(n_depth[1], n_depth[2])
        self.pool3 = strided_conv(n_depth[2])

        self.dconv_down4 = complex_double_conv(n_depth[2], n_depth[3])
        
        self.upsamp3 = strided_deconv(n_depth[3])
        self.dconv_up3 = complex_double_conv(n_depth[2] + n_depth[3], n_depth[2])

        self.upsamp2 = strided_deconv(n_depth[2])
        self.dconv_up2 = complex_double_conv(n_depth[1] + n_depth[2], n_depth[1])

        self.upsamp1 = strided_deconv(n_depth[1])
        self.dconv_up1 = complex_double_conv(n_depth[0] + n_depth[1], n_depth[0])

        # self.conv_last = ComplexConv2d(n_depth[0], n_classes, 1)
        self.conv_last = nn.Conv2d(n_depth[0], n_classes, 1)

        # self.maxpool = ComplexMaxPool2d(2)
        '''
        maxpool用於下採樣
        upsample用於上採樣
        '''
    def to_fourier(self, x):
        return self.fft.transform(x)
    
    def to_spatial(self, x):
        return self.fft.inverse(x)

    def forward(self, x):
        if self.n_channels != 3:
            x = self.conv0(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.to_fourier(x)

        conv1 = self.dconv_down1(x)
        # x = self.maxpool(conv1)
        x = self.pool1(conv1)

        conv2 = self.dconv_down2(x)
        # x = self.maxpool(conv2)
        x = self.pool2(conv2)
        
        conv3 = self.dconv_down3(x)
        # x = self.maxpool(conv3)
        x = self.pool3(conv3)
        
        x = self.dconv_down4(x)
        
        # x = complex_upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.upsamp3(x)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        # x = complex_upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.upsamp2(x)     
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        # x = complex_upsample(x, scale_factor=2, mode='bilinear', align_corners=True)     
        x = self.upsamp1(x)
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        x = self.to_spatial(x)

        out = self.conv_last(x)
        # out = self.to_spatial(x)
        return out

def testunet16(n_channels:int=3, n_classes:int=2):
    return test_unet(n_channels=n_channels, n_classes=n_classes, n_depth=[16, 32, 64, 128])

def testunet32(n_channels:int=3, n_classes:int=2):
    return test_unet(n_channels=n_channels, n_classes=n_classes, n_depth=[32, 64, 128, 256])

class tini_unet(nn.Module):
    def __init__(self, n_channels, n_classes, expand_dim=False):
        super(tini_unet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.expand_dim = expand_dim

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=n_channels,
                                    kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fft = FourierTransform_Coe()

        self.dconv_down1 = complex_double_conv(n_channels, 16)
        self.pool1 = strided_conv(16)

        self.dconv_down2 = complex_double_conv(16, 32)
        
        self.upsamp1 = strided_deconv(32)
        self.dconv_up1 = complex_double_conv(16 + 32, 16)

        # self.conv_last = ComplexConv2d(n_depth[0], n_classes, 1)
        self.conv_last = nn.Conv2d(16, n_classes, 1)

        # self.maxpool = ComplexMaxPool2d(2)
        '''
        maxpool用於下採樣
        upsample用於上採樣
        '''
    def to_fourier(self, x):
        return self.fft.transform(x)
    
    def to_spatial(self, x):
        return self.fft.inverse(x)

    def forward(self, x):
        if self.n_channels != 3:
            x = self.conv0(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.to_fourier(x)

        conv1 = self.dconv_down1(x)
        x = self.pool1(conv1)
        
        x = self.dconv_down2(x)

        x = self.upsamp1(x)
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        x = self.to_spatial(x)

        out = self.conv_last(x)
        return out

class fc1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(fc1, self).__init__
    
    def forward(self, x):
        return x

class tini_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(tini_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        factor = 2 if bilinear else 1
        self.down4 = Down(64, 128 // factor)
        self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16 // factor, bilinear)
        self.up4 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class fcn(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(fcn, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.fft = FourierTransform_Coe()
        self.conv1 = complexconv(in_channels=3, out_channels=32, groups=1)
        self.conv2 = complexconv(in_channels=32, out_channels=n_channels, groups=1)
        self.unet = tini_UNet(n_channels=n_channels, n_classes=n_classes)

    def forward(self, x):
        x = self.fft.transform(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fft.inverse(x)
        logits = self.unet(x)
        return logits
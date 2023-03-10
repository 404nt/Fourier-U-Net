import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

    
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True): #ratio=2 default, ratio=6 for gpunet
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            #nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.Conv2d(init_channels, init_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False), #for gpunet
            #nn.BatchNorm2d(new_channels),
            nn.BatchNorm2d(init_channels), #for gpunet
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        '''
        self.atrous_block1 = nn.Sequential(
            #nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.Conv2d(init_channels, init_channels, 1, 1, groups=init_channels, bias=False),
            #nn.Conv2d(init_channels, init_channels, 1, 1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.atrous_block6 = nn.Sequential(
            #nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.Conv2d(init_channels, init_channels, 3, 1, padding=6, dilation=6, groups=init_channels, bias=False),
            #nn.Conv2d(init_channels, init_channels, 3, 1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.atrous_block12 = nn.Sequential(
            #nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.Conv2d(init_channels, init_channels, 3, 1, padding=12, dilation=12, groups=init_channels, bias=False),
            #nn.Conv2d(init_channels, init_channels, 3, 1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.atrous_block18 = nn.Sequential(
            #nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.Conv2d(init_channels, init_channels, 3, 1, padding=18, dilation=18, groups=init_channels, bias=False),
            #nn.Conv2d(init_channels, init_channels, 3, 1, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        #self.conv_1x1_output = nn.Conv2d(new_channels * 6, oup, 1, 1)
        '''

    #'''
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]
    #'''

    '''
    ### aspp+ghost module
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        atrous_block1 = self.atrous_block1(x1)
        atrous_block6 = self.atrous_block6(x1)
        atrous_block12 = self.atrous_block12(x1)
        atrous_block18 = self.atrous_block18(x1)
        #out = self.conv_1x1_output(torch.cat([x1,x2,atrous_block1,atrous_block6,atrous_block12,atrous_block18], dim=1))
        out = torch.cat([x1,x2,atrous_block1,atrous_block6,atrous_block12,atrous_block18], dim=1)
        return out[:,:self.oup,:,:]
    '''

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class GhostU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(GhostU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        #self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv1 = GhostBottleneck(img_ch, 64, 64)

        #self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv2 = GhostBottleneck(64, 128, 128)

        #self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv3 = GhostBottleneck(128, 256, 256)

        #self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv4 = GhostBottleneck(256, 512, 512)

        #self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        self.Conv5 = GhostBottleneck(512, 1024, 1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class GhostU_Net1(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(GhostU_Net1,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        #self.Conv1 = GhostBottleneck(img_ch, 64, 64)

        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        #self.Conv2 = GhostBottleneck(64, 128, 128)

        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        #self.Conv3 = GhostBottleneck(128, 256, 256)

        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        #self.Conv4 = GhostBottleneck(256, 512, 512)

        self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        #self.Conv5 = GhostBottleneck(512, 1024, 1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        #self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up_conv5 = GhostBottleneck(1024, 512, 512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        #self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up_conv4 = GhostBottleneck(512, 256, 256)        

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        #self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up_conv3 = GhostBottleneck(256, 128, 128)          

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        #self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Up_conv2 = GhostBottleneck(128, 64, 64) 

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class GhostU_Net2(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(GhostU_Net2,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        #self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv1 = GhostBottleneck(img_ch, 64, 64)

        #self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv2 = GhostBottleneck(64, 128, 128)

        #self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv3 = GhostBottleneck(128, 256, 256)

        #self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv4 = GhostBottleneck(256, 512, 512)

        #self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        self.Conv5 = GhostBottleneck(512, 1024, 1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        #self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up_conv5 = GhostBottleneck(1024, 512, 512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        #self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up_conv4 = GhostBottleneck(512, 256, 256)        

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        #self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up_conv3 = GhostBottleneck(256, 128, 128)          

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        #self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Up_conv2 = GhostBottleneck(128, 64, 64) 

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return 
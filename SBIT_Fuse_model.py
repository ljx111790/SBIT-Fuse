import torch.nn as nn
import torch
from swin_transformer import SwinTransformer
import torch.nn.functional as F

class SBIT_Fuse(nn.Module):
    def __init__(self):
        super(SBIT_Fuse, self).__init__()

        channelsl=[1,16,32,64,128,256]

        self.IR_1 = nn.Sequential(
            nn.Conv2d(channelsl[0], channelsl[1], 3, 1, 1),
            nn.Conv2d(channelsl[1], channelsl[2], 3, 1, 1),
            nn.BatchNorm2d(channelsl[2])
        )

        self.VI_1 = nn.Sequential(
            nn.Conv2d(channelsl[0], channelsl[1], 3, 1, 1),
            nn.Conv2d(channelsl[1], channelsl[2], 3, 1, 1),
            nn.BatchNorm2d( channelsl[2])
        )

        self.CDAI_IR_5 = nn.Sequential(
            nn.Conv2d(channelsl[4], channelsl[3], 3, 1, 1),
            nn.Conv2d(channelsl[3], channelsl[2], 3, 1, 1),
            nn.BatchNorm2d(channelsl[2])
        )
        self.CDAI_VI_5 = nn.Sequential(
            nn.Conv2d(channelsl[4], channelsl[3], 3, 1, 1),
            nn.Conv2d(channelsl[3], channelsl[2], 3, 1, 1),
            nn.BatchNorm2d(channelsl[2])
        )

        self.lossconv_ir = nn.Conv2d(channelsl[2], channelsl[0], 3, 1, 1)
        self.lossconv_vi = nn.Conv2d(channelsl[2], channelsl[0], 3, 1, 1)

        self.CDAI_ReLUBlock_ir_1 = CDAI_ReLUBlock(channelsl[2], channelsl[3])
        self.CDAI_ReLUBlock_vi_1 = CDAI_ReLUBlock(channelsl[2], channelsl[3])

        self.CDAI_ReLUBlock_ir_2 = CDAI_ReLUBlock(channelsl[2], channelsl[4])
        self.CDAI_ReLUBlock_vi_2 = CDAI_ReLUBlock(channelsl[2], channelsl[4])

        self.CDAI_ReLUBlock_ir_3 = CDAI_ReLUBlock(channelsl[4], channelsl[3])
        self.CDAI_ReLUBlock_vi_3 = CDAI_ReLUBlock(channelsl[4], channelsl[3])

        self.attention_ir = Attention(channelsl[4])
        self.attention_vi = Attention(channelsl[4])

        self.reconstruction = nn.Sequential(

            nn.Conv2d(channelsl[2]*2+32, channelsl[3], 3, 1, 1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(channelsl[3],channelsl[2], 3, 1, 1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(channelsl[2], channelsl[0], 3, 1, 1), nn.LeakyReLU(inplace=True),
        )
        self.sw_T = SwinTransformer(in_chans=2, num_classes=32)

    def forward(self, x, y):
        xy = torch.cat([x, y], 1)

        sw_xy = self.sw_T(xy)
        sw_xy = F.softmax(sw_xy, dim=1)

        conv_ir_1 = self.IR_1(x)
        conv_vi_1 = self.VI_1(y)

        conv_ir_2, conv_ir_2_nl = self.CDAI_ReLUBlock_ir_1(conv_ir_1)
        conv_vi_2, conv_vi_2_nl = self.CDAI_ReLUBlock_ir_1(conv_vi_1)

        conv_ir_3, conv_ir_3_nl = self.CDAI_ReLUBlock_ir_2(conv_ir_2, conv_vi_2_nl)
        conv_vi_3, conv_vi_3_nl = self.CDAI_ReLUBlock_vi_2(conv_vi_2, conv_ir_2_nl)

        conv_ir_4, conv_ir_4_nl = self.CDAI_ReLUBlock_ir_3(conv_ir_3, conv_vi_3_nl)
        conv_vi_4, conv_vi_4_nl = self.CDAI_ReLUBlock_vi_3(conv_vi_3, conv_ir_3_nl)

        Attention_ir = self.attention_ir(conv_ir_4, conv_vi_4_nl)
        Attention_vi = self.attention_vi(conv_vi_4, conv_ir_4_nl)

        conv_ir_5 = self.CDAI_IR_5(Attention_ir)
        conv_vi_5 = self.CDAI_VI_5(Attention_vi)

        lossir = self.lossconv_ir(conv_ir_5)
        lossvi = self.lossconv_vi(conv_vi_5)

        fusion = torch.cat([conv_ir_5, conv_vi_5, sw_xy], 1)

        out = self.reconstruction(fusion)

        return out, lossir, lossvi

class CDAI_ReLUBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, stride=1,
                 norm=False, bias=True, fusion=True, fusion_rate=2, dilation=1):
        super(CDAI_ReLUBlock, self).__init__()

        padding = padding or dilation * (kernel_size - 1) // 2
        if fusion:
            self.fusion = nn.Sequential(
                nn.Conv2d(in_channels * fusion_rate, in_channels, kernel_size=1, dilation=dilation),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          padding=padding, stride=stride, bias=bias, dilation=dilation)
            )

        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.relu = nn.ReLU()

    def forward(self, input_features, additional_features=None):
        if additional_features is not None:
            input_features = torch.cat([input_features, additional_features], dim=1)
            input_features = self.fusion(input_features)

        if self.norm:
            input_features = self.norm(input_features)

        out_features = self.relu(input_features)
        return out_features, input_features - out_features


class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            PALayer(channel),
            CALayer(channel)
        )

    def forward(self, x, y):
        z = torch.cat([x, y], 1)
        z = self.attention(z)
        return z


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


if __name__ == "__main__":
    model = SBIT_Fuse()
    x = torch.randn(1, 1, 224, 224)
    y = torch.randn(1, 1, 224, 224)
    out = model(x, y)
    print(out.shape)

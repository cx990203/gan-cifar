import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSampleBlk(nn.Module):

    def __init__(self, in_channel, out_channel, inplace=True):
        super(DownSampleBlk, self).__init__()
        # 获取模块参数
        self.in_channel = in_channel
        self.out_channel = out_channel
        # b, c_in, h, w -> b, c_in, h/2, w/2
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=inplace)
        )
        # b, c_in, h/2, w/2 -> b, c_out, h/2, w/2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace)
        )
        # b, c_out, h/2, w/2 -> b, c_out, h/2, w/2
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        x = self.down_sample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpSampleBlk(nn.Module):

    def __init__(self, in_channel, out_channel, inplace=True):
        super(UpSampleBlk, self).__init__()
        # 获取模块参数
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x1, x2):
        # 获取目标大小
        b, c, h, w = x2.size()
        # 上采样
        x1 = F.interpolate(x1, (h, w), None, 'bilinear', True)
        # 特征合成
        feature = torch.cat([x1, x2], dim=1)
        # 特征卷积
        feature = self.conv1(feature)
        feature = self.conv2(feature)
        return feature


class Generator(nn.Module):

    def __init__(self, input_size, out_channel, inplace=True):
        # 获取输入信息
        super(Generator, self).__init__()
        c, h, w = input_size
        # 头部处理网络: b, c, h, w -> b, 16, h, w
        self.head = nn.Sequential(
            # 3, h, w -> 32, h, w
            nn.Conv2d(c, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=inplace),
            # 32, h, w -> 32, h, w
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=inplace),
            # 32, h, w -> 16, h, w
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=inplace),
        )
        # 下采样模块1: b, 16, h, w -> b, 32, h/2, w/2
        self.down_blk1 = DownSampleBlk(16, 32, inplace=inplace)
        # 下采样模块2: b, 32, h/2, w/2 -> b, 64, h/4, w/4
        self.down_blk2 = DownSampleBlk(32, 64, inplace=inplace)
        # 下采样模块3: b, 64, h/4, w/4 -> b, 128, h/8, w/8
        self.down_blk3 = DownSampleBlk(64, 128, inplace=inplace)
        # 深层特征卷积层: b, 128, h/8, w/8 -> b, 128, h/8, w/8
        self.feature_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=inplace),
        )
        # 上采样模块1: b, 128, h/8, w/8 + b, 64, h/4, w/4 -> b, 64, h/4, w/4
        self.up_blk1 = UpSampleBlk(128 + 64, 64, inplace=inplace)
        # 上采样模块2: b, 64, h/4, w/4 + b, 32, h/2, w/2 -> b, 32, h/2, w/2
        self.up_blk2 = UpSampleBlk(64 + 32, 32, inplace=inplace)
        # 上采样模块3: b, 32, h/2, w/2 + b, 16, h, w -> b, 16, h, w
        self.up_blk3 = UpSampleBlk(32 + 16, 16, inplace=inplace)
        # 输出层
        self.out_layer = nn.Sequential(
            nn.Conv2d(16, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        # 头部处理网络: b, c, h, w -> b, 16, h, w
        x = self.head(x)
        # 下采样模块1: b, 16, h, w -> b, 32, h/2, w/2
        down1 = self.down_blk1(x)
        # 下采样模块2: b, 32, h/2, w/2 -> b, 64, h/4, w/4
        down2 = self.down_blk2(down1)
        # 下采样模块3: b, 64, h/4, w/4 -> b, 128, h/8, w/8
        deep_feature = self.down_blk3(down2)
        # 深层特征卷积: b, 128, h/8, w/8 -> b, 128, h/8, w/8
        deep_feature = self.feature_conv(deep_feature)
        # 上采样模块1: b, 128, h/8, w/8 + b, 64, h/4, w/4 -> b, 64, h/4, w/4
        up1 = self.up_blk1(deep_feature, down2)
        # 上采样模块2：b, 64, h/4, w/4 + b, 32, h/2, w/2 -> b, 32, h/2, w/2
        up2 = self.up_blk2(up1, down1)
        # 上采样模块3: b, 32, h/2, w/2 + b, 16, h, w -> b, 16, h, w
        res = self.up_blk3(up2, x)
        # 输出层
        res = self.out_layer(res)
        return res


class Generator_Transpose(nn.Module):

    def __init__(self, in_channel, inplace=True):
        super(Generator_Transpose, self).__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=inplace)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=inplace)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=inplace)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=inplace)
        )
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        up1 = self.up1(x)
        up2 = self.up2(up1)
        up3 = self.up3(up2)
        up4 = self.up4(up3)
        out = self.out_layer(up4)
        return out


if __name__ == '__main__':
    x = torch.randn([64, 64, 1, 1])
    model = Generator_Transpose(in_channel=x.size()[1])
    pred = model(x)
    print(pred.size())

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in: 输入通道
        :param ch_out: 输出通道
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        res = F.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        res = res + self.extra(x)
        return res


class DiscriminatorResnet(nn.Module):

    def __init__(self, input_size, inplace=True):
        super(DiscriminatorResnet, self).__init__()
        # 获取图像输入大小
        c, h, w = input_size
        # 头部处理层
        self.head = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=inplace)
        )
        # resblk1
        self.blk1 = ResBlk(64, 128, stride=2)
        self.blk2 = ResBlk(128, 256, stride=2)
        self.blk3 = ResBlk(256, 512, stride=2)
        self.blk4 = ResBlk(512, 512, stride=2)

        self.outlayer = nn.Sequential(
            nn.Linear(512 * 1 * 1, 2)
        )

    def forward(self, x):
        x = self.head(x)

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        x = torch.sigmoid(x)
        return x


class DiscriminatorConv(nn.Module):

    def __init__(self, input_size, inplace=True):
        super(DiscriminatorConv, self).__init__()
        # 获取图像输入大小: 3x32x32
        c, h, w = input_size
        # 定义卷积层
        self.conv_layer = nn.Sequential(
            # b, 3, 32, 32 -> b, 64, 16, 16
            nn.Conv2d(c, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # b, 64, 16, 16 -> b, 128, 8, 8
            nn.LeakyReLU(0.2, inplace=inplace),
            nn.Conv2d(64, 64 * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=inplace),
            # b, 128, 8, 8 -> b, 256, 4, 4
            nn.Conv2d(64 * 2, 64 * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=inplace),
            # b, 256, 4, 4 -> b, 2, 2, 2
            nn.Conv2d(64 * 4, 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            # b, 256, 2, 2 -> b, 2, 1, 1
            nn.MaxPool2d(kernel_size=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 获取图像大小
        b, c, h, w = x.size()
        # 经过卷积层
        x = self.conv_layer(x)
        x = x.view(b, -1)
        return x


class DiscriminatorLinear(nn.Module):

    def __init__(self, input_size, inplace=True):
        super(DiscriminatorLinear, self).__init__()
        # 获取图像输入大小
        c, h, w = input_size
        # 定义全连接层
        self.linear_layer = nn.Sequential(
            nn.Linear(c * h * w, 512),
            nn.ReLU(inplace=inplace),
            nn.Linear(512, 256),
            nn.ReLU(inplace=inplace),
            nn.Linear(256, 128),
            nn.ReLU(inplace=inplace),
            nn.Linear(128, 64),
            nn.ReLU(inplace=inplace),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, -1)
        x = self.linear_layer(x)
        return x


if __name__ == '__main__':
    data = torch.randn([2, 3, 32, 32])
    model = DiscriminatorConv(input_size=[3, 32, 32])
    out = model(data)
    print(out.size())

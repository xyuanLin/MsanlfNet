import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from thop import profile
import torch.nn.functional as F


class FCN_8s_vgg16(nn.Module):
    def __init__(self, num_classes):
        super(FCN_8s_vgg16, self).__init__()
        pretrained_net = models.vgg16(pretrained=False)
        conv_sequential = list(pretrained_net.children())[0]
        modules_list = []
        for i in range(17):
            modules_list.append(conv_sequential._modules[str(i)])
        self.stage1 = nn.Sequential(*modules_list)

        modules_list = []
        for i in range(17, 24):
            modules_list.append(conv_sequential._modules[str(i)])
        self.stage2 = nn.Sequential(*modules_list)

        modules_list = []
        for i in range(24, 31):
            modules_list.append(conv_sequential._modules[str(i)])
        modules_list.append(nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=(1, 1), stride=(1, 1)))
        modules_list.append(nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=(1, 1), stride=(1, 1)))
        self.stage3 = nn.Sequential(*modules_list)

        self.scores3 = nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=(1, 1))
        self.scores2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=(1, 1))
        self.scores1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=(1, 1))

    def forward(self, x):
        stage_2 = self.stage1(x)
        stage_3 = self.stage2(stage_2)
        stage_4 = self.stage3(stage_3)

        stage_4 = self.scores3(stage_4)
        stage_4 = F.interpolate(stage_4, scale_factor=2, mode='bilinear', align_corners=True)

        stage_3 = self.scores2(stage_3)
        stage_3 = stage_3 + stage_4
        stage_3 = F.interpolate(stage_3, scale_factor=2, mode='bilinear', align_corners=True)

        stage_2 = self.scores1(stage_2)
        s = stage_2 + stage_3
        s = F.interpolate(s, scale_factor=8, mode='bilinear', align_corners=True)
        visualize = F.interpolate(s, size=(256, 256), mode='bilinear', align_corners=False)

        return s# ,visualize

if __name__ == '__main__':
    kk = torch.rand([4, 3, 256, 256])
    net = FCN_8s_vgg16(num_classes=6)
    print(net(kk).shape)
    flops, params = profile(net, inputs=(kk, ))
    print(fr'flops:{flops}, params:{params}')

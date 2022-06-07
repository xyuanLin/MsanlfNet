import torch.nn as nn
import math
import torch
from thop import profile
import torch.fft
import torch.nn.functional as F


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=13):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        convss = []
        bns = []
        for i in range(self.nums + 1):
            if i == 0 or stype == 'stage':
                convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
                bns.append(nn.BatchNorm2d(width))
                convss.append(nn.Conv2d(width, width, kernel_size=1, bias=False))
            else:
                convs.append(nn.Conv2d(width*2, width, kernel_size=3, stride=stride, padding=1, bias=False))
                bns.append(nn.BatchNorm2d(width))
                # convss.append(nn.Conv2d(width*2, width, kernel_size=1, bias=False))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.convss= nn.ModuleList(convss)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

        self.se = SEWeightModule(width)
        self.split_channel = width
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        residual = x
        batch_size = x.shape[0]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums + 1):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
                sp = self.convs[i](sp)
                sp = self.relu(self.bns[i](sp))
            else:
                sp = torch.cat([sp, spx[i]], dim=1)
                sp = self.convs[i](sp)
                sp = self.relu(self.bns[i](sp))
                # sp = self.convss[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        spx = torch.split(out, self.width, 1)
        feats = out.view(batch_size, 4, self.width, out.shape[2], out.shape[3])

        x1_se = self.se(spx[0])
        x2_se = self.se(spx[1])
        x3_se = self.se(spx[2])
        x4_se = self.se(spx[3])

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.width, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MsaNet(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(MsaNet, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        stage_0 = x
        x = self.maxpool(x)

        x = self.layer1(x)
        stage_1 = x
        x = self.layer2(x)
        stage_2 = x
        x = self.layer3(x)
        stage_3 = x
        x = self.layer4(x)
        stage_4 = x

        return stage_0, stage_1, stage_2, stage_3, stage_4


def msanet50(pretrained=False, **kwargs):
    model = MsaNet(Bottleneck, [3, 4, 6, 3], baseWidth=16, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict()
    return model


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.permute(0, 3, 1, 2)
        return x


class Fft(nn.Module):
    def __init__(self, dim, h, w, ratio, num_classes):
        super(Fft, self).__init__()
        inter_channels = dim // ratio
        self.Global = GlobalFilter(inter_channels, h, w)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.conv5a = nn.Sequential(nn.Conv2d(dim, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, num_classes, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.Global(feat1) + feat1
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        return sa_output


class MsanlfNet(nn.Module):
    def __init__(self, num_classes):
        super(MsanlfNet, self).__init__()

        self.model = msanet50(pretrained=False)
        ratio = [4, 4, 4, 4]

        self.head_1 = Fft(2048, 8, 5, ratio=ratio[0], num_classes=num_classes)
        self.head_2 = Fft(1024, 16, 9, ratio=ratio[0], num_classes=num_classes)
        self.head_3 = Fft(512, 32, 17, ratio=ratio[0], num_classes=num_classes)
        self.head_4 = Fft(256, 64, 33, ratio=ratio[0], num_classes=num_classes)

    def forward(self, x):
        stage_0, stage_1, stage_2, stage_3, stage_4 = self.model(x)

        stage_4 = self.head_1(stage_4)
        stage_4 = F.interpolate(stage_4, scale_factor=8, mode='bilinear', align_corners=True)

        stage_3 = self.head_2(stage_3)
        stage_3 = F.interpolate(stage_3, scale_factor=4, mode='bilinear', align_corners=True)

        stage_2 = self.head_3(stage_2)
        stage_2 = F.interpolate(stage_2, scale_factor=2, mode='bilinear', align_corners=True)

        stage_1 = self.head_4(stage_1)
        stage_1 = stage_1 + stage_2 + stage_3 + stage_4
        stage_1 = F.interpolate(stage_1, scale_factor=4, mode='bilinear', align_corners=True)

        return stage_1


if __name__ == '__main__':
    images = torch.rand(1, 3, 256, 256)
    model = MsanlfNet(num_classes=16)
    s0 = model(images)
    print(s0.shape)
    param, flops = profile(model, (images,))
    print(fr'Param:{param}')
    print(fr'FLOPs:{flops}')

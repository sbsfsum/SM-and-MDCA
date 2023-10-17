import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class DenseBlock(nn.Module):
    def __init__(self, input_num, num1, num2, rate, drop_out):
        super(DenseBlock, self).__init__()

        # C: 2048 --> 512 --> 256
        self.conv1x1 = nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=num1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dilaconv = nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3, padding=1 * rate, dilation=rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.ConvGN(self.conv1x1(x))
        x = self.relu1(x)
        x = self.dilaconv(x)
        x = self.relu2(x)
        x = self.drop(x)
        return x



class MDC(nn.Module):
    def __init__(self, inplanes=512, planes=512, os=16):
        super(MDC, self).__init__()
        
        if os == 16:
            dilations = [1, 6]
        elif os == 8:
            dilations = [1, 12]

        self.aspp1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, dilation=dilations[0], bias=False),
            nn.BatchNorm2d(planes),
            GELU())

        self.aspp2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1],
                      bias=False),
            nn.BatchNorm2d(planes),
            GELU())


        self.drop_out = 0.1
        self.channels1 = 512
        self.channels2 = 256
        self.num_channels = 512
        self.aspp3 = DenseBlock(self.num_channels, num1=self.channels1, num2=self.channels2, rate=3,
                                drop_out=self.drop_out)
        self.aspp6 = DenseBlock(self.num_channels + self.channels2 * 1, num1=self.channels1, num2=self.channels2,
                                rate=6,
                                drop_out=self.drop_out)
        self.aspp12 = DenseBlock(self.num_channels + self.channels2 * 2, num1=self.channels1, num2=self.channels2,
                                 rate=12,
                                 drop_out=self.drop_out)
        self.aspp18 = DenseBlock(self.num_channels + self.channels2 * 3, num1=self.channels1, num2=self.channels2,
                                 rate=18,
                                 drop_out=self.drop_out)
        self.aspp24 = DenseBlock(self.num_channels + self.channels2 * 4, num1=self.channels1, num2=self.channels2,
                                 rate=24,
                                 drop_out=self.drop_out)
        self.conv1x1 = nn.Conv2d(in_channels=5*self.channels2, out_channels=256, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=256)
        self.conv25 = nn.Conv2d(256, 512, 1, 1)

        # 图像级特征
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(512, 512, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(512),
                                             GELU())
        self.conv1 = nn.Conv2d(2304, 512, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

    def forward(self, x):
        feature = x
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        #x4 = self.aspp4(x)
        aspp3 = self.aspp3(feature)
        feature = torch.cat((aspp3, feature), dim=1)
        aspp6 = self.aspp6(feature)
        feature = torch.cat((aspp6, feature), dim=1)
        aspp12 = self.aspp12(feature)
        feature = torch.cat((aspp12, feature), dim=1)
        aspp18 = self.aspp18(feature)
        #x4 = self.aspp4(feature)
        feature = torch.cat((aspp18, feature), dim=1)
        aspp24 = self.aspp24(feature)
        x11 = torch.cat((aspp3, aspp6, aspp12,aspp18,aspp24), dim=1)
        x4 = self.ConvGN(self.conv1x1(x11))
        x4 = self.conv25(x4)
        x5 = self.global_avg_pool(x)

        # 对x5双线性插值方式进行上采样得到与x1-x5的尺寸，从而方便拼接
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        #print("x.size:", x.size())
        return x



class ShuffleAttention(nn.Module):

    def __init__(self, channel=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out

class MDCA (nn.Module):
    def __init__(self, inp):
        super(MDCA, self).__init__()
        self.TI = ShuffleAttention()
        self.ASPP = MDC()
    def forward(self, x):
        identity = x
        x0 = self.ASPP(x)
        x1 = x*self.TI(x)
        x = x1 + x0
        x = x + identity
        return x




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.CASA = MDCA(512)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # SM
        self.conv = nn.Conv2d(256,512,1,1)
        self.conv11 = nn.Conv2d(64,512,1,1)

        self.co1 = nn.Conv2d(64, 64, 3, 1, padding=3, dilation=3)
        self.co2 = nn.Conv2d(64, 64, 3, 1, padding=5, dilation=5)
        self.co3 = nn.Conv2d(64, 64, 3, 1, padding=7, dilation=7)
        self.co4 = nn.Conv2d(64,64,1,1)
        self.swih = GELU()
        self.cc = nn.Conv2d(2,64,1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feat = []
        feat.append(x)       # C0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feat.append(x)       # C1
        x = self.maxpool(x)

        x = self.layer1(x)

        x11 = torch.mean(x, dim=1, keepdim=True)
        x22, _ = torch.max(x, dim=1, keepdim=True)
        x33 = torch.cat([x11, x22], dim=1)
        x33 = self.cc(x33)

        out2 = self.co1(x)
        out3 = self.co2(x)
        out4 = self.co3(x)
        #out5 = torch.cat([out2,out3,out4],dim=1)
        out5 = out2 + out3 + out4

        out5 = self.co4(out5)
        out5 = self.swih(out5)

        x3 = x33 + out5
        x4 = self.conv11(x3)
        feat.append(x)       # C2
        x = self.layer2(x)
        feat.append(x)       # C3
        x = self.layer3(x)
        feat.append(x)       # C4
        x = self.layer4(x)
        x4 = F.interpolate(x4, x.shape[2:], mode='bilinear', align_corners=False)
        x = x + x4
        x1 = self.CASA(x)
        #x2 = self.DE(x)
        #x2 = self.conv(x2)
        x = x1 #+ x2
        feat.append(x)       # C5


        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        #
        return feat


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)

    summary(model, (3, 640, 800))

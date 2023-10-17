import torch.nn as nn
import numpy as np
import torch
from .model_parts import CombinationModule
from . import resnet

##------------------------------------------#
#  此为对于残差块的定义,其在DilatedEncoder
#  的残差结构中被应用
#  一个残差块共有三个卷积+正则化+relu
#  卷积核的大小分别为 1*1，3*3，1*1
#------------------------------------------#
class Bottleneck(nn.Module):

    def __init__(self,in_channels=512,mid_channels=128,dilation=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential(  #卷积+正则化+relu,其中为1*1的卷积
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(   #卷积+正则化+relu，其中为3*3的卷积
            nn.Conv2d(mid_channels, mid_channels,kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(   #卷积+正则化+relu，其中为1*1的卷积
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + x
        return out

#------------------------------------------#
#  DilatedEncoder结构
#------------------------------------------#
class DilatedEncoder(nn.Module):
    def __init__(self, in_channels=2048, encoder_channels=512, block_mid_channels=128, num_residual_blocks=4, block_dilations=[2, 4, 6, 8]):
        super(DilatedEncoder, self).__init__()

        self.in_channels         = in_channels
        self.encoder_channels    = encoder_channels
        self.block_mid_channels  = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations     = block_dilations
        #这个断言的作用是根据残差块的个数来确定block_dilations的长度
        #每个残差块中所对应的膨胀卷积率
        assert len(self.block_dilations) == self.num_residual_blocks
        # 初始化两个函数
        self._init_layers()
        self._init_weight()

#--------------------------------------#
#  此为对于投影层以及残差结构的初始化操作
#  残差结构中共有四个残差块
#--------------------------------------#
    def _init_layers(self):
        #-----------------------------#
        # 首先为先进行投影层的初始化操作
        #-----------------------------#
        #首先为1*1的卷积+正则化
        self.lateral_conv = nn.Conv2d(self.in_channels,self.encoder_channels,kernel_size=1)
        self.lateral_norm = nn.BatchNorm2d(self.encoder_channels)
        #接下来为3*3的卷积加正则化
        self.fpn_conv = nn.Conv2d(self.encoder_channels,self.encoder_channels,kernel_size=3,padding=1)
        self.fpn_norm = nn.BatchNorm2d(self.encoder_channels)
        #-----------------------------#
        # 接下来进行残差结构的初始化操作
        #-----------------------------#
        encoder_blocks = []
        #在这个结构中，共有四个残差块，每个残差块均由三个conv+bn+relu组成
        for i in range(self.num_residual_blocks):
            # 分配每一个残差块中的卷积所需要的膨胀率，4个残差块从左至右依次为 2，4，6，8
            dilation = self.block_dilations[i]
            # 对每个残差块进行初始化
            encoder_blocks.append(
                Bottleneck(
                    self.encoder_channels,
                    self.block_mid_channels,
                    dilation=dilation
                )
            )
        #将四个残差块依次进行排列，进而得到了我们的残差结构
        #print(encoder_blocks[0])
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)  #pack up

    def xavier_init(self, layer):
        if isinstance(layer, nn.Conv2d):  #对于layer中的二维卷积操作进行相应的初始化
            #---------------------------------------------#
            # xavier的基本思想是通过网络层时，输入和输出的方差相同，包括前向传播和反向传播
            # 用于激活函数中， 此处code的含义为服从均匀分布
            #---------------------------------------------#
            nn.init.xavier_uniform_(layer.weight, gain=1)

#-------------------------------------------#
# 初始化Dilated Encoder的权重值
#-------------------------------------------#
    def _init_weight(self):
        #为投影层中的卷积操作添加xavier
        self.xavier_init(self.lateral_conv)
        self.xavier_init(self.fpn_conv)
        #对投影层中的正则化操作进行相应的限制
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1) #使用1来填充weight的值
            nn.init.constant_(m.bias, 0) #使用0来填充bias的值，表示偏移量为0
        # 对于残差结构中的四个残差块
        for m in self.dilated_encoder_blocks.modules():
            # 对于conv操作
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)  #使其服从一个正太分布操作
                if hasattr(m, 'bias') and m.bias is not None:#如果这个卷积操作存在偏置项bias且不为空，则我们将其值初始为0
                    nn.init.constant_(m.bias, 0)
            # 对于正则化操作
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)   #将其weight值初始化为常数
                nn.init.constant_(m.bias, 0)     #将其bias值初始化为常数

# -------------------------------------------#
# 进行DilatedEncoder模块的前向传播函数的定义
# -------------------------------------------#
    def forward(self, feature): #可以理解为对于一个张量的操作
        out = self.lateral_norm(self.lateral_conv(feature))  #为先过卷积再过池化，投影层的第一层卷积
        out = self.fpn_norm(self.fpn_conv(out))  #先过卷积再过池化，投影层的第二层卷积
        out = self.dilated_encoder_blocks(out)   #最后经过残差结构，返回结果
        return out


class CTRBOX(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(CTRBOX, self).__init__()
        channels = [3, 64, 64, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        #self.base_network = resnet.resnet101(pretrained=pretrained)
        #self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        #self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        #self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        #101-18
        self.base_network = resnet.resnet18(pretrained=pretrained)
        #self.base_network = ConvNeXt_Tiny(pretrained=pretrained)
        self.dec_c2 = CombinationModule(128, 64, batch_norm=True)
        self.dec_c3 = CombinationModule(256, 128, batch_norm=True)
        self.dec_c4 = CombinationModule(512, 256, batch_norm=True)
        self.heads = heads
        #self.CASA1 = CASA(256,256)
        self.DilatedEncoder = DilatedEncoder(64, 64)



        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   #SPConv_3x3(channels[self.l1], head_conv),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True)
                                    )
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                   #SPConv_3x3(channels[self.l1], head_conv),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
                                  )
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])
        c2_combine = self.DilatedEncoder(c2_combine)
        #c4_combine = self.CASA1(c4_combine)
        #c4_combine1 = self.ConvTrans1(c4_combine)
        #c4_combine2 = self.ConvTrans2(c4_combine1)
        #c3_combine = self.dec_c3(c4_combine, x[-3])# + c4_combine1
        #c3_combine = self.CASA2(c3_combine)
        #c3_combine1 = self.ConvTrans3(c3_combine)
        #c2_combine = self.dec_c2(c3_combine, x[-4])# + c4_combine2 + c3_combine1








        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict

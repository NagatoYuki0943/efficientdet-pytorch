import torch
import torch.nn as nn
from utils.anchors import Anchors

from nets.efficientnet import EfficientNet as EffNet
# 起了别名
from nets.layers import (Conv2dStaticSamePadding as Conv2D, MaxPool2dStaticSamePadding as MaxPool2d, MemoryEfficientSwish, Swish)
"""Conv2dStaticSamePadding 中只有卷积,没哟bn,激活函数"""

#----------------------------------#
#   Xception中深度可分离卷积
#   先3x3的深度可分离卷积
#   再1x1的普通卷积
#   通道数和宽高不变
#----------------------------------#
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2D(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2D(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        # 两次使用没有使用bn,激活函数,最后才使用,不知道效果会有什么影响
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

"""加强特征提取"""
class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # DW+PW卷积, 通道数和宽高不变
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        # DW+PW卷积, 通道数和宽高不变
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # 上采样 宽高x2
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # 下采样 宽高减半 kernel=3. stride=2
        self.p4_downsample = MaxPool2d(3, 2)
        self.p5_downsample = MaxPool2d(3, 2)
        self.p6_downsample = MaxPool2d(3, 2)
        self.p7_downsample = MaxPool2d(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            """net层p3,p4,p5 都没有激活函数!!!"""
            # C3 64, 64, 40 -> 64, 64, 64
            self.p3_down_channel = nn.Sequential(
                Conv2D(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            # C4 32, 32, 112 -> 32, 32, 64
            self.p4_down_channel = nn.Sequential(
                Conv2D(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            # C5 16, 16, 320 -> 16, 16, 64
            self.p5_down_channel = nn.Sequential(
                Conv2D(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            # BIFPN第一轮的时候，跳线那里并不是同一个in,这个输出 _2, 上面输出 _1
            # C4 32, 32, 112 -> 32, 32, 64
            self.p4_down_channel_2 = nn.Sequential(
                Conv2D(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            # C5 16, 16, 320 -> 16, 16, 64
            self.p5_down_channel_2 = nn.Sequential(
                Conv2D(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            """下面两次是添加的数据获取,net只能获取p3,p4,p5,下面获取p5p6"""
            # 对输入进来的p5进行宽高的下采样
            # C5 16, 16, 320 -> 8, 8, 64
            self.p5_to_p6 = nn.Sequential(
                Conv2D(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2d(3, 2)    # kernel_size=3 stride=2
            )
            # P6_in 8, 8, 64 -> 4, 4, 64
            self.p6_to_p7 = nn.Sequential(
                MaxPool2d(3, 2)    # kernel_size=3 stride=2
            )

        # 简易注意力机制的weights
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """ bifpn模块结构示意图
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        #------------------------------------------------#
        #   当phi=1、2、3、4、5的时候使用fast_attention
        #   获得三个shape的有效特征层
        #   分别是C3   64, 64, 40
        #         C4  32, 32, 112
        #         C5  16, 16, 320
        #------------------------------------------------#
        if self.first_time:
            #------------------------------------------------------------------------#
            #   第一次BIFPN需要 下采样 与 调整通道 获得 p3_in p4_in p5_in p6_in p7_in
            #------------------------------------------------------------------------#
            p3, p4, p5 = inputs
            #-------------------------------------------#
            #   首先对通道数进行调整 1x1Conv+BN
            #   C3 64, 64, 40 -> 64, 64, 64
            #-------------------------------------------#
            p3_in = self.p3_down_channel(p3)

            #-------------------------------------------#
            #   首先对通道数进行调整 1x1Conv+BN
            #   C4 32, 32, 112 -> 32, 32, 64 -> 32, 32, 64
            #-------------------------------------------#
            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)

            #-------------------------------------------#
            #   首先对通道数进行调整 1x1Conv+BN
            #   C5 16, 16, 320 -> 16, 16, 64 -> 16, 16, 64
            #-------------------------------------------#
            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)

            """下面两次是添加的数据获取,net只能获取p3,p4,p5,下面获取p5p6"""
            #-------------------------------------------#
            #   对C5(Net最后一层)进行1x1Conv,BN和MaxPool下采样，调整通道数与宽高
            #   C5 16, 16, 320 -> 8, 8, 64
            #-------------------------------------------#
            p6_in = self.p5_to_p6(p5)
            #-------------------------------------------#
            #   对P6_in进行MaxPool下采样，调整宽高
            #   P6_in 8, 8, 64 -> 4, 4, 64
            #-------------------------------------------#
            p7_in = self.p6_to_p7(p6_in)


            """下面四次上采样特征融合,图中由上到下"""
            # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            p6_w1  = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            # p6_in * 权重 + p7_in上采样 * 权重, 最后DW+PW卷积,获得p6_td
            p6_td  = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in        P5_in分为 1 和 2
            p5_w1  = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            # p5_in_1 * 权重 + p6_td上采样 * 权重, 最后DW+PW卷积,获得p5_td
            p5_td  = self.conv5_up(self.swish(weight[0] * p5_in_1 + weight[1] * self.p5_upsample(p6_td)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in        P4_in分为 1 和 2
            p4_w1  = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            # p4_in_1 * 权重 + p5_td上采样 * 权重, 最后DW+PW卷积,获得p4_td
            p4_td  = self.conv4_up(self.swish(weight[0] * p4_in_1 + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1  = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            # p3_in * 权重 + p4_td上采样 * 权重, 最后DW+PW卷积,获得p3_out
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))


            """下面四次下采样特征融合,图中由下到上"""
            # 简单的注意力机制，用于确定更关注p4_in_2还是p4_up还是p3_out        P4_inn分为 1 和 2
            p4_w2  = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            # p4_in_2 * 权重 + p4_td * 权重 + p3_out下采样 * 权重, 最后DW+PW卷积,获得p4_out
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in_2 + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in_2还是p5_up还是p4_out        P5_in分为 1 和 2
            p5_w2  = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            # p5_in_2 * 权重 + p5_td * 权重 + p4_out下采样 * 权重, 最后DW+PW卷积,获得p5_out
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in_2 + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2  = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            # p6_in * 权重 + p6_td * 权重 + p5_out下采样 * 权重, 最后DW+PW卷积,获得p6_out
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out)))

            # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            p7_w2  = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            # p7_in * 权重 + p6_out下采样 * 权重, 最后DW+PW卷积,获得p7_out
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            """下面四次上采样特征融合,图中由上到下"""
            # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            p6_w1  = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            # p6_in * 权重 + p7_in上采样 * 权重, 最后DW+PW卷积,获得p6_td
            p6_td  = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p5_w1  = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            # p5_in* 权重 + p6_td上采样 * 权重, 最后DW+PW卷积,获得p5_td
            p5_td  = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_td)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p4_w1  = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            # p4_in * 权重 + p5_td上采样 * 权重, 最后DW+PW卷积,获得p4_td
            p4_td  = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1  = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            # p3_in * 权重 + p4_td上采样 * 权重, 最后DW+PW卷积,获得p3_out
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))


            """下面四次下采样特征融合,图中由下到上"""
            # 简单的注意力机制，用于确定更关注p4_in还是p4_up还是p3_out
            p4_w2  = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            # p4_in * 权重 + p4_td * 权重 + p3_out下采样 * 权重, 最后DW+PW卷积,获得p4_out
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_td+ weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in还是p5_up还是p4_out
            p5_w2  = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            # p5_in * 权重 + p5_td * 权重 + p4_out下采样 * 权重, 最后DW+PW卷积,获得p5_out
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in + weight[1] * p5_td+ weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2  = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            # p6_in * 权重 + p6_td * 权重 + p5_out下采样 * 权重, 最后DW+PW卷积,获得p6_out
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td+ weight[2] * self.p6_downsample(p5_out)))

            # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            p7_w2  = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            # p7_in * 权重 + p6_out下采样 * 权重, 最后DW+PW卷积,获得p7_out
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        # 当phi=6、7的时候使用_forward
        if self.first_time:
            # 第一次BIFPN需要下采样与降通道获得
            # p3_in p4_in p5_in p6_in p7_in
            p3, p4, p5 = inputs
            p3_in = self.p3_down_channel(p3)
            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)
            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p6_td= self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            p5_td= self.conv5_up(self.swish(p5_in_1 + self.p5_upsample(p6_td)))

            p4_td= self.conv4_up(self.swish(p4_in_1 + self.p4_upsample(p5_td)))

            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in_2 + p4_td+ self.p4_downsample(p3_out)))

            p5_out = self.conv5_down(
                self.swish(p5_in_2 + p5_td+ self.p5_downsample(p4_out)))

            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td+ self.p6_downsample(p5_out)))

            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            p6_td= self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            p5_td= self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_td)))

            p4_td= self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_td)))

            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in + p4_td+ self.p4_downsample(p3_out)))

            p5_out = self.conv5_down(
                self.swish(p5_in + p5_td+ self.p5_downsample(p4_out)))

            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td+ self.p6_downsample(p5_out)))

            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


"""
先验框预测
对上面BiFPN获得的5个有效特征层进行运算
5个有效特征层都使用同一个 BoxNet
5个有效特征层的卷积都是 conv_list,不过bn是不同的, bn_list 循环了5次
"""
class BoxNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(BoxNet, self).__init__()
        self.num_layers = num_layers
        # 3层深度可分离卷积特征融合 in_channels=64
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        # 每一个有效特征层对应的Batchnor,5个特征层有5个bn列表
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in range(5)])
        
        # 9个先验框 4个参数 中心，宽高
        # 调整通道数,获得最后的分类 num_anchors = 9 * 4 = 36
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        """
        inputs: 5个特征层
        """
        feats = []
        # 对每个特征层循环(5个特征层的bn不同)
        for feat, bn_list in zip(inputs, self.bn_list):
            # 每个特征层需要进行num_layer次卷积+标准化+激活函数 5个特征层的卷积是相同的
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            # [b, num_anchors*4, h, w]
            feat = self.header(feat)

            # [b, num_anchors*4, h, w] -> [b, h, w, num_anchors*4] 通道放到最后
            feat = feat.permute(0, 2, 3, 1)
            #                              b, 全部先验框, 先验框的4个参数    h*w表示网格总数,再乘以num_anchors就是全部先验框
            # [b, num_anchors*4, h, w] -> [b, h*w*num_anchors, 4]
            feat = feat.contiguous().view(feat.shape[0], -1, 4)
            feats.append(feat)
        # 维度上拼接
        feats = torch.cat(feats, dim=1)

        return feats


"""
类别预测
对上面BiFPN获得的5个有效特征层进行运算
5个有效特征层都使用同一个 ClassNet
5个有效特征层的卷积都是 conv_list,不过bn是不同的, bn_list 循环了5次
"""
class ClassNet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(ClassNet, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        # 3深度可分离卷积特征融合 in_channels=64
        self.conv_list  = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        # 每一个有效特征层对应的Batchnor,5个特征层有5个bn列表
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in range(5)])
        
        # 9 个先验框 num_classes
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        # 对每个特征层循环(5个特征层的bn不同)
        for feat, bn_list in zip(inputs, self.bn_list):
            # 每个特征层需要进行num_layer次卷积+标准化+激活函数 5个特征层的卷积是相同的
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            # [b, 9*num_classes, h, w]
            feat = self.header(feat)

            # [b, 9*num_classes, h, w]  => [b, h, w, 9*num_classes]
            feat = feat.permute(0, 2, 3, 1)
            # [b, h, w, 9*num_classes]  => [b, h, w, 9, num_classes]
            # feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors, self.num_classes)
            #                              [b, 全部先验框, 分类数]     h*w表示网格总数,再乘以num_anchors就是全部先验框
            # [b, h, w, 9, num_classes] => [b, h*w*9, num_classes]
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)
        # 维度上拼接
        feats = torch.cat(feats, dim=1)
        # 取sigmoid表示概率
        feats = feats.sigmoid()

        return feats

#------------------------------------------------------#
#   获得原始EfficientNet
#------------------------------------------------------#
class EfficientNet(nn.Module):
    def __init__(self, phi, pretrained=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}', pretrained)
        # 删除不要的部分
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            #------------------------------------------------------#
            #   取出对应的特征层，如果某个EffcientBlock的步长为2的话
            #   意味着它的前一个特征层为有效特征层
            #   除此之外，最后一个EffcientBlock的输出为有效特征层
            #------------------------------------------------------#
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]

"""预测模型"""
class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes = 80, phi = 0, pretrained = False):
        super(EfficientDetBackbone, self).__init__()
        #--------------------------------#
        #   phi指的是efficientdet的版本
        #--------------------------------#
        self.phi = phi
        #---------------------------------------------------#
        #   backbone_phi指的是该efficientdet对应的efficient
        #---------------------------------------------------#
        self.backbone_phi = [0, 1, 2, 3, 4, 5, 6, 6]
        #--------------------------------#
        #   BiFPN所用的通道数
        #--------------------------------#
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        #--------------------------------#
        #   BiFPN的重复次数
        #--------------------------------#
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        #---------------------------------------------------#
        #   Effcient Head卷积重复次数
        #---------------------------------------------------#
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        #---------------------------------------------------#
        #   基础的先验框大小
        #---------------------------------------------------#
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        num_anchors = 9 # 每个点对应9个先验框

        conv_channel_coef = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        #------------------------------------------------------#
        #   在经过多次BiFPN模块的堆叠后，我们获得的fpn_features
        #   假设我们使用的是efficientdet-D0包括五个有效特征层：
        #   P3_out      64,64,64
        #   P4_out      32,32,64
        #   P5_out      16,16,64
        #   P6_out      8,8,64
        #   P7_out      4,4,64
        #------------------------------------------------------#
        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.phi],
                    conv_channel_coef[phi],
                    True if _ == 0 else False,  # 第一次BiFPN为True
                    attention=True if phi < 6 else False)
              for _ in range(self.fpn_cell_repeats[phi])])

        self.num_classes = num_classes
        #------------------------------------------------------#
        #   创建efficient head
        #   可以将特征层转换成预测结果
        #------------------------------------------------------#
        # 先验框预测
        self.regressor      = BoxNet(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
                                    num_layers=self.box_class_repeats[self.phi])
        # 分类预测
        self.classifier     = ClassNet(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
                                    num_classes=num_classes, num_layers=self.box_class_repeats[self.phi])
        # 生成先验框
        self.anchors        = Anchors(anchor_scale=self.anchor_scale[phi])

        #-------------------------------------------#
        #   获得三个shape的有效特征层
        #   分别是C3  64, 64, 40
        #         C4  32, 32, 112
        #         C5  16, 16, 320
        #   bifpn负责C6, C7的创建
        #-------------------------------------------#
        self.backbone_net   = EfficientNet(self.backbone_phi[phi], pretrained)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        # bifpn负责p6, p7的创建
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs)

        # features:       (p3, p4, p5)
        # regression:     BoxNet先验框调整 [1, h*w*num_anchors, 4]
        # classification: ClassNet分类预测 [1, h*w*num_anchors, 90]
        # anchors:        先验框           [1, h*w*num_anchors, 4]
        return features, regression, classification, anchors


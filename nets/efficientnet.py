"""
SiLU == Swish
"""

import torch
from torch import nn
from torch.nn import functional as F

from nets.layers import (MemoryEfficientSwish, Swish, drop_connect,
                         efficientnet_params, get_model_params,
                         get_same_padding_conv2d, load_pretrained_weights,
                         round_filters, round_repeats)


class MBConvBlock(nn.Module):
    '''
    EfficientNet-b0:
    [BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16, expand_ratio=1, id_skip=True, stride=[1], se_ratio=0.25),   256,256,32 -> 256,256,16
     BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),   256,256,16 -> 128,128,24
     BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),   128,128,24 -> 64,64,40  P3 宽高减半
     BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),   64,64,40 -> 32,32,80
     BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25),  32,32,80 -> 32,32,112   P4 宽高不变
     BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25), 32,32,112 -> 16,16,192
     BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25)] 16,16,192 -> 16,16,320  p5 宽高不变

    P3:64,64,40   P4:32,32,112   p5:16,16,320

    GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=0.001, dropout_rate=0.2, num_classes=1000, width_coefficient=1.0,
                    depth_coefficient=1.0, depth_divisor=8, min_depth=None, drop_connect_rate=0.2, image_size=224)
    '''
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        # 获得一种卷积方法
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # 获得标准化的参数
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon

        #----------------------------#
        #   计算是否施加注意力机制
        #----------------------------#
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        #----------------------------#
        #   判断是否添加残差边
        #----------------------------#
        self.id_skip = block_args.id_skip

        #-------------------------------------------------#
        #   利用Inverted residuals
        #   part1 利用1x1卷积进行通道数上升
        #-------------------------------------------------#
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        # 扩张维度不为1才使用第一层卷积
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        #------------------------------------------------------#
        #   如果步长为2x2的话，利用深度可分离卷积进行高宽压缩
        #   part2 利用3x3卷积对每一个channel进行卷积
        #------------------------------------------------------#
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        #------------------------------------------------------#
        #   完成深度可分离卷积后
        #   对深度可分离卷积的结果施加注意力机制
        #   in_channel -> 初始channel/4 -> in_channel
        #------------------------------------------------------#
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            #------------------------------------------------------#
            #   通道先压缩后上升，最后利用sigmoid将值固定到0-1之间
            #------------------------------------------------------#
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        #------------------------------------------------------#
        #   part3 利用1x1卷积进行通道下降
        #------------------------------------------------------#
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

        # 前两层和se中第一层的激活函数 silu,别名swish
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        #-------------------------------------------------#
        #   利用Inverted residuals
        #   part1 利用1x1卷积进行通道数上升
        #-------------------------------------------------#
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))

        #------------------------------------------------------#
        #   如果步长为2x2的话，利用深度可分离卷积进行高宽压缩
        #   part2 利用3x3卷积对每一个channel进行卷积
        #------------------------------------------------------#
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        #------------------------------------------------------#
        #   完成深度可分离卷积后
        #   对深度可分离卷积的结果施加注意力机制
        #------------------------------------------------------#
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(
                self._swish(self._se_reduce(x_squeezed)))
            # sigmoid将权重变换到0~1之间
            x = torch.sigmoid(x_squeezed) * x

        #------------------------------------------------------#
        #   part3 利用1x1卷积进行通道下降
        #------------------------------------------------------#
        x = self._bn2(self._project_conv(x))

        #------------------------------------------------------#
        #   part4 如果满足残差条件，那么就增加残差边
        #------------------------------------------------------#
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        # 如果通道且宽高不变
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class EfficientNet(nn.Module):
    '''
    EfficientNet-b0:
    [BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16, expand_ratio=1, id_skip=True, stride=[1], se_ratio=0.25),   256,256,32 -> 256,256,16
     BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),   256,256,16 -> 128,128,24
     BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),   128,128,24 -> 64,64,40  P3 宽高减半
     BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25),   64,64,40 -> 32,32,80
     BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25),  32,32,80 -> 32,32,112   P4 宽高不变
     BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192, expand_ratio=6, id_skip=True, stride=[2], se_ratio=0.25), 32,32,112 -> 16,16,192
     BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320, expand_ratio=6, id_skip=True, stride=[1], se_ratio=0.25)] 16,16,192 -> 16,16,320  p5 宽高不变

    P3:64,64,40   P4:32,32,112   p5:16,16,320

     GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=0.001, dropout_rate=0.2, num_classes=1000, width_coefficient=1.0,
                    depth_coefficient=1.0, depth_divisor=8, min_depth=None, drop_connect_rate=0.2, image_size=224)
    '''
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        # 获得一种卷积方法
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # 获得标准化的参数
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        #-------------------------------------------------#
        #   网络主干部分开始
        #   设定输入进来的是RGB三通道图像
        #   利用round_filters可以使得通道可以被8整除
        #-------------------------------------------------#
        in_channels = 3
        out_channels = round_filters(32, self._global_params)

        #-------------------------------------------------#
        #   创建stem部分 宽高减半
        #   512,512,3 -> 256,256,32
        #-------------------------------------------------#
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d( num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        #-------------------------------------------------#
        #   在这个地方对大结构块进行循环
        #-------------------------------------------------#
        self._blocks = nn.ModuleList([])
        for i in range(len(self._blocks_args)):
            #-------------------------------------------------------------#
            #   对每个block的参数进行修改，根据所选的efficient版本进行修改
            #-------------------------------------------------------------#
            self._blocks_args[i] = self._blocks_args[i]._replace(
                input_filters=round_filters(self._blocks_args[i].input_filters, self._global_params),
                output_filters=round_filters(self._blocks_args[i].output_filters, self._global_params),
                num_repeat=round_repeats(self._blocks_args[i].num_repeat, self._global_params)
            )

            #-------------------------------------------------------------#
            #   每个大结构块里面的第一个EfficientBlock
            #   都需要考虑步长和输入通道数
            #-------------------------------------------------------------#
            self._blocks.append(MBConvBlock(self._blocks_args[i], self._global_params))

            if self._blocks_args[i].num_repeat > 1:
                self._blocks_args[i] = self._blocks_args[i]._replace(input_filters=self._blocks_args[i].output_filters, stride=1)

            #---------------------------------------------------------------#
            #   在利用第一个EfficientBlock进行通道数的调整或者高和宽的压缩后
            #   进行EfficientBlock的堆叠
            #---------------------------------------------------------------#
            for _ in range(self._blocks_args[i].num_repeat - 1):
                self._blocks.append(MBConvBlock(self._blocks_args[i], self._global_params))

        #----------------------------------------------------------------#
        #   这是efficientnet的尾部部分，在进行effcientdet构建的时候没用到
        #   只在利用efficientnet进行分类的时候用到。
        #----------------------------------------------------------------#
        in_channels = self._blocks_args[len(self._blocks_args)-1].output_filters
        out_channels = round_filters(1280, self._global_params)

        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        # swish函数
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, load_weights=True, advprop=True, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        if load_weights:
            load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

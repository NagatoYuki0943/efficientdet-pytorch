"""
生成先验框 y1,x1,y2,x2
9个
三个正方形,三个竖的长方形,三个横的长方形
"""

import itertools

import numpy as np
import torch
import torch.nn as nn

# 生成先验框 y1,x1,y2,x2
class Anchors(nn.Module):
    def __init__(self, anchor_scale=4., pyramid_levels=[3, 4, 5, 6, 7]):
        super().__init__()
        self.anchor_scale = anchor_scale
        self.pyramid_levels = pyramid_levels
        # stride = 原宽 / 特征宽
        # 512 / 64 = 8 以此类推
        # strides步长为[8, 16, 32, 64, 128]， 特征点的间距
        self.strides = [2 ** x for x in self.pyramid_levels]
        # scales,ratios长度都是3
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        self.ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    def forward(self, image):
        image_shape = image.shape[2:]

        boxes_all = []
        # 循环步长,相当于对5个特征层进行循环
        for stride in self.strides:
            boxes_level = []
            # 组合scales和ratios进行循环
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                # 计算先验框基础的边长
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                # 计算每个网格点的xy左边
                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                # 组合坐标
                xv, yv = np.meshgrid(x, y)
                # 平铺
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # 先验框定义是左上角坐标和右下角坐标 y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        # 所有先验框进行堆叠
        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        return anchor_boxes

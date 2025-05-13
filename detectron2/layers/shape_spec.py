# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from collections import namedtuple


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.

    Attributes:
        channels:通道数
        height:高度
        width:宽度
        stride:步长, 相对于输入图像下采样的比率
    """
    # 继承自namedtuple，通过指定字段访问元素
    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)

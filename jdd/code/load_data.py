#!/usr/bin/env python
# encoding: utf-8

from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np
from util import utils
from setting import *

def transform_train(data, label):
    im = image.imresize(data.astype('float32') / 255, 224, 224)
    auglist = image.CreateAugmenter(data_shape=(3, 224, 224), resize=0,
                        rand_crop=True, rand_resize=True, rand_mirror=True,
                        mean=None, std=None,
                        brightness=0, contrast=0,
                        saturation=0, hue=0,
                        pca_noise=0, rand_gray=0, inter_method=2)
    for aug in auglist:
        im = aug(im)
    # 将数据格式从"高*宽*通道"改为"通道*高*宽"。
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

# 测试时，无需对图像做标准化以外的增强数据处理。
def transform_test(data, label):
    im = image.imresize(data.astype('float32') / 255, 224, 224)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

def load_data(path, transform_func, shuffle):
    ds = vision.ImageFolderDataset(path, flag=1, transform=transform_func)
    loader = gluon.data.DataLoader
    data = loader(ds, batch_size, shuffle=shuffle, last_batch='keep')
    return data


#!/usr/bin/env python
# encoding: utf-8

from model import SimpleCnn, ResNet, DenseNet
import mxnet as mx
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
from mxnet import init
from mxnet import nd
from setting import *
from load_data import load_data, transform_train, transform_test
from util import utils

if __name__ == "__main__":
    ctx = utils.try_gpu()
    net = models.resnet50_v2(classes=num_outputs, ctx=ctx)
    net.hybridize()

    net.load_params(model_path, ctx)
    test_data = load_data(data_dir + 'test_A', transform_test, False)
    files = []
    for i in test_data._dataset.items:
        files.append(i[0].split("/")[-1].split(".")[0])
    preds = []
    for data, label in test_data:
        output = nd.softmax(net(data.as_in_context(ctx)))
        preds.extend(output)
    assert len(files) == len(preds)

    train_ds = vision.ImageFolderDataset(data_dir + 'train',
                                           flag=1, transform=transform_train)
    fwp = open('submission.csv', 'w')
    data_labels = zip(files, preds)
    for data_label in data_labels:
        for i in range(0, 30):
            fwp.write(data_label[0] + ',' + train_ds.synsets[i] + ',' + ('%.10f' % data_label[1][i].asnumpy()) + '\n')
    fwp.close()


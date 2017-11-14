from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np
from util import utils
from model import get_net, train
from setting import *
from load_data import load_data, transform_train, transform_test


def load_data():
# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
    train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1,
                                        transform=transform_train)
    valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1,
                                        transform=transform_test)
    test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1,
                                        transform=transform_test)

    loader = gluon.data.DataLoader
    train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
    valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
    test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')
    return train_data, valid_data, test_data

if __name__ == '__main__':
    ctx = utils.try_gpu()
    train_data, valid_data, test_data = load_data()
    print("load data train_size: %s, valid_size: %s, test_size: %s" % (len(train_data), len(valid_data), len(test_data)))
    net = get_net(ctx)
    net.hybridize()
    train_data = load_data(input_str + 'train', transform_train, True)
    valid_data = load_data(input_str + 'vaild', transform_test, True)
    train(net, train_data, valid_data, num_epochs, learning_rate,
          weight_decay, ctx, lr_period, lr_decay)
    net.save_params(model_path)



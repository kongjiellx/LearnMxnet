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

if __name__ == '__main__':
    ctx = utils.try_gpu()
    net = get_net(ctx)
    net.hybridize()
    train_data = load_data(input_str + 'train', transform_train, True)
    valid_data = load_data(input_str + 'valid', transform_test, True)
    train(net, train_data, valid_data, num_epochs, learning_rate,
          weight_decay, ctx, lr_period, lr_decay)
    net.save_params(model_path)



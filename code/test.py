#!/usr/bin/env python
# encoding: utf-8

from model import get_net
import mxnet as mx
from setting import *
from load_data import load_data, transform_test
from util import utils

if __name__ == "__main__":
    ctx = utils.try_gpu()
    net = get_net(ctx)
    net.load_params(model_path, ctx)
    test_data = load_data(input_str + 'test', transform_test, False)
    for data, label in test_data:
        print(net(data.as_in_context(ctx)))


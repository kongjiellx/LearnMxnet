#!/usr/bin/env python
# encoding: utf-8

from model import SimpleCnn, ResNet
import mxnet as mx
from mxnet.gluon.data import vision
from setting import *
from load_data import load_data, transform_train, transform_test
from util import utils
import pandas as pd

if __name__ == "__main__":
    ctx = utils.try_gpu()
    net = SimpleCnn(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier())
    net.hybridize()

    net.load_params(model_path, ctx)
    test_data = load_data(input_str + 'test', transform_test, False)
    preds = []
    for data, label in test_data:
        output = net(data.as_in_context(ctx))
        preds.extend(output.argmax(axis=1).astype(int).asnumpy())
        break

    sorted_ids = list(range(1, 300000 + 1))
    sorted_ids.sort(key = lambda x:str(x))

    train_ds = vision.ImageFolderDataset(input_str + 'train',
                                           flag=1, transform=transform_train)
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_ds.synsets[x])
    df.to_csv('submission.csv', index=False)

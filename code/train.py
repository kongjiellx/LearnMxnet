from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np
from util import utils
from model import ResNet, SimpleCnn
from setting import *
import datetime
from load_data import load_data, transform_train, transform_test

def train(net, train_data, valid_data, num_epochs, lr, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(
        net.collect_params(), 'adam', {'learning_rate': lr})

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    prev_time = datetime.datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        num = 0
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
            num += 1
            if num % 1000 == 0:
                print(num)

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data), valid_acc))
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
        net.save_params(model_path)

if __name__ == '__main__':
    train_data = load_data(input_str + 'train', transform_train, True)
    valid_data = load_data(input_str + 'valid', transform_test, True)

    ctx = utils.try_gpu()
    net = SimpleCnn(num_outputs)
    net.initialize(ctx=ctx, init=init.Xavier())
    net.hybridize()
    train(net, train_data, valid_data, num_epochs, learning_rate,
          ctx, lr_period, lr_decay)
    net.save_params(model_path)



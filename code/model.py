from mxnet.gluon import nn
from mxnet import autograd
from mxnet import nd
from mxnet import gluon
from util import utils
from setting import *


class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                      strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)


class ResNet(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # block 1
            net.add(nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            # block 2
            for _ in range(3):
                net.add(Residual(channels=32))
            # block 3
            net.add(Residual(channels=64, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=64))
            # block 4
            net.add(Residual(channels=128, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=128))
            # block 5
            net.add(nn.AvgPool2D(pool_size=8))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out


class DenseBlock(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=1, use_bias=False)
            self.bn1 = nn.BatchNorm()

    def hybrid_forward(self, F, x):
        out = self.conv1(F.relu(self.bn1(x)))
        return F.concat(out, x)


class TransitionBlock(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=1, use_bias=False)
            self.bn1 = nn.BatchNorm()
            self.pool1 = nn.AvgPool2D(pool_size=2, strides=(2, 2))

    def hybrid_forward(self, F, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.pool1(out)
        return out


class DenseNet(nn.HybridBlock):
    def __init__(self, num_classes, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            net.add(nn.Conv2D(channels=16, kernel_size=3, strides=1, padding=1, use_bias=False))
            num_channles = 16
            for _ in range(12):
                net.add(DenseBlock(channels=12))
                num_channles += 12
            net.add(TransitionBlock(channels=num_channles))
            for _ in range(12):
                net.add(DenseBlock(channels=12))
                num_channles += 12
            net.add(TransitionBlock(channels=num_channles))
            for _ in range(12):
                net.add(DenseBlock(channels=12))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            net.add(nn.AvgPool2D(pool_size=8))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))


    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            print('Block %d output: %s'%(i+1, out.shape))
        return out


class SimpleCnn(nn.HybridBlock):
    def __init__(self, num_classes, **kwargs):
        super(SimpleCnn, self).__init__(**kwargs)
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            for _ in range(3):
                net.add(nn.Conv2D(channels=32, kernel_size=1, strides=1))
                net.add(nn.BatchNorm())
                net.add(nn.Activation(activation='relu'))
            net.add(nn.Conv2D(channels=64, kernel_size=2, strides=1))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            for _ in range(2):
                net.add(nn.Conv2D(channels=64, kernel_size=1, strides=1))
                net.add(nn.BatchNorm())
                net.add(nn.Activation(activation='relu'))
            net.add(nn.Conv2D(channels=128, kernel_size=2, strides=1))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            for _ in range(2):
                net.add(nn.Conv2D(channels=128, kernel_size=1, strides=1))
                net.add(nn.BatchNorm())
                net.add(nn.Activation(activation='relu'))
            net.add(nn.AvgPool2D(pool_size=8))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            print('Block %d output: %s'%(i+1, out.shape))
        return out

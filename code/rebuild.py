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

def transform_train(data, label):
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0,
                        rand_crop=False, rand_resize=False, rand_mirror=True,
                        mean=np.array([0.4914, 0.4822, 0.4465]),
                        std=np.array([0.2023, 0.1994, 0.2010]),
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
    im = data.astype('float32') / 255
    auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                        mean=np.array([0.4914, 0.4822, 0.4465]),
                        std=np.array([0.2023, 0.1994, 0.2010]))
    for aug in auglist:
        im = aug(im)
    im = nd.transpose(im, (2,0,1))
    return (im, nd.array([label]).asscalar().astype('float32'))

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
    num_epochs = 10
    learning_rate = 0.1
    weight_decay = 5e-4
    lr_period = 80
    lr_decay = 0.1
    train_data, valid_data, test_data = load_data()
    print("load data train_size: %s, valid_size: %s, test_size: %s" % (len(train_data), len(valid_data), len(test_data)))
    net = get_net(ctx)
    net.hybridize()
    train(net, train_data, valid_data, num_epochs, learning_rate,
          weight_decay, ctx, lr_period, lr_decay)



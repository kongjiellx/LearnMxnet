#!/usr/bin/env python
# encoding: utf-8

import os, shutil

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_train_valid(path):
    paths = os.listdir(path)
    train_path = path + '../train/'
    valid_path = path + '../valid/'
    mkdir_if_not_exist(train_path)
    mkdir_if_not_exist(valid_path)

    for p in paths:
        files = os.listdir(os.path.join(path, p))
        mkdir_if_not_exist(train_path +  p)
        mkdir_if_not_exist(valid_path +  p)
        for f in files:
            num = int(f.split('.')[0].split('_')[1])
            if num < 550:
                shutil.copy(path +  p + '/' +  f, train_path +  p + '/' +  f)
            else:
                shutil.copy(path +  p + '/' +  f, valid_path +  p + '/' +  f)


if __name__ == '__main__':
    path = '../../data/train_valid/'
    split_train_valid(path)



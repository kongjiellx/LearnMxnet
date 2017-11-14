import os
import shutil
import random

def reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio, part_rate):
    # 读取训练数据标签。
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # 跳过文件头行（栏名称）。
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((int(idx), label) for idx, label in tokens))
    labels = set(idx_label.values())

    num_train = len(os.listdir(os.path.join(data_dir, train_dir))) * part_rate
    num_train_tuning = int(num_train * (1 - valid_ratio))
    num_valid_tuning = int(num_train * valid_ratio)
    assert 0 < num_train_tuning < num_train
    num_train_tuning_per_label = num_train_tuning // len(labels)
    num_valid_tuning_per_label = num_valid_tuning // len(labels)
    train_label_count = dict()
    valid_label_count = dict()

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 整理训练和验证集。
    train_files = os.listdir(os.path.join(data_dir, train_dir))
    random.shuffle(train_files)
    for train_file in train_files:
        try:
            idx = int(train_file.split('.')[0])
        except:
            continue
        label = idx_label[idx]
        # mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        # shutil.copy(os.path.join(data_dir, train_dir, train_file),
        #             os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in train_label_count or train_label_count[label] < num_train_tuning_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
            train_label_count[label] = train_label_count.get(label, 0) + 1
        elif label not in valid_label_count or valid_label_count[label] < num_valid_tuning_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))
            valid_label_count[label] = valid_label_count.get(label, 0) + 1
    print("train_len: %s" % sum(train_label_count.values()))
    print("valid_len: %s" % sum(valid_label_count.values()))

    # 整理测试集。
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))


if __name__ == "__main__":
    train_dir = 'train'
    test_dir = 'test'
    batch_size = 128

    data_dir = '../../data/'
    label_file = 'trainLabels.csv'
    input_dir = 'train_valid_test'
    valid_ratio = 0.1
    reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio, 1)

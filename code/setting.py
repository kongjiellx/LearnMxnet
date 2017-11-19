data_dir = '../data/'
label_file = 'trainLabels.csv'
input_dir = 'train_valid_test'
train_dir = 'train_tiny'
test_dir = 'test_tiny'
input_str = data_dir + input_dir + '/'
model_path = '../model/cifar10.params'

batch_size = 64
num_epochs = 100
learning_rate = 0.1
num_outputs = 10

lr_period = 50
lr_decay = 0.2
weight_decay = 1e-4


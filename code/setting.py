data_dir = '../data/'
label_file = 'trainLabels.csv'
input_dir = 'train_valid_test'
train_dir = 'train_tiny'
test_dir = 'test_tiny'
input_str = data_dir + input_dir + '/'
model_path = '../model/cifar10.params'

batch_size = 128
num_epochs = 50
learning_rate = 0.1
num_outputs = 10

weight_decay = 5e-4
lr_period = 80
lr_decay = 0.1

data_dir = '../data/'
label_file = 'trainLabels.csv'
input_dir = 'train_valid_test'
train_dir = 'train_tiny'
test_dir = 'test_tiny'
input_str = data_dir + input_dir + '/'
model_path = '../model/cifar10.params'

batch_size = 2048
num_epochs = 100
learning_rate = 0.1
num_outputs = 10

lr_period = 10
lr_decay = 0.5

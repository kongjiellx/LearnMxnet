data_dir = '../data/'
label_file = 'trainLabels.csv'
input_dir = 'train_valid_test'
train_dir = 'train_tiny'
test_dir = 'test_tiny'
input_str = data_dir + input_dir + '/'
model_path = '../model/pig.params'

batch_size = 128
num_epochs = 220
learning_rate = 0.1
num_outputs = 30

lr_period = 50
lr_decay = 0.2
weight_decay = 1e-4


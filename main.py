from networks.models import Model
from networks.solver import Solver
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


num_train = 1000
num_val = 500

small_data = {
  'X_train': x_train[:num_train],
  'y_train': x_train[:num_train],
  'X_val': x_test[:num_val],
  'y_val': x_test[:num_val],
}

model = Model(hidden_dims=[128,256,512],dropout=0.75,normalization='batchnorm')
solver = Solver(model, small_data,
                print_every=10, num_epochs=20, batch_size=25,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-2,
                }
         )
solver.train()






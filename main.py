from networks.models import Model
from networks.solver import Solver
from keras.datasets import mnist,fashion_mnist
from keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
'''
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


mean_image = np.mean(x_train, axis=0)
x_train -= mean_image
x_test -= mean_image
'''


num_train = 1000
num_val = 500



small_data = {
  'X_train': x_train,
  'y_train': y_train,
  'X_val': x_test,
  'y_val': y_test,
}

def load_data(name):
  if name =='mnist':
    data = mnist
  elif name == 'fashion_mnist':
    data = fashion_mnist

  (x_train, y_train), (x_test, y_test) = data.load_data()

  small_data = {
    'X_train': x_train,
    'y_train': y_train,
    'X_val': x_test,
    'y_val': y_test,
  }

  return small_data



print('--'*20)
model = Model(hidden_dims=[128,256,512],dropout=0.3,num_networks=3,input_dim=784,normalization='batchnorm',sub_network=3)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
model.predict(x_test, y_test,which_network=[0,0,0])
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
model.predict(x_test, y_test,which_network=[1,1,1])
model.define_parameters([0,0,0],trianable_mask=[1,1,1])
solver = Solver(model, small_data,
                print_every=10, num_epochs=10, batch_size=100,
                update_rule='adam',checkpoint_name ='./check_point/first',
                optim_config={
                  'learning_rate': 5e-4,
                }
         )
solver.train()
print('--'*20)
small_data = load_data('fashion_mnist')
model.define_parameters([1,1,1],trianable_mask=[1,1,1])
solver = Solver(model, small_data,
                print_every=10, num_epochs=10, batch_size=100,
                update_rule='adam',checkpoint_name='./check_point/second',
                optim_config={
                  'learning_rate': 5e-4,
                }
         )
solver.train()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
model.predict(x_test, y_test,which_network=[0,0,0])
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
model.predict(x_test, y_test,which_network=[1,1,1])






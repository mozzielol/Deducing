from networks.models import Model
from networks.solver import Solver
from keras.datasets import mnist,fashion_mnist
from keras.utils import to_categorical
import numpy as np
from networks.data_utils import get_CIFAR10_data

'''
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


mean_image = np.mean(x_train, axis=0)
x_train -= mean_image
x_test -= mean_image
'''


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


#1. clustering
#2. 


#small_data = load_data('mnist')
small_data = get_CIFAR10_data()
model = Model(hidden_dims=[128,256,512],dropout=0.3,num_networks=5,input_dim=3072,sub_network=3)

model.define_parameters(which_network=[0,0,0,0,0],trainable_mask=[1,1,1,1,1])
solver = Solver(model, small_data,
                print_every=10, num_epochs=30, batch_size=128,
                update_rule='adam',checkpoint_name ='./check_point/first',
                optim_config={
                  'learning_rate': 5e-4,
                }
         )
solver.train()
print('--'*20)

model.predict(small_data['X_val'], small_data['y_val'])

'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model.predict(x_test, y_test,which_network=[0,0,0])
model.predict(x_test, y_test,which_network=[1,1,1])
model.predict(x_test, y_test,which_network=[2,2,2])
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
model.predict(x_test, y_test,which_network=[0,0,0])
model.predict(x_test, y_test,which_network=[1,1,1])
model.predict(x_test, y_test,which_network=[2,2,2])

'''





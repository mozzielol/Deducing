from builtins import range
from builtins import object
import numpy as np

from networks.layer.layers import *
from networks.layer.layer_utils import *
from copy import deepcopy



class Model(object): 
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None,
                 num_networks = 1, sub_network = 1):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        - network_param:
            order of the network, the part of network, the parameters
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.num_networks = num_networks
        self.network_param = {}
        self.sub_network = sub_network

        net_dims = [input_dim] + hidden_dims + [num_classes]
        self.net_dims = net_dims
        #1. initialze the network
        network_param = {}
        for i in range(self.num_layers):
            self.params['W%d'%(i+1)] = np.random.normal(loc=0.0,scale=weight_scale,size=(net_dims[i],net_dims[i+1]))
            self.params['b%d'%(i+1)] = np.zeros(net_dims[i+1])
            if (self.normalization is not None) & (i!=self.num_layers - 1):
                self.params['gamma%d'%(i+1)] = np.ones(net_dims[i+1])
                self.params['beta%d'%(i+1)] = np.zeros(net_dims[i+1])
            step = net_dims[i] // num_networks
            start = 0

            step2 = net_dims[i+1] // num_networks
            start2 = 0
        #2. Take Apart the network
            for e in range(num_networks):
                end = 0
                end2 = 0
                if e == num_networks - 1:
                    end = net_dims[i]
                    end2 = net_dims[i+1]
                else:
                    end = step * (e+1)
                    end2 = step2 * (e+1)
                network_param[(e,'W%d'%(i+1))] = self.params['W%d'%(i+1)][start:end]
                network_param[(e,'b%d'%(i+1))] = self.params['b%d'%(i+1)][start2:end2]
                if (normalization is not None) & (i!=self.num_layers - 1):
                    network_param[(e,'gamma%d'%(i+1))] = self.params['gamma%d'%(i+1)][start2:end2]
                    network_param[(e,'beta%d'%(i+1))] = self.params['beta%d'%(i+1)][start2:end2]
                start = end
                start2 = end2

        #Take apart the network
        for i in range(sub_network):
            self.network_param[i] = deepcopy(network_param)



        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed


        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for i in self.network_param:
            for k, v in self.network_param[i].items():
                self.network_param[i][k] = v.astype(dtype)


        #Approach 1 - Reference Points

        #1. define the sub-vector index
        step = input_dim // num_networks
        reference_index = np.arange(input_dim)
        start = 0
        end = 0
        for i in range(num_networks):
            if i == num_networks -1 :
                end = len(reference_index)
            else:
                end = step * (i+1)
            reference_index[start:end] = i
            start = end
        self.reference_index = reference_index
        #2. define the reference point
        self.reference_point = {}
        sub_vector = []
        
        for i in range(num_networks):
            mask = reference_index == i
            sub_vector.append(reference_index.copy()[mask])
            

        start = 0
        end = 0
        for i in range(sub_network):
            self.reference_point[i] = []
            for m,n in enumerate(sub_vector):
                step = len(n) // sub_network 
                if i == sub_network:
                    end = len(n)
                else:
                    end = (i+1) * step
                temp = n.copy()
                temp[start:end] = 1
                temp[:start] = 0
                temp[end:] = 0
                self.reference_point[i].append(temp)
            start = end

        

                
                

        



    def update_parameters(self):
        network_param = {}
        for i in range(self.num_layers):
            step = self.net_dims[i] // self.num_networks
            start = 0

            step2 = self.net_dims[i+1] // self.num_networks
            start2 = 0
            
            for e in range(self.num_networks):
                end = 0
                end2 = 0
                if e == self.num_networks - 1:
                    end = self.net_dims[i]
                    end2 = self.net_dims[i+1]
                else:
                    end = step * (e+1)
                    end2 = step2 * (e+1)
                network_param[(e,'W%d'%(i+1))] = self.params['W%d'%(i+1)][start:end].copy()
                network_param[(e,'b%d'%(i+1))] = self.params['b%d'%(i+1)][start2:end2].copy()
                if (self.normalization is not None) & (i!=self.num_layers - 1):
                    network_param[(e,'gamma%d'%(i+1))] = self.params['gamma%d'%(i+1)][start2:end2].copy()
                    network_param[(e,'beta%d'%(i+1))] = self.params['beta%d'%(i+1)][start2:end2].copy()
                start = end
                start2 = end2

        for i in range(self.num_layers):
            for n,j in enumerate(self.which_network):
                self.network_param[j][(n,'W%d'%(i+1))] = deepcopy(network_param[(n,'W%d'%(i+1))])
                self.network_param[j][(n,'b%d'%(i+1))] = deepcopy(network_param[(n,'b%d'%(i+1))])
                if (self.normalization is not None) & (i!=self.num_layers - 1):
                    self.network_param[j][(n,'gamma%d'%(i+1))] = deepcopy(network_param[(n,'gamma%d'%(i+1))])
                    self.network_param[j][(n,'beta%d'%(i+1))] = deepcopy(network_param[(n,'beta%d'%(i+1))])




    def define_parameters(self,which_network= [0],trainable_mask=[0]):
        self.which_network = which_network


        if len(which_network)!= self.num_networks:
            raise ValueError('network length is not sufficient')
        if len(which_network)!= len(trainable_mask):
            print(len(which_network),len(trainable_mask))
            raise ValueError('trainable_mask length is not sufficient')
        
        self.training_mask = {}

        for i in range(self.num_layers):
            for n,j in enumerate(which_network):
                
                if n == 0:
                    self.params['W%d'%(i+1)] = deepcopy(self.network_param[j][(n,'W%d'%(i+1))])
                    self.params['b%d'%(i+1)] = deepcopy(self.network_param[j][(n,'b%d'%(i+1))])
                    mask = self._create_mask('W%d'%(i+1),j,n,trainable_mask[n])
                    self.training_mask['W%d'%(i+1)] = mask
                    mask = self._create_mask('b%d'%(i+1),j,n,trainable_mask[n])
                    self.training_mask['b%d'%(i+1)] = mask
                    if (self.normalization is not None) & (i!=self.num_layers - 1):
                        self.params['gamma%d'%(i+1)] = deepcopy(self.network_param[j][(n,'gamma%d'%(i+1))])
                        self.params['beta%d'%(i+1)] = deepcopy(self.network_param[j][(n,'beta%d'%(i+1))])

                        mask = self._create_mask('gamma%d'%(i+1),j,n,trainable_mask[n])
                        self.training_mask['gamma%d'%(i+1)] = mask
                        mask = self._create_mask('beta%d'%(i+1),j,n,trainable_mask[n])
                        self.training_mask['beta%d'%(i+1)] = mask
                        #print(self.params['W%d'%(i+1)].shape,self.params['b%d'%(i+1)].shape,self.params['gamma%d'%(i+1)].shape,self.params['beta%d'%(i+1)].shape)
                else:
                    self.params['W%d'%(i+1)] = np.concatenate((self.params['W%d'%(i+1)],deepcopy(self.network_param[j][(n,'W%d'%(i+1))])))
                    self.params['b%d'%(i+1)] = np.concatenate((self.params['b%d'%(i+1)],deepcopy(self.network_param[j][(n,'b%d'%(i+1))])))
                    mask = self._create_mask('W%d'%(i+1),j,n,trainable_mask[n])
                    self.training_mask['W%d'%(i+1)] = np.concatenate((self.training_mask['W%d'%(i+1)],mask))
                    mask = self._create_mask('b%d'%(i+1),j,n,trainable_mask[n])
                    self.training_mask['b%d'%(i+1)] = np.concatenate((self.training_mask['b%d'%(i+1)],mask))
                    if (self.normalization is not None) & (i!=self.num_layers - 1):
                        self.params['gamma%d'%(i+1)] = np.concatenate((self.params['gamma%d'%(i+1)],deepcopy(self.network_param[j][(n,'gamma%d'%(i+1))])))
                        self.params['beta%d'%(i+1)] = np.concatenate((self.params['beta%d'%(i+1)],deepcopy(self.network_param[j][(n,'beta%d'%(i+1))])))

                        mask = self._create_mask('gamma%d'%(i+1),j,n,trainable_mask[n])
                        self.training_mask['gamma%d'%(i+1)] = np.concatenate((self.training_mask['gamma%d'%(i+1)],mask))
                        mask = self._create_mask('beta%d'%(i+1),j,n,trainable_mask[n])
                        self.training_mask['beta%d'%(i+1)] = np.concatenate((self.training_mask['beta%d'%(i+1)],mask))
                        #print(self.params['W%d'%(i+1)].shape,self.params['b%d'%(i+1)].shape,self.params['gamma%d'%(i+1)].shape,self.params['beta%d'%(i+1)].shape)



    
    def _create_mask(self,para_name,j,n,num):
        mask = None
        mask = self.network_param[j][(n,para_name)].copy()
        mask.fill(num)

        return mask.astype(bool)



    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None


        cache = {}
        layer_output = {}
        dropout_cache = {}
        layer_output[0] = X
        if self.normalization=='batchnorm':
            for i in range(1, self.num_layers +1):
                if i != self.num_layers:
                    layer_output[i], cache[i] = batch_relu_forward(
                      layer_output[i-1],
                      self.params['W%d' %i],
                      self.params['b%d' %i],
                      self.params['gamma%d' %i],
                      self.params['beta%d' %i],
                      bn_param=self.bn_params[i-1]
                      )
                    if self.use_dropout:
                        layer_output[i], dropout_cache[i] = dropout_forward(layer_output[i],self.dropout_param)

                else:
                    layer_output[i], cache[i] = affine_forward(
                                          layer_output[i-1],
                                          self.params['W%d' %i],
                                          self.params['b%d' %i]
                                          )

        elif self.normalization is None:
            for i in range(1,self.num_layers+1):
                if i != self.num_layers:
                    layer_output[i], cache[i] = affine_relu_forward(layer_output[i-1], self.params['W{}'.format(i)],self.params['b{}'.format(i)])
                    if self.use_dropout:
                        layer_output[i],dropout_cache[i] = dropout_forward(layer_output[i],self.dropout_param)
                else:
                    layer_output[i], cache[i] = affine_forward(layer_output[i-1],self.params['W%d'%i],self.params['b%d'%i])
            
        scores = layer_output[self.num_layers]


        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores,y)

        for i in range(1, self.num_layers+1):
            loss += 0.5*self.reg*np.sum(self.params['W%d' %i]*self.params['W%d' %i])

        if self.normalization == 'batchnorm':
            for i in range(self.num_layers, 0, -1):
                if i!=self.num_layers:
                    if self.use_dropout:
                        dscores = dropout_backward(dscores,dropout_cache[i])
                    (dscores, grads['W%d' %i], grads['b%d' %i],
                     grads['gamma%d' %i], grads['beta%d' %i]) = batch_relu_backward(
                                                                dscores,
                                                                cache[i]
                                                                )

                else:
                    dscores, grads['W%d' %i], grads['b%d' %i] = affine_backward(
                                                                dscores,
                                                                cache[i]
                                                                )
                # Add regularization to the weights gradient
                grads['W%d' %i] += self.reg*self.params['W%d' %i]
                

        elif self.normalization is None:
            for i in range(self.num_layers,0,-1):
                if i == self.num_layers:
                    dscores, grads['W%d'%i], grads['b%d'%i] = affine_backward(dscores,cache[i])
                else:
                    if self.use_dropout:
                        dscores = dropout_backward(dscores,dropout_cache[i])
                    dscores, grads['W%d'%i], grads['b%d'%i] = affine_relu_backward(dscores,cache[i])
                grads['W%d'%i] += self.reg * self.params['W%d'%i]
                


        return loss, grads


    def predict(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            self._which_network(X[start:end])
            scores = self.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        print('Accuracy is %f'%acc)

        return acc

    def _which_network(self,X_batch):
        sub_vector = np.average(X_batch.reshape(X_batch.shape[0],-1),axis=0)
        
        which_network = []
        for i in range(self.num_networks):
            mask = self.reference_index==i
            vector = sub_vector[mask]
            dist = None
            num = 0
            
            for n in range(self.sub_network):
                cur_dist = np.linalg.norm(vector - self.reference_point[n][i])
                if (dist is None) or (cur_dist < dist):
                    dist = cur_dist
                    num = n
            which_network.append(num)
        if which_network != self.which_network:
            pass
            #print(which_network)
        trainable_mask = [0] * len(which_network)
        self.define_parameters(which_network,trainable_mask=trainable_mask)
















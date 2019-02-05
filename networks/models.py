from builtins import range
from builtins import object
import numpy as np

from networks.layer.layers import *
from networks.layer.layer_utils import *



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
                 weight_scale=1e-2, dtype=np.float32, seed=None):
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
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        net_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W%d'%(i+1)] = np.random.normal(loc=0.0,scale=weight_scale,size=(net_dims[i],net_dims[i+1]))
            self.params['b%d'%(i+1)] = np.zeros(net_dims[i+1])
            if (self.normalization is not None) & (i!=self.num_layers - 1):
                self.params['gamma%d'%(i+1)] = np.ones(net_dims[i+1])
                self.params['beta%d'%(i+1)] = np.zeros(net_dims[i+1])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


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
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
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

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
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
                


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

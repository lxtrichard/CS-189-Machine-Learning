from builtins import range
from builtins import object
import numpy as np

from layers import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of [H, ...], and perform classification over C classes.

    The architecure should be like affine - relu - affine - softmax for a one
    hidden layer network, and affine - relu - affine - relu- affine - softmax for
    a two hidden layer network, etc.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim, hidden_dim=[10, 5], num_classes=8,
                 weight_scale=0.1):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A list of integer giving the sizes of the hidden layers
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {}
        self.hidden_dim = hidden_dim
        ############################################################################
        # TODO: Initialize the weights and biases of the net. Weights              #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.num_layers = 1+len(hidden_dim)
        all_dims = [input_dim] + hidden_dim + [num_classes]
        for k in range(1,self.num_layers+1):
            self.params["W{}".format(str(k))] = np.random.randn(all_dims[k-1],all_dims[k])*weight_scale
            self.params["b{}".format(str(k))] = np.zeros(all_dims[k])


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the net, computing the              #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        caches = []
        L = self.num_layers
        for i in range(1,L):
            X, cache1 = affine_forward(X,self.params["W"+str(i)],self.params["b"+str(i)])
            X, cache2 = relu_forward(X)
            cache = (cache1, cache2)
            caches.append(cache)
        
        scores, cache = affine_forward(X,self.params["W"+str(L)],self.params["b"+str(L)])
        caches.append(cache)
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the net. Store the loss            #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k].                                                          #
        ############################################################################
        L = len(caches)
        loss, final_dev = softmax_loss(scores,y)
        cur_cache = caches[-1]
        cur_dev = final_dev
        cur_dev, dw, db = affine_backward(cur_dev,cur_cache)
        grads["W"+str(L)] = dw + self.params["W"+str(L)]
        grads["b"+str(L)] = db

        for i in range(L-1,0,-1):
            cur_cache = caches[i-1]
            affine_cache, relu_cache = cur_cache
            dx = relu_backward(cur_dev, relu_cache)
            cur_dev, dw, db = affine_backward(dx,affine_cache)
            grads["W"+str(i)] = dw + self.params["W"+str(i)]
            grads["b"+str(i)] = db

        return loss, grads

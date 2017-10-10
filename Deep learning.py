import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2, inspect, sys
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.framework import dtypes, random_seed
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

class Data(object):

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype

        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid dtype %r, expect uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s'%(images.shape,labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
      """Return the next `batch_size` examples from this data set."""
      if fake_data:
          fake_image = [1] * 784
          if self.one_hot:
              fake_label = [1] + [0] * 9
          else:
              fake_label = 0
          return [fake_image for _ in xrange(batch_size)], [
              fake_label for _ in xrange(batch_size)
          ]
        
      start = self._index_in_epoch
      if self._epochs_completed == 0 and start == 0 and shuffle:
          perm0 = np.arange(self._num_examples)
          np.random.shuffle(perm0)
          self._images = self.images[perm0]
          self._labels = self.labels[perm0]
    
      if start + batch_size > self._num_examples:
          self._epochs_completed += 1

          rest_num_examples = self._num_examples - start
          images_rest_part = self._images[start:self._num_examples]
          labels_rest_part = self._labels[start:self._num_examples]

          if shuffle:
              perm = np.arange(self._num_examples)
              np.random.shuffle(perm)
              self._images = self.images[perm]
              self._labels = self.labels[perm]
              
          start = 0
          self._index_in_epoch = batch_size - rest_num_examples
          end = self._index_in_epoch
          images_new_part = self._images[start:end]
          labels_new_part = self._labels[start:end]
          return np.concantenate((images_rest_part,images_new_part),axis=0),\
                 np.concatenate((labels_rest_part,labels_new_part), axis=0)
      else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          return self._images[start:end], self._labels[start:end]
        
def _placeholder(dtype, shape=None, name=None):
   r"""A placeholder op for a value that will be fed into the computation.

   N.B. This operation will fail with an error if it is executed. It is
   intended as a way to represent a value that will always be fed, and to
   provide attrs that enable the fed value to be checked at runtime.

   Args:
     dtype: A `tf.DType`. The type of elements in the tensor.
     shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
       (Optional) The shape of the tensor. If the shape has 0 dimensions, the
       shape is unconstrained.
     name: A name for the operation (optional).

   Returns:
     A `Tensor` of type `dtype`.
     A placeholder tensor that must be replaced using the feed mechanism.
    """
   result = _op_def_lib.apply_op("Placeholder", dtype=dtype, shape=shape,
                                name=name)
   return result

def stepFunction(t):
    """
    perceptronStep helper function. takes a logit as input
    """
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    """
    perceptronStep helper function. takes an data value, a weight, and a
    bias as input
    """
    return stepFunction((np.matmul(X,W)+b)[0])

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    """
    updates weights and biases to help improve perceptron best fit line.
    X is inputs of the data, y is the labels, W is the weights(as an array),
    and bias is b. returns updated weight array and bias
    """
    for i in range(len(y)):
        y_hat = prediction(X[i],W,b)
        n = 0
        if y[i]-y_hat == 1:
            while n < len(W):
                W[n] += X[i][n]*learn_rate
                n+=1
            b += learn_rate
        elif y[i]-y_hat == -1:
            while n < len(W):
                W[n] -= X[i][n]*learn_rate
                n+=1
            b -= learn_rate
    return W, b
    
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    """
    runs perceptronStep repeatedly on a dataset. returns a few boundary
    lines obtained in the iterations
    """
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max

    boundary_lines = []
    for i in range(num_epochs):
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

def softmax(logit_list):
    """
    compute softmax values for a list of logits
    """
    expL = np.exp(logit_list)
    sumExpL = sum(logit_list)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result

def cross_entropy(Y, P):
    """
    computes cross_entropy for two lists
    """
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

def sigmoid(x):
    """
    x is the summation of weight vector dot product with input vector added
    to bias.
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    derivative of the sigmoid function
    """
    return sigmoid(x)*(1-sigmoid(x))

def feed_forward(inputs, weights, biases):
    """
    performs a neural network feedforward operation.
    from neural network strucutre:
    (input1)----(weight1)----(hidden1)-------(weight5)------(output1)
        =                     =    =                          =
           -(weight3)-     =           =(weight7)=        =
                        =                             =
                        =                             =
           -(weight2)-     =           =(weight6)=        = 
        -                    =     =                          =
    (input2)----(weight4)----(hidden2)-------(weight8)-------(output2)
                   =                             =
            (bias1)                       (bias2)
              
    inputs is a np array of inputs structured (input1 input2), weights is
    an np array of weights structured
    (weight1  weight2
     weight3  weight4)
    biases is a np array of biases structured (bias1 bias2)
    with each entry corresponding to the bias of a hidden layer.
    """
    pass


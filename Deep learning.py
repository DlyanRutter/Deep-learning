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


def softmax(x):
    """
    compute softmax values for x
    """
    x = x * 10
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

keep_rate = 0.8

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):           #size of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],\
                          padding='SAME')

def convolutional_neural_network(x):

    weights = {'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}
               
    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
               'b_conv2': tf.Variable(tf.random_normal([64])),
               'b_fc': tf.Variable(tf.random_normal([1024])),
               'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1,28,28,1])

    conv1 =  tf.nn.relu( conv2d(x,weights['W_conv1']) + biases['b_conv1'] )
    conv1 =  maxpool2d(conv1)

    conv2 =  tf.nn.relu( conv2d(conv1,weights['W_conv2']) + biases['b_conv2'] )
    conv2 = maxpool2d(conv2)
                   
    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
               
                _,c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y}) 
                epoch_loss += c
              
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, cv2, inspect, sys
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.framework import dtypes, random_seed

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


import sys
sys.path.append('../..') # so we can see the rts.

import typedflow_rts as tyf
import tensorflow as tf
import numpy as np
from mnist_model import mkModel
import os

# comment out if you don't have CUDA
tyf.cuda_use_one_free_device()

model = mkModel(tf.train.AdamOptimizer(1e-4))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def train_generator(batch_size):
    for _ in range(1000):
        (x,y) = mnist.train.next_batch(100)
        yield {"x":x,"y":y}

sess = tf.Session()

tyf.initialize_params(sess,model)
tyf.train(sess,model,train_generator)

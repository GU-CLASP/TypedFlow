import typedflow_rts
import tensorflow as tf
import numpy as np
import mnist_model
import os

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model = typedflow_example.mkModel()


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def train_generator(batch_size):
    for _ in range(1000):
        yield mnist.train.next_batch(100)

def valid_generator():
    return []

sess = tf.Session()

typedflow_rts.initialize_params(sess,model)
typedflow_rts.train(sess,model,optimizer=tf.train.AdamOptimizer(1e-4),train_generator)

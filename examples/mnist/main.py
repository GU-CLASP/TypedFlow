import sys
sys.path.append('../..') # so we can see the rts.

import typedflow_rts as tyf
import tensorflow as tf
import numpy as np
from mnist_model import mkModel,runModel
import os

# comment out if you don't have CUDA
tyf.cuda_use_one_free_device()

optimizer = tf.keras.optimizers.Adam(1e-4)

# import tfds.image_classification.MNIST as mnist # need to package tfds

def train_generator(batch_size):
    for _ in range(1000):
        # (x,y) = mnist.batch(100)
        yield {"x":np.zeros((100,784), dtype=np.float32), # FIXME
               "y":np.zeros((100,10), dtype=np.float32)
               }


model = mkModel()

tyf.train(optimizer,model,runModel,train_generator)

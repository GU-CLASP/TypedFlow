
import tensorflow as tf
def mkModel(optimizer=tf.train.AdamOptimizer()):
  
  training_phase=tf.placeholder(tf.bool, shape=[], name="training_phase")
  x=tf.placeholder(tf.float32, shape=[None,784], name="x")
  y=tf.placeholder(tf.float32, shape=[None,10], name="y")
  var0=tf.Variable(tf.transpose(tf.reshape(tf.random_uniform([800,1],
                                                             minval=-8.654846e-2,
                                                             maxval=8.654846e-2,
                                                             dtype=tf.float32),
                                           [32,5,5,1]),
                                perm=[1,2,3,0]),
                   name="f1_filters",
                   trainable=True)
  var1=tf.Variable(
         tf.constant(0.1, shape=[32], dtype=tf.float32), name="f1_biases", trainable=True)
  var2=tf.Variable(tf.transpose(tf.reshape(tf.random_uniform([1600,32],
                                                             minval=-6.0633905e-2,
                                                             maxval=6.0633905e-2,
                                                             dtype=tf.float32),
                                           [64,5,5,32]),
                                perm=[1,2,3,0]),
                   name="f2_filters",
                   trainable=True)
  var3=tf.Variable(
         tf.constant(0.1, shape=[64], dtype=tf.float32), name="f2_biases", trainable=True)
  var4=tf.Variable(
         tf.random_uniform(
           [1024,3136], minval=-3.7977725e-2, maxval=3.7977725e-2, dtype=tf.float32),
         name="w1_w",
         trainable=True)
  var5=tf.Variable(tf.truncated_normal([1024], stddev=0.1, dtype=tf.float32),
                   name="w1_bias",
                   trainable=True)
  var6=tf.Variable(tf.random_uniform(
                     [10,1024], minval=-7.61755e-2, maxval=7.61755e-2, dtype=tf.float32),
                   name="w2_w",
                   trainable=True)
  var7=tf.Variable(tf.truncated_normal([10], stddev=0.1, dtype=tf.float32),
                   name="w2_bias",
                   trainable=True)
  var8=tf.add(tf.matmul(
                tf.nn.relu(
                  tf.add(tf.matmul(tf.reshape(
                                     tf.nn.max_pool(
                                       tf.nn.relu(
                                         tf.add(tf.nn.convolution(
                                                  tf.nn.max_pool(
                                                    tf.nn.relu(
                                                      tf.add(tf.nn.convolution(
                                                               tf.reshape(x, [-1,28,28,1]),
                                                               var0,
                                                               padding="SAME",
                                                               data_format="NHWC"),
                                                             var1)),
                                                    [1,2,2,1],
                                                    [1,2,2,1],
                                                    padding="SAME"),
                                                  var2,
                                                  padding="SAME",
                                                  data_format="NHWC"),
                                                var3)),
                                       [1,2,2,1],
                                       [1,2,2,1],
                                       padding="SAME"),
                                     [-1,3136]),
                                   tf.transpose(var4)),
                         var5)),
                tf.transpose(var6)),
              var7)
  var9=tf.equal(tf.argmax(var8, axis=1, output_type=tf.int32),
                tf.argmax(y, axis=1, output_type=tf.int32))
  var10=tf.reduce_mean(tf.cast(var9, tf.float32))
  var11=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=var8))
  var12=tf.nn.softmax(var8)
  var13=var11
  var14=var10
  var15=tf.trainable_variables()
  var16=optimizer.minimize(var13)
  return {"train":var16
         ,"params":var15
         ,"accuracy":var14
         ,"loss":var13
         ,"y_":var12
         ,"y":y
         ,"x":x
         ,"training_phase":training_phase
         ,"batch_size":None
         ,"optimizer":optimizer}
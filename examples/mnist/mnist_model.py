
import tensorflow as tf
def mkModel():
  
  x=tf.placeholder(tf.float32, shape=[None, 784])
  y=tf.placeholder(tf.float32, shape=[None, 10])
  var0=tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name="f1_fst")
  var1=tf.Variable(tf.constant(0.1, shape=[28, 28, 32]), name="f1_snd")
  var2=tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name="f2_fst")
  var3=tf.Variable(tf.constant(0.1, shape=[14, 14, 64]), name="f2_snd")
  var4=tf.Variable(tf.random_uniform(
                     [1024, 3136], minval=-0.1519109, maxval=0.1519109, dtype=tf.float32),
                   name="w1_fst")
  var5=tf.Variable(tf.truncated_normal([1024], stddev=0.1), name="w1_snd")
  var6=tf.Variable(tf.random_uniform(
                     [10, 1024], minval=-0.304702, maxval=0.304702, dtype=tf.float32),
                   name="w2_fst")
  var7=tf.Variable(tf.truncated_normal([10], stddev=0.1), name="w2_snd")
  var8=tf.add(
         tf.matmul(tf.nn.relu(
                     tf.add(tf.matmul(tf.reshape(
                                        tf.nn.max_pool(
                                          tf.nn.relu(
                                            tf.add(tf.nn.convolution(
                                                     tf.nn.max_pool(
                                                       tf.nn.relu(
                                                         tf.add(tf.nn.convolution(
                                                                  tf.reshape(
                                                                    x, [-1, 28, 28, 1]),
                                                                  var0,
                                                                  padding="SAME",
                                                                  data_format="NHWC"),
                                                                var1)),
                                                       [1, 2, 2, 1],
                                                       [1, 2, 2, 1],
                                                       padding="SAME"),
                                                     var2,
                                                     padding="SAME",
                                                     data_format="NHWC"),
                                                   var3)),
                                          [1, 2, 2, 1],
                                          [1, 2, 2, 1],
                                          padding="SAME"),
                                        [-1, 3136]),
                                      tf.transpose(var4)),
                            var5)),
                   tf.transpose(var6)),
         var7)
  var9=tf.equal(tf.argmax(var8, 1), tf.argmax(y, 1))
  var10=tf.reduce_mean(tf.cast(var9, tf.float32))
  var11=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=var8))
  var12=tf.nn.softmax(var8)
  return (x, y, var12, var10, var11)
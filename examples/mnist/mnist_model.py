
import tensorflow as tf
def mkModel():
  
  global var3, var4, var6, var7, var10, var11, var14, var15
  #[25,32]
  var12345=tf.random.uniform(
             [25,32], minval=-0.32444283, maxval=0.32444283, dtype=tf.float32) # 2
  #[5,5,1,32]
  var12346=tf.reshape(var12345, [5,5,1,32])
  var3=tf.Variable(name="f1f1_filters", trainable=True, initial_value=var12346)
  #[]
  var12347=tf.constant(0.1, shape=[], dtype=tf.float32)
  #[32]
  var12348=tf.broadcast_to(tf.reshape(var12347, [1]), [32])
  #[32]
  var12349=tf.reshape(var12348, [32])
  var4=tf.Variable(name="f1f1_biases", trainable=True, initial_value=var12349)
  #[800,64]
  var12350=tf.random.uniform(
             [800,64], minval=-8.3333336e-2, maxval=8.3333336e-2, dtype=tf.float32) # 5
  #[5,5,32,64]
  var12351=tf.reshape(var12350, [5,5,32,64])
  var6=tf.Variable(name="f2f2_filters", trainable=True, initial_value=var12351)
  #[64]
  var12352=tf.broadcast_to(tf.reshape(var12347, [1]), [64])
  #[64]
  var12353=tf.reshape(var12352, [64])
  var7=tf.Variable(name="f2f2_biases", trainable=True, initial_value=var12353)
  #[3136,1024]
  var12354=tf.random.uniform(
             [3136,1024], minval=-3.7977725e-2, maxval=3.7977725e-2, dtype=tf.float32) # 8
  var10=tf.Variable(name="w1w1_w", trainable=True, initial_value=var12354)
  #[1024]
  var12355=tf.random.truncated_normal([1024], stddev=0.1, dtype=tf.float32) # 9
  var11=tf.Variable(name="w1w1_bias", trainable=True, initial_value=var12355)
  #[1024,10]
  var12356=tf.random.uniform(
             [1024,10], minval=-7.61755e-2, maxval=7.61755e-2, dtype=tf.float32) # 12
  var14=tf.Variable(name="w2w2_w", trainable=True, initial_value=var12356)
  #[10]
  var12357=tf.random.truncated_normal([10], stddev=0.1, dtype=tf.float32) # 13
  var15=tf.Variable(name="w2w2_bias", trainable=True, initial_value=var12357)
  return {"batch_size":100,"parameters":[var3,var4,var6,var7,var10,var11,var14,var15]}
def runModel(isTraining, var0, var1):
  
  #[100,10]
  var12358=var1
  #[100,784]
  var12359=var0 # tf.cast(var0,dtype=tf.float32)
  #[100,28,28,1]
  var12360=tf.reshape(var12359, [100,28,28,1])
  #[5,5,1,32]
  var12361=var3
  #[100,28,28,32]
  print((var12360.dtype))
  print((var12361.dtype))
  var12362=tf.nn.convolution(var12360, var12361, padding="SAME", data_format="NHWC")
  #[100,784,32]
  var12363=tf.reshape(var12362, [100,784,32])
  #[32]
  var12364=var4
  #[784,32]
  var12365=tf.broadcast_to(tf.reshape(var12364, [1,32]), [784,32])
  #[100,784,32]
  var12366=tf.broadcast_to(tf.reshape(var12365, [1,784,32]), [100,784,32])
  #[100,784,32]
  var12367=tf.add(var12363, var12366)
  #[100,28,28,32]
  var12368=tf.reshape(var12367, [100,28,28,32])
  #[100,28,28,32]
  var12369=tf.nn.relu(var12368)
  #[100,28,28,32]
  var12370=tf.reshape(var12369, [100,28,28,32])
  #[100,14,14,32]
  var12371=tf.nn.pool(var12370, [2,2], "MAX", padding="SAME", strides=[2,2])
  #[100,14,14,32]
  var12372=tf.reshape(var12371, [100,14,14,32])
  #[5,5,32,64]
  var12373=var6
  #[100,14,14,64]
  var12374=tf.nn.convolution(var12372, var12373, padding="SAME", data_format="NHWC")
  #[100,196,64]
  var12375=tf.reshape(var12374, [100,196,64])
  #[64]
  var12376=var7
  #[196,64]
  var12377=tf.broadcast_to(tf.reshape(var12376, [1,64]), [196,64])
  #[100,196,64]
  var12378=tf.broadcast_to(tf.reshape(var12377, [1,196,64]), [100,196,64])
  #[100,196,64]
  var12379=tf.add(var12375, var12378)
  #[100,14,14,64]
  var12380=tf.reshape(var12379, [100,14,14,64])
  #[100,14,14,64]
  var12381=tf.nn.relu(var12380)
  #[100,14,14,64]
  var12382=tf.reshape(var12381, [100,14,14,64])
  #[100,7,7,64]
  var12383=tf.nn.pool(var12382, [2,2], "MAX", padding="SAME", strides=[2,2])
  #[100,3136]
  var12384=tf.reshape(var12383, [100,3136])
  #[3136,1024]
  var12385=var10
  #[100,1024]
  var12386=tf.matmul(var12384, var12385)
  #[100,1024]
  var12387=tf.reshape(var12386, [100,1024])
  #[1024]
  var12388=var11
  #[100,1024]
  var12389=tf.broadcast_to(tf.reshape(var12388, [1,1024]), [100,1024])
  #[100,1024]
  var12390=tf.add(var12387, var12389)
  #[100,1024]
  var12391=tf.nn.relu(var12390)
  #[100,1024]
  var12392=tf.reshape(var12391, [100,1024])
  #[1024,10]
  var12393=var14
  #[100,10]
  var12394=tf.matmul(var12392, var12393)
  #[100,10]
  var12395=tf.reshape(var12394, [100,10])
  #[10]
  var12396=var15
  #[100,10]
  var12397=tf.broadcast_to(tf.reshape(var12396, [1,10]), [100,10])
  #[100,10]
  var12398=tf.add(var12395, var12397)
  #[100]
  var12399=tf.nn.softmax_cross_entropy_with_logits(labels=var12358, logits=var12398)
  #[100]
  var12400=tf.reshape(var12399, [100])
  #[]
  var12401=tf.reduce_sum(var12400, axis=0)
  #[]
  var12402=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var12403=tf.broadcast_to(tf.reshape(var12402, [1]), [1])
  #[]
  var12404=tf.reshape(var12403, [])
  #[]
  var12405=tf.add(var12401, var12404)
  #[100,10]
  var12406=tf.reshape(var12398, [100,10])
  #[100,10]
  var12407=tf.nn.softmax(var12406, axis=1)
  #[100,10]
  var12408=tf.reshape(var12407, [100,10])
  #[100]
  var12409=tf.argmax(var12398, axis=1, output_type=tf.int32)
  #[100]
  var12410=tf.argmax(var12358, axis=1, output_type=tf.int32)
  #[100]
  var12411=tf.equal(var12409, var12410)
  #[100]
  var12412=tf.cast(var12411, tf.float32)
  return {"loss":var12405,"accuracy":var12412,"y_":var12408}

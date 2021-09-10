
import tensorflow as tf
def mkModel():
  
  #[25,32]
  var12345=tf.random.uniform(
             [25,32], minval=-0.32444283, maxval=0.32444283, dtype=tf.float32) # 0
  #[5,5,1,32]
  var12346=tf.reshape(var12345, [5,5,1,32])
  var12347=tf.Variable(name="f1_filters", trainable=True, initial_value=var12346)
  #[]
  var12348=tf.constant(0.1, shape=[], dtype=tf.float32)
  #[32]
  var12349=tf.broadcast_to(tf.reshape(var12348, [1]), [32])
  #[32]
  var12350=tf.reshape(var12349, [32])
  var12351=tf.Variable(name="f1_biases", trainable=True, initial_value=var12350)
  #[800,64]
  var12352=tf.random.uniform(
             [800,64], minval=-8.3333336e-2, maxval=8.3333336e-2, dtype=tf.float32) # 1
  #[5,5,32,64]
  var12353=tf.reshape(var12352, [5,5,32,64])
  var12354=tf.Variable(name="f2_filters", trainable=True, initial_value=var12353)
  #[64]
  var12355=tf.broadcast_to(tf.reshape(var12348, [1]), [64])
  #[64]
  var12356=tf.reshape(var12355, [64])
  var12357=tf.Variable(name="f2_biases", trainable=True, initial_value=var12356)
  #[3136,1024]
  var12358=tf.random.uniform(
             [3136,1024], minval=-3.7977725e-2, maxval=3.7977725e-2, dtype=tf.float32) # 2
  var12359=tf.Variable(name="w1_w", trainable=True, initial_value=var12358)
  #[1024]
  var12360=tf.random.truncated_normal([1024], stddev=0.1, dtype=tf.float32) # 3
  var12361=tf.Variable(name="w1_bias", trainable=True, initial_value=var12360)
  #[1024,10]
  var12362=tf.random.uniform(
             [1024,10], minval=-7.61755e-2, maxval=7.61755e-2, dtype=tf.float32) # 4
  var12363=tf.Variable(name="w2_w", trainable=True, initial_value=var12362)
  #[10]
  var12364=tf.random.truncated_normal([10], stddev=0.1, dtype=tf.float32) # 5
  var12365=tf.Variable(name="w2_bias", trainable=True, initial_value=var12364)
  return {"batch_size":100
         ,"parameters":[var12347
                       ,var12351
                       ,var12354
                       ,var12357
                       ,var12359
                       ,var12361
                       ,var12363
                       ,var12365]
         ,"paramsdict":{"f1_filters":var12347
                       ,"f1_biases":var12351
                       ,"f2_filters":var12354
                       ,"f2_biases":var12357
                       ,"w1_w":var12359
                       ,"w1_bias":var12361
                       ,"w2_w":var12363
                       ,"w2_bias":var12365}
         ,"placeholders":{"x":{"shape":[100,784],"dtype":tf.float32}
                         ,"y":{"shape":[100,10],"dtype":tf.float32}}}
@tf.function
def runModel(training_placeholder,
             f1_filters,
             f1_biases,
             f2_filters,
             f2_biases,
             w1_w,
             w1_bias,
             w2_w,
             w2_bias,
             x,
             y):
  
  #[100,784]
  var12366=x
  #[100,28,28,1]
  var12367=tf.reshape(var12366, [100,28,28,1])
  #[5,5,1,32]
  var12368=f1_filters
  #[100,28,28,32]
  var12369=tf.nn.convolution(var12367, var12368, padding="SAME", data_format="NHWC")
  #[100,784,32]
  var12370=tf.reshape(var12369, [100,784,32])
  #[32]
  var12371=f1_biases
  #[784,32]
  var12372=tf.broadcast_to(tf.reshape(var12371, [1,32]), [784,32])
  #[100,784,32]
  var12373=tf.broadcast_to(tf.reshape(var12372, [1,784,32]), [100,784,32])
  #[100,784,32]
  var12374=tf.add(var12370, var12373)
  #[100,28,28,32]
  var12375=tf.reshape(var12374, [100,28,28,32])
  #[100,28,28,32]
  var12376=tf.nn.relu(var12375)
  #[100,28,28,32]
  var12377=tf.reshape(var12376, [100,28,28,32])
  #[100,14,14,32]
  var12378=tf.nn.pool(var12377, [2,2], "MAX", strides=[2,2], padding="SAME")
  #[100,14,14,32]
  var12379=tf.reshape(var12378, [100,14,14,32])
  #[5,5,32,64]
  var12380=f2_filters
  #[100,14,14,64]
  var12381=tf.nn.convolution(var12379, var12380, padding="SAME", data_format="NHWC")
  #[100,196,64]
  var12382=tf.reshape(var12381, [100,196,64])
  #[64]
  var12383=f2_biases
  #[196,64]
  var12384=tf.broadcast_to(tf.reshape(var12383, [1,64]), [196,64])
  #[100,196,64]
  var12385=tf.broadcast_to(tf.reshape(var12384, [1,196,64]), [100,196,64])
  #[100,196,64]
  var12386=tf.add(var12382, var12385)
  #[100,14,14,64]
  var12387=tf.reshape(var12386, [100,14,14,64])
  #[100,14,14,64]
  var12388=tf.nn.relu(var12387)
  #[100,14,14,64]
  var12389=tf.reshape(var12388, [100,14,14,64])
  #[100,7,7,64]
  var12390=tf.nn.pool(var12389, [2,2], "MAX", strides=[2,2], padding="SAME")
  #[100,3136]
  var12391=tf.reshape(var12390, [100,3136])
  #[3136,1024]
  var12392=w1_w
  #[100,1024]
  var12393=tf.matmul(var12391, var12392)
  #[100,1024]
  var12394=tf.reshape(var12393, [100,1024])
  #[1024]
  var12395=w1_bias
  #[100,1024]
  var12396=tf.broadcast_to(tf.reshape(var12395, [1,1024]), [100,1024])
  #[100,1024]
  var12397=tf.add(var12394, var12396)
  #[100,1024]
  var12398=tf.nn.relu(var12397)
  #[100,1024]
  var12399=tf.reshape(var12398, [100,1024])
  #[1024,10]
  var12400=w2_w
  #[100,10]
  var12401=tf.matmul(var12399, var12400)
  #[100,10]
  var12402=tf.reshape(var12401, [100,10])
  #[10]
  var12403=w2_bias
  #[100,10]
  var12404=tf.broadcast_to(tf.reshape(var12403, [1,10]), [100,10])
  #[100,10]
  var12405=tf.add(var12402, var12404)
  #[100]
  var12406=tf.argmax(var12405, axis=1, output_type=tf.int32)
  #[100,10]
  var12407=y
  #[100]
  var12408=tf.argmax(var12407, axis=1, output_type=tf.int32)
  #[100]
  var12409=tf.equal(var12406, var12408)
  #[100]
  var12410=tf.cast(var12409, tf.float32)
  #[100]
  var12411=tf.reshape(var12410, [100])
  #[]
  var12412=tf.reduce_mean(var12411, axis=0)
  #[]
  var12413=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var12414=tf.broadcast_to(tf.reshape(var12413, [1]), [1])
  #[]
  var12415=tf.reshape(var12414, [])
  #[]
  var12416=tf.add(var12412, var12415)
  #[100]
  var12417=tf.nn.softmax_cross_entropy_with_logits(labels=var12407, logits=var12405)
  #[100]
  var12418=tf.reshape(var12417, [100])
  #[]
  var12419=tf.reduce_mean(var12418, axis=0)
  #[100,10]
  var12420=tf.reshape(var12405, [100,10])
  #[100,10]
  var12421=tf.nn.softmax(var12420, axis=1)
  #[100,10]
  var12422=tf.reshape(var12421, [100,10])
  return {"loss":var12416,"accuracy":var12419,"y_":var12422}
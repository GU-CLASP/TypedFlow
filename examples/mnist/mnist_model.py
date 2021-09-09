
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
  
  #[100,10]
  var12366=y
  #[100,784]
  var12367=x
  #[100,28,28,1]
  var12368=tf.reshape(var12367, [100,28,28,1])
  #[5,5,1,32]
  var12369=f1_filters
  #[100,28,28,32]
  var12370=tf.nn.convolution(var12368, var12369, padding="SAME", data_format="NHWC")
  #[100,784,32]
  var12371=tf.reshape(var12370, [100,784,32])
  #[32]
  var12372=f1_biases
  #[784,32]
  var12373=tf.broadcast_to(tf.reshape(var12372, [1,32]), [784,32])
  #[100,784,32]
  var12374=tf.broadcast_to(tf.reshape(var12373, [1,784,32]), [100,784,32])
  #[100,784,32]
  var12375=tf.add(var12371, var12374)
  #[100,28,28,32]
  var12376=tf.reshape(var12375, [100,28,28,32])
  #[100,28,28,32]
  var12377=tf.nn.relu(var12376)
  #[100,28,28,32]
  var12378=tf.reshape(var12377, [100,28,28,32])
  #[100,14,14,32]
  var12379=tf.nn.pool(var12378, [2,2], "MAX", strides=[2,2], padding="SAME")
  #[100,14,14,32]
  var12380=tf.reshape(var12379, [100,14,14,32])
  #[5,5,32,64]
  var12381=f2_filters
  #[100,14,14,64]
  var12382=tf.nn.convolution(var12380, var12381, padding="SAME", data_format="NHWC")
  #[100,196,64]
  var12383=tf.reshape(var12382, [100,196,64])
  #[64]
  var12384=f2_biases
  #[196,64]
  var12385=tf.broadcast_to(tf.reshape(var12384, [1,64]), [196,64])
  #[100,196,64]
  var12386=tf.broadcast_to(tf.reshape(var12385, [1,196,64]), [100,196,64])
  #[100,196,64]
  var12387=tf.add(var12383, var12386)
  #[100,14,14,64]
  var12388=tf.reshape(var12387, [100,14,14,64])
  #[100,14,14,64]
  var12389=tf.nn.relu(var12388)
  #[100,14,14,64]
  var12390=tf.reshape(var12389, [100,14,14,64])
  #[100,7,7,64]
  var12391=tf.nn.pool(var12390, [2,2], "MAX", strides=[2,2], padding="SAME")
  #[100,3136]
  var12392=tf.reshape(var12391, [100,3136])
  #[3136,1024]
  var12393=w1_w
  #[100,1024]
  var12394=tf.matmul(var12392, var12393)
  #[100,1024]
  var12395=tf.reshape(var12394, [100,1024])
  #[1024]
  var12396=w1_bias
  #[100,1024]
  var12397=tf.broadcast_to(tf.reshape(var12396, [1,1024]), [100,1024])
  #[100,1024]
  var12398=tf.add(var12395, var12397)
  #[100,1024]
  var12399=tf.nn.relu(var12398)
  #[100,1024]
  var12400=tf.reshape(var12399, [100,1024])
  #[1024,10]
  var12401=w2_w
  #[100,10]
  var12402=tf.matmul(var12400, var12401)
  #[100,10]
  var12403=tf.reshape(var12402, [100,10])
  #[10]
  var12404=w2_bias
  #[100,10]
  var12405=tf.broadcast_to(tf.reshape(var12404, [1,10]), [100,10])
  #[100,10]
  var12406=tf.add(var12403, var12405)
  #[100]
  var12407=tf.nn.softmax_cross_entropy_with_logits(labels=var12366, logits=var12406)
  #[100]
  var12408=tf.reshape(var12407, [100])
  #[]
  var12409=tf.reduce_sum(var12408, axis=0)
  #[]
  var12410=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var12411=tf.broadcast_to(tf.reshape(var12410, [1]), [1])
  #[]
  var12412=tf.reshape(var12411, [])
  #[]
  var12413=tf.add(var12409, var12412)
  #[100,10]
  var12414=tf.reshape(var12406, [100,10])
  #[100,10]
  var12415=tf.nn.softmax(var12414, axis=1)
  #[100,10]
  var12416=tf.reshape(var12415, [100,10])
  #[100]
  var12417=tf.argmax(var12406, axis=1, output_type=tf.int32)
  #[100]
  var12418=tf.argmax(var12366, axis=1, output_type=tf.int32)
  #[100]
  var12419=tf.equal(var12417, var12418)
  #[100]
  var12420=tf.cast(var12419, tf.float32)
  #[100]
  var12421=tf.reshape(var12420, [100])
  #[]
  var12422=tf.reduce_sum(var12421, axis=0)
  return {"loss":var12413,"accuracy":var12422,"y_":var12416}
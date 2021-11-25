import tensorflow as tf
def mkModel():
  #shape: [25, 32]
  var10000=tf.random.uniform([25, 32],
                             minval=-0.32444283,
                             maxval=0.32444283,
                             dtype=tf.float32) # 0
  #shape: [5, 5, 1, 32]
  var10001=tf.reshape(var10000, [5, 5, 1, 32])
  var10002=tf.Variable(name="f1_filters", trainable=True, initial_value=var10001)
  #shape: []
  var10003=tf.constant(0.1, shape=[], dtype=tf.float32)
  #shape: [32]
  var10004=ERROR:BroadcastT(var10003)
  #shape: [32]
  var10005=tf.reshape(var10004, [32])
  var10006=tf.Variable(name="f1_biases", trainable=True, initial_value=var10005)
  #shape: [800, 64]
  var10007=tf.random.uniform([800, 64],
                             minval=-8.3333336e-2,
                             maxval=8.3333336e-2,
                             dtype=tf.float32) # 1
  #shape: [5, 5, 32, 64]
  var10008=tf.reshape(var10007, [5, 5, 32, 64])
  var10009=tf.Variable(name="f2_filters", trainable=True, initial_value=var10008)
  #shape: [64]
  var10010=tf.reshape(var10004, [64])
  var10011=tf.Variable(name="f2_biases", trainable=True, initial_value=var10010)
  #shape: [3136, 1024]
  var10012=tf.random.uniform([3136, 1024],
                             minval=-3.7977725e-2,
                             maxval=3.7977725e-2,
                             dtype=tf.float32) # 2
  var10013=tf.Variable(name="w1_w", trainable=True, initial_value=var10012)
  #shape: [1024]
  var10014=tf.random.truncated_normal([1024], stddev=0.1, dtype=tf.float32) # 3
  var10015=tf.Variable(name="w1_bias", trainable=True, initial_value=var10014)
  #shape: [1024, 10]
  var10016=tf.random.uniform([1024, 10],
                             minval=-7.61755e-2,
                             maxval=7.61755e-2,
                             dtype=tf.float32) # 4
  var10017=tf.Variable(name="w2_w", trainable=True, initial_value=var10016)
  #shape: [10]
  var10018=tf.random.truncated_normal([10], stddev=0.1, dtype=tf.float32) # 5
  var10019=tf.Variable(name="w2_bias", trainable=True, initial_value=var10018)
  return {"batch_size":100,
          "parameters":[ var10002
          , var10006
          , var10009
          , var10011
          , var10013
          , var10015
          , var10017
          , var10019 ],
          "paramsdict":{"f1_filters":var10002,
                        "f1_biases":var10006,
                        "f2_filters":var10009,
                        "f2_biases":var10011,
                        "w1_w":var10013,
                        "w1_bias":var10015,
                        "w2_w":var10017,
                        "w2_bias":var10019}}
@tf.function
def runModel_fn(training_placeholder,
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
  #shape: [100, 10]
  var10020=y
  #shape: [100, 784]
  var10021=x
  #shape: [100, 28, 28, 1]
  var10022=tf.reshape(var10021, [100, 28, 28, 1])
  #shape: [5, 5, 1, 32]
  var10023=f1_filters
  #shape: [100, 28, 28, 32]
  var10024=tf.nn.convolution(var10022, var10023, padding="SAME", data_format="NHWC")
  #shape: [100, 784, 32]
  var10025=tf.reshape(var10024, [100, 784, 32])
  #shape: [32]
  var10026=f1_biases
  #shape: [784, 32]
  var10027=tf.broadcast_to(tf.reshape(var10026, [1, 32]), [784, 32])
  #shape: [100, 784, 32]
  var10028=tf.broadcast_to(tf.reshape(var10027, [1, 784, 32]), [100, 784, 32])
  #shape: [100, 784, 32]
  var10029=tf.add(var10025, var10028)
  #shape: [100, 28, 28, 32]
  var10030=tf.reshape(var10029, [100, 28, 28, 32])
  #shape: [100, 28, 28, 32]
  var10031=tf.nn.relu(var10030)
  #shape: [100, 28, 28, 32]
  var10032=tf.reshape(var10031, [100, 28, 28, 32])
  #shape: [100, 14, 14, 32]
  var10033=tf.nn.pool(var10032, [2, 2], "MAX", strides=[2, 2], padding="SAME")
  #shape: [100, 14, 14, 32]
  var10034=tf.reshape(var10033, [100, 14, 14, 32])
  #shape: [5, 5, 32, 64]
  var10035=f2_filters
  #shape: [100, 14, 14, 64]
  var10036=tf.nn.convolution(var10034, var10035, padding="SAME", data_format="NHWC")
  #shape: [100, 196, 64]
  var10037=tf.reshape(var10036, [100, 196, 64])
  #shape: [64]
  var10038=f2_biases
  #shape: [196, 64]
  var10039=tf.broadcast_to(tf.reshape(var10038, [1, 64]), [196, 64])
  #shape: [100, 196, 64]
  var10040=tf.broadcast_to(tf.reshape(var10039, [1, 196, 64]), [100, 196, 64])
  #shape: [100, 196, 64]
  var10041=tf.add(var10037, var10040)
  #shape: [100, 14, 14, 64]
  var10042=tf.reshape(var10041, [100, 14, 14, 64])
  #shape: [100, 14, 14, 64]
  var10043=tf.nn.relu(var10042)
  #shape: [100, 14, 14, 64]
  var10044=tf.reshape(var10043, [100, 14, 14, 64])
  #shape: [100, 7, 7, 64]
  var10045=tf.nn.pool(var10044, [2, 2], "MAX", strides=[2, 2], padding="SAME")
  #shape: [100, 3136]
  var10046=tf.reshape(var10045, [100, 3136])
  #shape: [3136, 1024]
  var10047=w1_w
  #shape: [100, 1024]
  var10048=tf.matmul(var10046, var10047)
  #shape: [100, 1024]
  var10049=tf.reshape(var10048, [100, 1024])
  #shape: [1024]
  var10050=w1_bias
  #shape: [100, 1024]
  var10051=tf.broadcast_to(tf.reshape(var10050, [1, 1024]), [100, 1024])
  #shape: [100, 1024]
  var10052=tf.add(var10049, var10051)
  #shape: [100, 1024]
  var10053=tf.nn.relu(var10052)
  #shape: [100, 1024]
  var10054=tf.reshape(var10053, [100, 1024])
  #shape: [1024, 10]
  var10055=w2_w
  #shape: [100, 10]
  var10056=tf.matmul(var10054, var10055)
  #shape: [100, 10]
  var10057=tf.reshape(var10056, [100, 10])
  #shape: [10]
  var10058=w2_bias
  #shape: [100, 10]
  var10059=tf.broadcast_to(tf.reshape(var10058, [1, 10]), [100, 10])
  #shape: [100, 10]
  var10060=tf.add(var10057, var10059)
  #shape: [100]
  var10061=tf.nn.softmax_cross_entropy_with_logits(labels=var10020, logits=var10060)
  #shape: [100]
  var10062=tf.reshape(var10061, [100])
  #shape: []
  var10063=tf.reduce_mean(var10062, axis=0)
  #shape: []
  var10064=tf.constant(0.0, shape=[], dtype=tf.float32)
  #shape: [1]
  var10065=tf.broadcast_to(tf.reshape(var10064, [1]), [1])
  #shape: []
  var10066=tf.reshape(var10065, [])
  #shape: []
  var10067=tf.add(var10063, var10066)
  #shape: [100]
  var10068=tf.argmax(var10060, axis=1, output_type=tf.int32)
  #shape: [100]
  var10069=tf.argmax(var10020, axis=1, output_type=tf.int32)
  #shape: [100]
  var10070=tf.equal(var10068, var10069)
  #shape: [100]
  var10071=tf.cast(var10070, tf.float32)
  #shape: [100]
  var10072=tf.reshape(var10071, [100])
  #shape: []
  var10073=tf.reduce_mean(var10072, axis=0)
  #shape: [100, 10]
  var10074=tf.reshape(var10060, [100, 10])
  #shape: [100, 10]
  var10075=tf.nn.softmax(var10074, axis=1)
  #shape: [100, 10]
  var10076=tf.reshape(var10075, [100, 10])
  return {"loss":var10067, "accuracy":var10073, "y_":var10076}
runModel = {"function":runModel_fn,
            "batched":True,
            "placeholders":{"x":{"shape":[100, 784], "dtype":tf.float32},
                            "y":{"shape":[100, 10], "dtype":tf.float32}}}
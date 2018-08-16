
import tensorflow as tf
def mkModel(optimizer=tf.train.AdamOptimizer()):
  
  training_phase=tf.placeholder(tf.bool, shape=[], name="training_phase")
  x=tf.placeholder(tf.float32, shape=[100,784], name="x")
  y=tf.placeholder(tf.float32, shape=[100,10], name="y")
  #[25,32]
  var1=tf.random_uniform(
         [25,32], minval=-0.32444283, maxval=0.32444283, dtype=tf.float32) # 0
  #[5,5,1,32]
  var2=tf.reshape(var1, [5,5,1,32])
  var3=tf.Variable(var2, name="f1_filters", trainable=True)
  #[32]
  var4=tf.constant(0.1, shape=[32], dtype=tf.float32)
  var5=tf.Variable(var4, name="f1_biases", trainable=True)
  #[800,64]
  var7=tf.random_uniform(
         [800,64], minval=-8.3333336e-2, maxval=8.3333336e-2, dtype=tf.float32) # 6
  #[5,5,32,64]
  var8=tf.reshape(var7, [5,5,32,64])
  var9=tf.Variable(var8, name="f2_filters", trainable=True)
  #[64]
  var10=tf.constant(0.1, shape=[64], dtype=tf.float32)
  var11=tf.Variable(var10, name="f2_biases", trainable=True)
  #[3136,1024]
  var14=tf.random_uniform(
          [3136,1024], minval=-3.7977725e-2, maxval=3.7977725e-2, dtype=tf.float32) # 12
  var15=tf.Variable(var14, name="w1_w", trainable=True)
  #[1024]
  var16=tf.truncated_normal([1024], stddev=0.1, dtype=tf.float32) # 13
  var17=tf.Variable(var16, name="w1_bias", trainable=True)
  #[1024,10]
  var20=tf.random_uniform(
          [1024,10], minval=-7.61755e-2, maxval=7.61755e-2, dtype=tf.float32) # 18
  var21=tf.Variable(var20, name="w2_w", trainable=True)
  #[10]
  var22=tf.truncated_normal([10], stddev=0.1, dtype=tf.float32) # 19
  var23=tf.Variable(var22, name="w2_bias", trainable=True)
  #[100,10]
  var24=y
  #[100,784]
  var25=x
  #[100,28,28,1]
  var26=tf.reshape(var25, [100,28,28,1])
  #[5,5,1,32]
  var27=var3
  #[100,28,28,32]
  var28=tf.nn.convolution(var26, var27, padding="SAME", data_format="NHWC")
  #[100,784,32]
  var29=tf.reshape(var28, [100,784,32])
  #[32]
  var30=var5
  #[784,32]
  var31=tf.add(tf.reshape(var30, [1,32]), tf.zeros([784,32], dtype=tf.float32))
  #[100,784,32]
  var32=tf.add(tf.reshape(var31, [1,784,32]), tf.zeros([100,784,32], dtype=tf.float32))
  #[100,784,32]
  var33=tf.add(var29, var32)
  #[100,28,28,32]
  var34=tf.reshape(var33, [100,28,28,32])
  #[100,28,28,32]
  var35=tf.nn.relu(var34)
  #[100,28,28,32]
  var36=tf.reshape(var35, [100,28,28,32])
  #[100,14,14,32]
  var37=tf.nn.pool(var36, [2,2], "MAX", "SAME", strides=[2,2])
  #[100,14,14,32]
  var38=tf.reshape(var37, [100,14,14,32])
  #[5,5,32,64]
  var39=var9
  #[100,14,14,64]
  var40=tf.nn.convolution(var38, var39, padding="SAME", data_format="NHWC")
  #[100,196,64]
  var41=tf.reshape(var40, [100,196,64])
  #[64]
  var42=var11
  #[196,64]
  var43=tf.add(tf.reshape(var42, [1,64]), tf.zeros([196,64], dtype=tf.float32))
  #[100,196,64]
  var44=tf.add(tf.reshape(var43, [1,196,64]), tf.zeros([100,196,64], dtype=tf.float32))
  #[100,196,64]
  var45=tf.add(var41, var44)
  #[100,14,14,64]
  var46=tf.reshape(var45, [100,14,14,64])
  #[100,14,14,64]
  var47=tf.nn.relu(var46)
  #[100,14,14,64]
  var48=tf.reshape(var47, [100,14,14,64])
  #[100,7,7,64]
  var49=tf.nn.pool(var48, [2,2], "MAX", "SAME", strides=[2,2])
  #[100,3136]
  var50=tf.reshape(var49, [100,3136])
  #[3136,1024]
  var51=var15
  #[100,1024]
  var52=tf.matmul(var50, var51)
  #[100,1024]
  var53=tf.reshape(var52, [100,1024])
  #[1024]
  var54=var17
  #[100,1024]
  var55=tf.add(tf.reshape(var54, [1,1024]), tf.zeros([100,1024], dtype=tf.float32))
  #[100,1024]
  var56=tf.add(var53, var55)
  #[100,1024]
  var57=tf.nn.relu(var56)
  #[100,1024]
  var58=tf.reshape(var57, [100,1024])
  #[1024,10]
  var59=var21
  #[100,10]
  var60=tf.matmul(var58, var59)
  #[100,10]
  var61=tf.reshape(var60, [100,10])
  #[10]
  var62=var23
  #[100,10]
  var63=tf.add(tf.reshape(var62, [1,10]), tf.zeros([100,10], dtype=tf.float32))
  #[100,10]
  var64=tf.add(var61, var63)
  #[100]
  var65=tf.nn.softmax_cross_entropy_with_logits(labels=var24, logits=var64)
  #[100]
  var66=tf.reshape(var65, [100])
  #[]
  var67=tf.reduce_mean(var66, axis=0)
  #[]
  var68=tf.zeros([], dtype=tf.float32)
  #[]
  var69=tf.add(var67, var68)
  var70=optimizer.minimize(var69)
  #[100]
  var71=tf.argmax(var64, axis=1, output_type=tf.int32)
  #[100]
  var72=tf.argmax(var24, axis=1, output_type=tf.int32)
  #[100]
  var73=tf.equal(var71, var72)
  #[100]
  var74=tf.cast(var73, tf.float32)
  #[100]
  var75=tf.cast(var74, tf.float32)
  #[100]
  var76=tf.reshape(var75, [100])
  #[]
  var77=tf.reduce_mean(var76, axis=0)
  #[100,10]
  var78=tf.nn.softmax(var64, dim=1)
  #[]
  var79=training_phase
  return {"accuracy":var77
         ,"y_":var78
         ,"w2_bias":var62
         ,"w2_w":var59
         ,"w1_bias":var54
         ,"w1_w":var51
         ,"f2_biases":var42
         ,"f2_filters":var39
         ,"f1_biases":var30
         ,"f1_filters":var27
         ,"y":var24
         ,"x":var25
         ,"training_phase":var79
         ,"optimizer":optimizer
         ,"batch_size":100
         ,"params":tf.trainable_variables()
         ,"train":var70
         ,"update":[]}
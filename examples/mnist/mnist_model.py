
import tensorflow as tf
def mkModel(optimizer=tf.train.AdamOptimizer()):
  
  var0=tf.placeholder(tf.bool, shape=[], name="training_phase")
  var1=tf.placeholder(tf.float32, shape=[100,784], name="x")
  var2=tf.placeholder(tf.float32, shape=[100,10], name="y")
  #[25,32]
  var4=tf.random_uniform(
         [25,32], minval=-0.32444283, maxval=0.32444283, dtype=tf.float32) # 3
  #[5,5,1,32]
  var5=tf.reshape(var4, [5,5,1,32])
  var6=tf.Variable(var5, name="f1_filters", trainable=True)
  #[]
  var7=tf.constant(0.1, shape=[], dtype=tf.float32)
  #[32]
  var8=tf.broadcast_to(tf.reshape(var7, [1]), [32])
  #[32]
  var9=tf.reshape(var8, [32])
  var10=tf.Variable(var9, name="f1_biases", trainable=True)
  #[800,64]
  var12=tf.random_uniform(
          [800,64], minval=-8.3333336e-2, maxval=8.3333336e-2, dtype=tf.float32) # 11
  #[5,5,32,64]
  var13=tf.reshape(var12, [5,5,32,64])
  var14=tf.Variable(var13, name="f2_filters", trainable=True)
  #[64]
  var15=tf.broadcast_to(tf.reshape(var7, [1]), [64])
  #[64]
  var16=tf.reshape(var15, [64])
  var17=tf.Variable(var16, name="f2_biases", trainable=True)
  #[3136,1024]
  var20=tf.random_uniform(
          [3136,1024], minval=-3.7977725e-2, maxval=3.7977725e-2, dtype=tf.float32) # 18
  var21=tf.Variable(var20, name="w1_w", trainable=True)
  #[1024]
  var22=tf.truncated_normal([1024], stddev=0.1, dtype=tf.float32) # 19
  var23=tf.Variable(var22, name="w1_bias", trainable=True)
  #[1024,10]
  var26=tf.random_uniform(
          [1024,10], minval=-7.61755e-2, maxval=7.61755e-2, dtype=tf.float32) # 24
  var27=tf.Variable(var26, name="w2_w", trainable=True)
  #[10]
  var28=tf.truncated_normal([10], stddev=0.1, dtype=tf.float32) # 25
  var29=tf.Variable(var28, name="w2_bias", trainable=True)
  #[100,10]
  var30=var2
  #[100,784]
  var31=var1
  #[100,28,28,1]
  var32=tf.reshape(var31, [100,28,28,1])
  #[5,5,1,32]
  var33=var6
  #[100,28,28,32]
  var34=tf.nn.convolution(var32, var33, padding="SAME", data_format="NHWC")
  #[100,784,32]
  var35=tf.reshape(var34, [100,784,32])
  #[32]
  var36=var10
  #[784,32]
  var37=tf.broadcast_to(tf.reshape(var36, [1,32]), [784,32])
  #[100,784,32]
  var38=tf.broadcast_to(tf.reshape(var37, [1,784,32]), [100,784,32])
  #[100,784,32]
  var39=tf.add(var35, var38)
  #[100,28,28,32]
  var40=tf.reshape(var39, [100,28,28,32])
  #[100,28,28,32]
  var41=tf.nn.relu(var40)
  #[100,28,28,32]
  var42=tf.reshape(var41, [100,28,28,32])
  #[100,14,14,32]
  var43=tf.nn.pool(var42, [2,2], "MAX", "SAME", strides=[2,2])
  #[100,14,14,32]
  var44=tf.reshape(var43, [100,14,14,32])
  #[5,5,32,64]
  var45=var14
  #[100,14,14,64]
  var46=tf.nn.convolution(var44, var45, padding="SAME", data_format="NHWC")
  #[100,196,64]
  var47=tf.reshape(var46, [100,196,64])
  #[64]
  var48=var17
  #[196,64]
  var49=tf.broadcast_to(tf.reshape(var48, [1,64]), [196,64])
  #[100,196,64]
  var50=tf.broadcast_to(tf.reshape(var49, [1,196,64]), [100,196,64])
  #[100,196,64]
  var51=tf.add(var47, var50)
  #[100,14,14,64]
  var52=tf.reshape(var51, [100,14,14,64])
  #[100,14,14,64]
  var53=tf.nn.relu(var52)
  #[100,14,14,64]
  var54=tf.reshape(var53, [100,14,14,64])
  #[100,7,7,64]
  var55=tf.nn.pool(var54, [2,2], "MAX", "SAME", strides=[2,2])
  #[100,3136]
  var56=tf.reshape(var55, [100,3136])
  #[3136,1024]
  var57=var21
  #[100,1024]
  var58=tf.matmul(var56, var57)
  #[100,1024]
  var59=tf.reshape(var58, [100,1024])
  #[1024]
  var60=var23
  #[100,1024]
  var61=tf.broadcast_to(tf.reshape(var60, [1,1024]), [100,1024])
  #[100,1024]
  var62=tf.add(var59, var61)
  #[100,1024]
  var63=tf.nn.relu(var62)
  #[100,1024]
  var64=tf.reshape(var63, [100,1024])
  #[1024,10]
  var65=var27
  #[100,10]
  var66=tf.matmul(var64, var65)
  #[100,10]
  var67=tf.reshape(var66, [100,10])
  #[10]
  var68=var29
  #[100,10]
  var69=tf.broadcast_to(tf.reshape(var68, [1,10]), [100,10])
  #[100,10]
  var70=tf.add(var67, var69)
  #[100]
  var71=tf.nn.softmax_cross_entropy_with_logits(labels=var30, logits=var70)
  #[100]
  var72=tf.reshape(var71, [100])
  #[]
  var73=tf.reduce_mean(var72, axis=0)
  #[]
  var74=tf.constant(0.0, shape=[], dtype=tf.float32)
  #[1]
  var75=tf.broadcast_to(tf.reshape(var74, [1]), [1])
  #[]
  var76=tf.reshape(var75, [])
  #[]
  var77=tf.add(var73, var76)
  var78=optimizer.minimize(var77)
  #[100]
  var79=tf.argmax(var70, axis=1, output_type=tf.int32)
  #[100]
  var80=tf.argmax(var30, axis=1, output_type=tf.int32)
  #[100]
  var81=tf.equal(var79, var80)
  #[100]
  var82=tf.cast(var81, tf.float32)
  #[100]
  var83=tf.cast(var82, tf.float32)
  #[100]
  var84=tf.reshape(var83, [100])
  #[]
  var85=tf.reduce_mean(var84, axis=0)
  #[100,10]
  var86=tf.reshape(var70, [100,10])
  #[100,10]
  var87=tf.nn.softmax(var86, axis=1)
  #[100,10]
  var88=tf.reshape(var87, [100,10])
  #[]
  var89=var0
  return {"batch_size":100
         ,"accuracy":var85
         ,"y_":var88
         ,"w2_bias":var68
         ,"w2_w":var65
         ,"w1_bias":var60
         ,"w1_w":var57
         ,"f2_biases":var48
         ,"f2_filters":var45
         ,"f1_biases":var36
         ,"f1_filters":var33
         ,"y":var30
         ,"x":var31
         ,"training_phase":var89
         ,"optimizer":optimizer
         ,"params":tf.trainable_variables()
         ,"train":var78
         ,"loss":var77
         ,"update":[]}

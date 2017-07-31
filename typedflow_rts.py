import tensorflow as tf


# optimize is one of tf.train.GradientDescentOptimizer(0.05), etc.
def train (sess, model, optimizer, train_generator, valid_generator, epochs):
    (x,y,y_,accuracy,loss) = model
    train = optimizer.minimize(loss) # must come before the initializer (this line creates variables!)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        totalAccur = 0
        totalLoss = 0
        n = 0
        for (x_train,y_train) in train_generator():
            _,lossAcc,accur = sess.run([train,loss,accuracy], feed_dict={x:x_train, y:y_train})
            n+=1
            totalLoss += lossAcc
            totalAccur += accur
        print ("Training Loss = ", totalLoss / float(n), " Training accuracy = ", totalAccur / float(n))

        totalLoss = 0
        totalAccur = 0
        n = 0
        for (x_train,y_train) in valid_generator():
            lossAcc,accur = sess.run([loss,accuracy], feed_dict={x:x_train, y:y_train})
            totalLoss += lossAcc
            totalAccur += totalAccur
            n+=1
        if n > 0:
            print ("Validation Loss = ", totalLoss / float(n), " Validation accuracy = ", totalAccur / float(n))
        else:
            print ("No validation data.")


def predict (sess, model, x_generator):
    (x,y,y_,accuracy,loss) = model
    sess.run(init)
    for (n,i) in enumerate(x_generator()):
        sess.run(y, feed_dict={x:x_generator})

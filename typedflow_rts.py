import tensorflow as tf


# optimize is one of tf.train.GradientDescentOptimizer(0.05), etc.
def train (sess, model, optimizer, train_generator, valid_generator, epochs):
    (x,y,y_,accuracy,loss,params,gradients) = model
    # must come before the initializer (this line creates variables!)
    train = optimizer.apply_gradients(zip(gradients, params))
    # train = optimizer.minimize(loss)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        totalAccur = 0
        totalLoss = 0
        n = 0
        for (x_train,y_train) in train_generator():
            print(".",end="")
            _,lossAcc,accur = sess.run([train,loss,accuracy], feed_dict={x:x_train, y:y_train})
            n+=1
            totalLoss += lossAcc
            totalAccur += accur
        print(".")
        print ("Training Loss = ", totalLoss / float(n), " Training accuracy = ", totalAccur / float(n))

        totalLoss = 0
        totalAccur = 0
        n = 0
        for (x_train,y_train) in valid_generator():
            print(".",end="")
            lossAcc,accur = sess.run([loss,accuracy], feed_dict={x:x_train, y:y_train})
            totalLoss += lossAcc
            totalAccur += accur
            n+=1
        print(".")
        if n > 0:
            print ("Validation Loss = ", totalLoss / float(n), " Validation accuracy = ", totalAccur / float(n))
        else:
            print ("No validation data.")


def predict (sess, model, x_generator):
    (x,y,y_,accuracy,loss,params,gradients) = model
    sess.run(init)
    for (n,i) in enumerate(x_generator()):
        sess.run(y, feed_dict={x:x_generator})



    # k-Beam search at index i in a sequence.
    # work with k-size batch.
    # keep for every j < k a sum of the log probs, r(j).
    # for every possible output work w at the i-th step in the sequence, compute r'(j,w) = r(j) * logit(i,w)
    # compute the k pairs (j,w) which minimize r'(j,w). Let M this set.
    # r(l) = r'(j,w) for (l,(j,w)),in enumarate(M)
    
    
    # # beam search
    # def translate(src_sent, k=1):
    #     # (log(1), initialize_of_zeros)
    #     k_beam = [(0, [0]*(sequence_max_len+1))]
    
    #     # l : point on target sentence to predict
    #     for l in range(sequence_max_len):
    #         all_k_beams = []
    #         for prob, trg_sent_predict in k_beam:
    #             predicted       = encoder_decoder.predict([np.array([src_sent]), np.array([trg_sent_predict])])[0]
    #             # top k!
    #             possible_k_trgs = predicted[l].argsort()[-k:][::-1]
    
    #             # add to all possible candidates for k-beams
    #             all_k_beams += [
    #                 (
    #                     sum(np.log(predicted[i][trg_sent_predict[i+1]]) for i in range(l)) + np.log(predicted[l][next_wid]),
    #                     list(trg_sent_predict[:l+1])+[next_wid]+[0]*(sequence_max_len-l-1)
    #                 )
    #                 for next_wid in possible_k_trgs
    #             ]
    #         # top k
    #         k_beam = sorted(all_k_beams)[-k:]

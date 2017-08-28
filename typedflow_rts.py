import tensorflow as tf
import sys
from time import time

# optimizer is one of tf.train.GradientDescentOptimizer(0.05), tf.train.AdamOptimizer() etc.
def train (sess, model, optimizer, train_generator, valid_generator, epochs, callbacks=[]):
    (training_phase,x,y,y_,accuracy,loss,params,gradients) = model
    # must come before the initializer (this line creates variables!)
    train = optimizer.apply_gradients(zip(gradients, params))
    # train = optimizer.minimize(loss)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    def halfEpoch(isTraining):
        totalAccur = 0
        totalLoss = 0
        n = 0
        print ("Training" if isTraining else "Validation", end="")
        start_time = time()
        for (x_train,y_train) in train_generator() if isTraining else valid_generator():
            print(".",end="")
            sys.stdout.flush()
            _,lossAcc,accur = sess.run([train,loss,accuracy], feed_dict={x:x_train, y:y_train, training_phase:isTraining})
            n+=1
            totalLoss += lossAcc
            totalAccur += accur
        end_time = time()
        if n > 0:
            avgLoss = totalLoss / float(n)
            avgAccur = totalAccur / float(n)
            print(".")
            print ("Time=%.1f" % (end_time - start_time), "loss=%g" % avgLoss, "accuracy=%.3f" % avgAccur)
            return (avgLoss,avgAccur)
        else:
            print ("No data")
            return (0,0)

    for e in range(epochs):
        (trainLoss,trainAccur) = halfEpoch(True)
        (valLoss,valAccur) = halfEpoch(False)
        if any(c({"train_loss":trainLoss,
                  "train_accur":trainAccur,
                  "val_loss":valLoss,
                  "val_accur":valAccur,
                  "time":end_time - start_time
                  }) for c in callbacks):
            break

def earlyStopping(patience = 1):
    oldLoss = 10000000000
    p = patience
    def callback(values):
        nonlocal oldLoss, p
        newLoss = values["val_loss"]
        if newLoss > oldLoss:
            p -= 1
        if p <= 0:
            return True
        oldLoss = newLoss
        return False
    return callback

def predict (sess, model, x_generator):
    (training_phase,x,y,y_,accuracy,loss,params,gradients) = model
    return [sess.run(y, feed_dict={x:x_train, training_phase:False}) (x_train,i) in enumerate(x_generator())]

# Given a pair of x and y (each being a list or a np array) and a
# batch size, return a generator function which will yield the input
# in bs-sized chunks. Attention: if the size of the input is not
# divisible by bs, then the remainer will not be fed. Consider
# shuffling the input.
def bilist_generator(l,bs):
    (l0,l1) = l
    def gen():
      for i in range(0, bs*(len(l0)//bs), bs):
        yield (l0[i:i+bs],l1[i:i+bs])
    return gen

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

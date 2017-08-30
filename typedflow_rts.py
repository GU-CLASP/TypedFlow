import tensorflow as tf
import numpy as np
import sys
from time import time

def cuda_pref_device(n):
    os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(n)


# Given a pair of x and y (each being a list or a np array) and a
# batch size, return a generator function which will yield the input
# in bs-sized chunks. Attention: if the size of the input is not
# divisible by bs, then the remainer will not be fed. Consider
# shuffling the input.
def bilist_generator(l):
    (l0,l1) = l
    def gen(bs):
      for i in range(0, bs*(len(l0)//bs), bs):
        yield (l0[i:i+bs],l1[i:i+bs])
    return gen

def initialize_params (session,model):
    # it'd be nice to do:

    # session.run(tf.variables_initializer(model["params"]))

    # However this does not initialize the optimizer's variables. So,
    # instead we do:

    session.run(tf.local_variables_initializer())
    session.run(tf.global_variables_initializer())

def train (session, model, train_generator,
           valid_generator=bilist_generator(([],[])),
           epochs=100,
           callbacks=[]):
    batch_size = model["batch_size"]
    stats = []
    def halfEpoch(isTraining):
        totalAccur = 0
        totalLoss = 0
        n = 0
        print ("Training" if isTraining else "Validation", end="")
        start_time = time()
        for (x_train,y_train) in train_generator(batch_size) if isTraining else valid_generator(batch_size):
            print(".",end="")
            sys.stdout.flush()
            _,loss,accur = session.run([model["train"],model["loss"],model["accuracy"]],
                                       feed_dict={model["x"]:x_train,
                                                  model["y"]:y_train,
                                                  model["training_phase"]:isTraining})
            n+=1
            totalLoss += loss
            totalAccur += accur
        end_time = time()
        if n > 0:
            avgLoss = totalLoss / float(n)
            avgAccur = totalAccur / float(n)
            print(".")
            print ("Time=%.1f" % (end_time - start_time), "loss=%g" % avgLoss, "accuracy=%.3f" % avgAccur)
            return {"loss":avgLoss,"accuracy":avgAccur,"time":(end_time - start_time),"error_rate":1-avgAccur,"start_time":start_time}
        else:
            print ("No data")
            return {"loss":0,"accur":0,"time":0,"error_rate":0,"start_time":0}

    for e in range(epochs):
        print ("Epoch {0}/{1}".format(e, epochs))
        tr = halfEpoch(True)
        va = halfEpoch(False)
        epoch_stats = {"train":tr, "val":va, "epoch":e}
        stats.append(epoch_stats)
        if any(c(epoch_stats) for c in callbacks):
            break
    return stats

def StopWhenValidationGetsWorse(patience = 1):
    bestLoss = 10000000000
    p = patience
    def callback(values):
        nonlocal bestLoss, p, patience
        newLoss = values["val"]["loss"]
        if newLoss > bestLoss:
            p -= 1
        else:
            bestLoss = newLoss
            p = patience
        if p <= 0:
            return True
        return False
    return callback

def StopWhenAccurate(error_rate = .01):
    def callback(values):
        nonlocal error_rate
        return values["val"]["error_rate"] < error_rate
    return callback


def predict (session, model, xs, result="y_"):
    bs = model["batch_size"]
    zeros = np.zeros_like(xs[0]) # at least one example is needed
    results = []
    def run():
        for i in range(0, bs*(-(-len(xs)//bs)), bs):
            chunk = xs[i:i+bs]
            if i + bs > len(xs):
                origLen = len(chunk)
                chunk = list(chunk) + [zeros] * (bs - origLen) # pad the last chunk
            else:
                origLen = bs
            print (".")
            yield (session.run(model[result], feed_dict={model["x"]:chunk, model["training_phase"]:False}))[:origLen]
    return np.concatenate(list(run()))
    
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

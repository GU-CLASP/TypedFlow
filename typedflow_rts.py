import tensorflow as tf
import numpy as np
import sys
from time import time
import os

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
        yield {"x":l0[i:i+bs],"y":l1[i:i+bs]}


# Given a pair of l=(x,y) (both x,y being a list or a np array) and a
# batch size, return a generator function which will yield the input
# in bs*maxlen-sized chunks. This generator is intended to be used for
# stateful language models. That is, batch sequencing corresponds to 
def bilist_generator_transposed(model,l):
    (batch_size,maxlen) = model["x"].shape
    (xs,ys) = l
    num_items = len(xs) // (batch_size*maxlen)
    x = np.zeros(shape=(num_items,batch_size,maxlen))
    y = np.zeros(shape=(num_items,batch_size,maxlen))
    for i in range(num_items):
        for j in range(batch_size):
            for k in range(maxlen):
                x[i][j][k] = xs[k+j*(num_items*maxlen)+i*maxlen]
                y[i][j][k] = ys[k+j*(num_items*maxlen)+i*maxlen]
    def gen(_bs):
        nonlocal num_items, x, y
        for i in range(num_items):
            yield {"x":x[i],"y":y[i]}
    return gen

def dict_generator (xs):
    k0 = next (iter (xs.keys())) # at least one key is needed
    total_len = len(xs[k0])

    def gen(bs):
      for i in range(0, bs*(total_len//bs), bs):
        yield dict((k,xs[k][i:i+bs]) for k in xs)


def initialize_params (session,model):
    # it'd be nice to do:

    # session.run(tf.variables_initializer(model["params"]))

    # However this does not initialize the optimizer's variables. So,
    # instead we do:

    session.run(tf.local_variables_initializer())
    session.run(tf.global_variables_initializer())

def train (session, model,
           train_generator=bilist_generator(([],[])),
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
        for inputs in train_generator(batch_size) if isTraining else valid_generator(batch_size):
            print(".",end="")
            sys.stdout.flush()
            _,loss,accur = session.run([model["train"],model["loss"],model["accuracy"]],
                                       feed_dict=dict([(model["training_phase"],isTraining)] +
                                                      [(model[k],inputs[k]) for k in inputs]))
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


def s2s_generator(src_in,tgt_in,tgt_out,tgt_weights):
    def gen(bs):
      for i in range(0, bs*(len(src_in)//bs), bs):
        yield {"src_in":src_in[i:i+bs],
               "tgt_in":tgt_in[i:i+bs],
               "tgt_out":tgt_out[i:i+bs],
               "tgt_weights":tgt_weights[i:i+bs]}
    return gen

def predict (session, model, xs, result="y_"):
    bs = model["batch_size"]
    k0 = next (iter (xs.keys())) # at least one key is needed
    total_len = len(xs[k0])
    zeros = dict((k,np.zeros_like(xs[k][0])) for k in xs) # at least one example is needed
    results = []
    def run():
        for i in range(0, bs*(-(-total_len//bs)), bs):
            chunks = dict((k,xs[k][i:i+bs]) for k in xs)
            if i + bs > total_len:
                origLen = total_len - i
                for k in xs:
                    chunks[k] = list(chunks[k]) + [zeros[k]] * (bs - origLen) # pad the last chunk
            else:
                origLen = bs
            # print (".")
            yield (session.run(model[result],
                               dict([(model["training_phase"],False)] +
                                    [(model[k],chunks[k]) for k in xs])))[:origLen]
    return np.concatenate(list(run()))


def beam_translate(session, model, k, x, xlen, start_symbol):
    (_,voc_size,out_len) = model["y_"].shape
    xs = np.array ([x] * k) # The input is always the same
    xs_len = np.array ([xlen]*k) # this is very important to get right
    ys = [[start_symbol]]  # start with a single thing; otherwise the minimum will be repeated k times
    probs = [1]
    hist = [[]]

    def pad(z):
        return np.array(z + [0] * (out_len - len(z)))
    for i in range(out_len-1):
        print ("beam search at:", i)
        inputs = {"src_len":xs_len[:len(ys)], "src_in":xs[:len(ys)], "tgt_in":np.array([pad(y) for y in ys])}
        y_s = predict(session,model,inputs)
        all_words = sorted([(y_s[j][w][i] * probs[j], ys[j] + [w], hist[j] + [y_s[j][w][i]])
                            for j in range(len(y_s))
                            for w in range(voc_size)])
        best = all_words[-k:]
        # for (p,y,h) in best: print("Prob", p, decode(y),h)
        (probs,ys,hist) = zip(*best)

    return ys

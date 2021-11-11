import tensorflow as tf
import numpy as np
import sys
from time import time
import os
import random

###############################################################
# Devices
###############################################################


def cuda_use_device(n):
    """Attempt to use a given CUDA device by setting the appropriate environment variables"""
    os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(n)

def find_free_cuda_device():
    currentGPU = -1
    gpuMemory=dict()
    gpuUtil=dict()
    for line in os.popen("nvidia-smi -q"):
        fields = list(map(lambda x: x.strip(), line.split(":")))
        k = fields[0]
        if k == "Minor Number":
            currentGPU += 1
            gpuMemory[currentGPU] = 0
        elif k == "Used GPU Memory":
            gpuMemory[currentGPU] = int(fields[1][:-4]) # last characters are " MiB"
        elif k == "Gpu":
            gpuUtil[currentGPU] = fields[1] # last characters are " %"
    minUse = min(gpuMemory.values())
    freeGpus = [g for g in gpuMemory.keys() if gpuMemory[g] == minUse]
    if freeGpus == []:
        print("No free GPU could be found.")
        assert False
    else:
        result = random.choice(freeGpus)
        print ("Found device",result,"currently used at",gpuUtil[result],"and with",gpuMemory[result],"MB taken.")
        return result

def cuda_use_one_free_device():
    """Attempt to use a free CUDA device by setting the appropriate environment variables"""
    cuda_use_device(find_free_cuda_device())

###############################################################
# Generators
###############################################################

def bilist_generator(l):
    """
    Given a pair of x and y (each being a list or a np array) and a
    batch size, return a generator function which will yield the input
    in bs-sized chunks. Attention: if the size of the input is not
    divisible by bs, then the remainer will not be fed. Consider
    shuffling the input.
    """
    (l0,l1) = l
    def gen(bs):
        if len(l0) == 0:
            return
        for i in range(0, bs*(len(l0)//bs), bs):
            yield {"x":l0[i:i+bs],"y":l1[i:i+bs]}
    return gen


def bilist_generator_transposed(model,l):
    '''
    Given a pair of l=(x,y) (both x,y being a list or a np array) and a
    batch size, return a generator function which will yield the input
    in bs*maxlen-sized chunks. This generator is intended to be used for
    stateful language models. That is, batch sequencing corresponds to 
    '''
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

    return gen


def initialize_params (session,model):
    '''Initialize the learnable parameters of the model'''
    # it'd be nice to do:

    # session.run(tf.variables_initializer(model["params"]))

    # However this does not initialize the optimizer's variables. So,
    # instead we do:

    session.run(tf.local_variables_initializer())
    session.run(tf.global_variables_initializer())

def train (optimizer, model_static, model_fn,
           train_generator=bilist_generator(([],[])),
           valid_generator=bilist_generator(([],[])),
           epochs=100,
           callbacks=[],
           extraVectors=[]):
    '''
    Train the given model.

    train_generator: training data

    valid_generator: validation data

    epochs: number of epochs

    callbacks: list of callbacks.
      Each callback receives an epoch entry (see below). If it returns False then the training is aborted.

    extraVectors: list of extra vectors to pass to session.run when training.

    modelPrefix: in case of a multitask/multimodel, give the prefix of the model to use.

    This function returns a list of epoch entries. Each entry is a dictionary with:
     - "epoch": current epoch
     - "val" and "train": dictionaries with
        - "loss", "accuracy", "error_rate", time", "start_time", "end_time"
    '''
    batch_size = model_static["batch_size"]
    train_vars = model_static["parameters"]
    placeholders_info = model_fn["placeholders"]
    stats = []
    def halfEpoch(isTraining):
        totalAccur = 0
        totalLoss = 0
        n = 0
        print ("Training" if isTraining else "Validation", end="")
        start_time = time()
        for inputs in train_generator(batch_size) if isTraining else valid_generator(batch_size):
            cast_inputs = dict((k,tf.cast(inputs[k], placeholders_info[k]["dtype"])) for k in placeholders_info)
            # the above forces inputs to be tensors. (It's convenient to pass just lists here)
            print(".",end="")
            sys.stdout.flush()
            with tf.GradientTape() as tape:
                results = model_fn["function"](tf.constant(isTraining, shape=[]), **{**(model_static["paramsdict"]), **cast_inputs})
                loss = results["loss"]
                accur = results["accuracy"]
                grads = tape.gradient(loss, train_vars)
                optimizer.apply_gradients(zip(grads, train_vars))
                n+=1
                totalLoss += loss
                totalAccur += accur
        end_time = time()
        totalAccur = totalAccur.numpy()
        totalLoss = totalLoss.numpy()
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
    '''Return a callback which stops training if validation loss gets worse.'''
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

def StopWhenAccurate(phase="val",error_rate = .01):
    '''Return a callback which stops training if error rate drops below 1%'''
    def callback(values):
        nonlocal error_rate
        return values[phase]["error_rate"] < error_rate
    return callback

def Every(n,f):
    '''Return a callback which calls its argument every n epochs'''
    def callback(values):
        nonlocal n,f
        if values["epoch"] % n == (n-1):
            return f(values)
        else:
            return False
    return callback

def Save(sess,saver,ckptfile):
    def callback(values):
        nonlocal sess,saver
        print("Saving to",ckptfile)
        saver.save(sess, ckptfile)
        return False
    return callback

################################################################################################
# Prediction and evaluation


def evaluate (model_static, model_fn, xs, result="y_"):
    '''Evaluate the model for given input and result.
    Input is given as a dictionary of lists to pass to session.run'''
    phs = model_fn["placeholders"]
    if phs:
        k0 = next (iter (phs.keys())) # 1st placeholder
        total_len = len(xs[k0]) # total length
    else:
        total_len = 1
    zeros = dict((k,tf.zeros(phs[k]["shape"][1:], # remove the batch size
                             dtype=phs[k]["dtype"])) for k in phs.keys())
    results = []
    if model_fn["batched"]:
        def run():
          bs = model_static["batch_size"]
          for i in range(0, bs*(-(-total_len//bs)), bs):
              print(".",end="")
              chunks = dict()
              for k in phs:
                  chunks[k] = xs[k][i:i+bs]
              if i + bs > total_len:
                  # dealing with an incomplete last chunk
                  origLen = total_len - i
                  for k in chunks:
                      chunks[k] = list(chunks[k]) + [zeros[k]] * (bs - origLen)  # pad the last chunk
              else:
                  origLen = bs
              chunks = {k: tf.cast(v,dtype=phs[k]["dtype"]) for (k,v) in chunks.items()}
              results = model_fn["function"](tf.constant(False, shape=[]), **{**(model_static["paramsdict"]), **chunks}) 
              yield results[result][:origLen]
        return np.concatenate(list(run()))
    else:
        def run():
            for i in range(total_len):
                inputs = {k: tf.cast(xs[k][i], dtype=phs[k]["dtype"]) for k in phs}
                results = model_fn["function"](tf.constant(False, shape=[]), **{**(model_static["paramsdict"]), **inputs})
                yield results[result]
        return list(run())
        

predict = evaluate

def beam_translate(session, model, k, x, xlen, start_symbol, stop_symbol, debug=None):
    '''Beam translation of ONE input sentence.'''
    (_,out_len,voc_size) = model["y_"].shape
    xs = np.array ([x] * k) # The input is always the same
    xs_len = np.array ([xlen]*k) # it is VERY important to get the length right
    ys = [[start_symbol]]  # start with a single thing; otherwise the minimum will be repeated k times
    probs = [1]
    results = []
    hist = [[]]
    def pad(z):
        return np.array(z + [0] * (out_len - len(z)))
    for i in range(out_len-1):
        print ("beam search at:", i)
        inputs = {"src_len":xs_len[:len(ys)], "src_in":xs[:len(ys)], "tgt_in":np.array([pad(y) for y in ys])}
        y_s = predict(session,model,inputs)
        all_words = sorted([(y_s[j][i][w] * probs[j], ys[j] + [w], hist[j] + [y_s[j][i][w]])
                            for j in range(len(y_s))
                            for w in range(voc_size)])
        best = all_words[-k:]
        if debug is not None:
            for x in best: debug(x)
        results  += [(p,y,h) for (p,y,h) in best if y[i+1] == stop_symbol]
        continued = [(p,y,h) for (p,y,h) in best if y[i+1] != stop_symbol]
        if len(continued) == 0: break
        (probs,ys,hist) = zip(*continued)
    return sorted(results)

######################################################
# Saving and loading

def save(model_static, file):
  numpy_tensors = {k:v.numpy() for (k,v) in model_static["paramsdict"].items()}
  print("Saving parameters: ", model_static["paramsdict"].keys())
  np.savez(file,**numpy_tensors)
  print("Done")


def load(model_static, file):
  print("Loading parameters")
  numpy_tensors = np.load(file)
  print("Loaded parameters: ", list(numpy_tensors.keys()))
  for k,v in model_static["paramsdict"].items():
      v.assign(numpy_tensors[k])
  print("Done")

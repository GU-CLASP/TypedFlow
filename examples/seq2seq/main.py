import sys
sys.path.append('../..') # so we can see the rts.

import typedflow_rts as tyf
import tensorflow as tf
import numpy as np
from s2s import mkModel
import os
import math
import random

tyf.cuda_pref_device(2)

chars = sorted(list("()01234abcde^$ "))

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

MAXLEN = 22
def pad(ws): return (ws + ' '*(MAXLEN - len(ws)))

def encode(s):
    # print ("proun", s)
    return np.array([char_indices[c] for c in s])

def decode(s): return "".join([indices_char[c] for c in list(s)])

def pad_right(sentence): return (MAXLEN - len(sentence)) * " " + sentence
def pad_left(sentence): return  sentence + (MAXLEN - len(sentence)) * " "

def source_input_conversion(s):
    return encode(pad_left(s))

def target_input_conversion(sentence):
    return encode(pad_left("^"+sentence))

def target_output_conversion(sentence):
    return encode(pad_left(sentence+"$"))

def sentence_target_weights(sentence):
    l = len(sentence)
    w = (l + 1) * [1] + (MAXLEN - (l + 1)) * [0]
    return np.array(w)

def map(f,l):
    return [f(x) for x in l]

def make_examples(l):
    (l1,l2) = zip(*l)
    return {"src_in":map(source_input_conversion,l1),
            "src_len":map(len,l1),
            "tgt_in":map(target_input_conversion,l2),
            "tgt_out":map(target_output_conversion,l2),
            "tgt_weights":map(sentence_target_weights,l2)}

def s2s_generator(src_len,src_in,tgt_in,tgt_out,tgt_weights):
    def gen(bs):
      for i in range(0, bs*(len(src_in)//bs), bs):
          # print ({"src_len":src_len[i:i+bs], "src_in":src_in[i:i+bs], "tgt_in":tgt_in[i:i+bs], "tgt_out":tgt_out[i:i+bs], "tgt_weights":tgt_weights[i:i+bs]})
          yield {"src_len":src_len[i:i+bs],
                 "src_in":src_in[i:i+bs],
                 "tgt_in":tgt_in[i:i+bs],
                 "tgt_out":tgt_out[i:i+bs],
                 "tgt_weights":tgt_weights[i:i+bs]}
    return gen


def my_sample(l,n):
    return list(random.sample(l,min(n,len(l))))

print("Reading sentences...")
all_sentences = [l.strip().split("\t") for l in open("synthtrees.txt").readlines()]

val_set = make_examples(all_sentences[:2000])
train_set = make_examples(all_sentences[2000:])

print("Loading model")
model = mkModel(tf.train.AdamOptimizer())
sess = tf.Session()
saver = tf.train.Saver()

def printer(x):
    (p,y,h) = x
    print("Prob", p, decode(y),h)


def translate(s):
    r = tyf.beam_translate(sess,model,14,
                           source_input_conversion(s),
                           len(s),
                           char_indices["^"], char_indices["$"],
                           printer)
    for x in r: printer(x)

def translate_cb(values):
    if values["epoch"] % 10 == 0:
        save_path = saver.save(sess, "model.ckpt")
        translate("(1(3cb)b)")
        print ("Desired:", "((3cb)b1)")
        return False

tyf.initialize_params(sess,model)
train_stats = tyf.train(sess,
                        model,
                        s2s_generator(**train_set),
                        valid_generator = s2s_generator(**val_set),
                        epochs=5000,
                        callbacks=[tyf.StopWhenAccurate(.01), tyf.StopWhenValidationGetsWorse(2), translate_cb])

translate("(1(3cb)b)")
translate("(1(2c(3e(4(1cb)b)))c)")

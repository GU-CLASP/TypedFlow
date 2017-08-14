{-# LANGUAGE AllowAmbiguousTypes #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}
module Seq2Seq where

-- https://github.com/tensorflow/nmt

-- Neural Machine Translation and Sequence-to-sequence Models:
-- A Tutorial
-- Graham Neubig
-- Language Technologies Institute, Carnegie Mellon University
-- -- https://arxiv.org/pdf/1703.01619.pdf


import TypedFlow

encoder :: forall (vocSize :: Nat) (n :: Nat) (bs :: Nat). 
                 KnownNat vocSize => (KnownNat bs, KnownNat n) =>
                 [Char]
                 -> Tensor '[n, bs] Int32
                 -> Gen
                      (HList '[(T '[512, bs] Float32, T '[512, bs] Float32), (T '[512, bs] Float32, T '[512, bs] Float32)],
                       Tensor '[n, 512, bs] Float32)
encoder prefix input = do
  embs <- parameter (prefix++"embs") embeddingInitializer
  lstm1 <- parameter (prefix++"w1") lstmInitializer
  lstm2 <- parameter (prefix++"w2") lstmInitializer
  let drp = dropout (KeepProb 0.8)
  -- todo: what to do about sentence length?
  (sFinal,h) <-
    (rnn (timeDistribute (embedding @50 @vocSize embs))
      .--.
     rnn (timeDistribute drp)
      .--.
     rnn (onState drp (lstm @512 lstm1))
      .--.
     rnn (timeDistribute drp)
      .--.
     rnnBackwards (onState drp (lstm @512 lstm2))
     ) (I (zeros,zeros) :* I (zeros,zeros) :* Unit) input
  h' <- assign h  -- will be used many times as input to attention model
  return (sFinal,h')

decoder :: forall (n :: Nat) (outVocabSize :: Nat) (bs :: Nat) (d::Nat).
                 KnownNat d => (KnownNat bs, KnownNat outVocabSize, KnownNat n) =>
                 [Char]
                 -> T '[n, d, bs] Float32 -- todo: consider a larger size for the output string
                 -> (HList '[(T '[512, bs] Float32, T '[512, bs] Float32), (T '[512, bs] Float32, T '[512, bs] Float32)])
                 -> Tensor '[n, bs] Int32
                 -> Gen (Tensor '[n, outVocabSize, bs] Float32)
decoder prefix hs thoughtVectors target = do
  embs <- parameter (prefix++"embs") embeddingInitializer
  -- note: for an intra-language translation the embeddings can be shared.
  projs <- parameter (prefix++"proj") denseInitialiser
  lstm1 <- parameter (prefix++"w1") lstmInitializer
  lstm2 <- parameter (prefix++"w2") lstmInitializer
  w1 <- parameter (prefix++"att1") glorotUniform
  w2 <- parameter (prefix++"att2") glorotUniform
  let attn = (luongAttention @64 (luongMultiplicativeScoring w1) w2 hs)
      initAttn = zeros
  let drp = dropout (KeepProb 0.8)
  (_sFinal,outFinal) <-
    (rnn (timeDistribute (embedding @50 @outVocabSize embs)
          .-.
          timeDistribute drp
          .-.
          addAttentionWithFeedback attn
           ((lstm @512 lstm1)
             .-.
             timeDistribute drp
             .-.
             (lstm @512 lstm2))
          .-.
          (timeDistribute (softmax0 . dense projs)) -- TODO: add a softmax?
         )) (I initAttn :* thoughtVectors) target

     -- TODO: should we use the states for all layers as
     -- thoughtVectors? Or just the top one?
  return outFinal

seq2seq :: forall (inVocSize :: Nat) (outVocSize :: Nat) (n :: Nat) (bs :: Nat).
                 KnownNat inVocSize => KnownNat outVocSize => 
                 (KnownNat bs, KnownNat n) =>
                 Tensor '[n, bs] Int32 ->
                 Tensor '[n, bs] Int32 ->
                 Gen (Tensor '[n, outVocSize, bs] Float32)
seq2seq input gold = do
  (thought,h) <- encoder @inVocSize "enc" input
  decoder "dec" h thought gold

-- TODO: beam search decoder. Perhaps best implemented outside of tensorflow?

trainModel :: Tensor '[20, 128] Int32
                    -> Tensor '[20, 128] Int32 -> Gen (ModelOutput '[20, 128] Int32)
trainModel input gold = do
  y_ <- seq2seq @10000 @10000 @20 @128 input gold
  timedCategorical y_ gold

main :: IO ()
main = generateFile "s2s_model.py" (compile (defaultOptions {maxGradientNorm = Just 1})
                                    trainModel)

{-> main

Parameters:
decatt2: T [64, 1024] tf.float32
decatt1: T [512, 512] tf.float32
decw2_4_snd: T [512] tf.float32
decw2_4_fst: T [512, 1024] tf.float32
decw2_3_snd: T [512] tf.float32
decw2_3_fst: T [512, 1024] tf.float32
decw2_2_snd: T [512] tf.float32
decw2_2_fst: T [512, 1024] tf.float32
decw2_1_snd: T [512] tf.float32
decw2_1_fst: T [512, 1024] tf.float32
decw1_4_snd: T [512] tf.float32
decw1_4_fst: T [512, 626] tf.float32
decw1_3_snd: T [512] tf.float32
decw1_3_fst: T [512, 626] tf.float32
decw1_2_snd: T [512] tf.float32
decw1_2_fst: T [512, 626] tf.float32
decw1_1_snd: T [512] tf.float32
decw1_1_fst: T [512, 626] tf.float32
decproj_snd: T [10000] tf.float32
decproj_fst: T [10000, 64] tf.float32
decembs: T [50, 10000] tf.float32
encw2_4_snd: T [512] tf.float32
encw2_4_fst: T [512, 1024] tf.float32
encw2_3_snd: T [512] tf.float32
encw2_3_fst: T [512, 1024] tf.float32
encw2_2_snd: T [512] tf.float32
encw2_2_fst: T [512, 1024] tf.float32
encw2_1_snd: T [512] tf.float32
encw2_1_fst: T [512, 1024] tf.float32
encw1_4_snd: T [512] tf.float32
encw1_4_fst: T [512, 562] tf.float32
encw1_3_snd: T [512] tf.float32
encw1_3_fst: T [512, 562] tf.float32
encw1_2_snd: T [512] tf.float32
encw1_2_fst: T [512, 562] tf.float32
encw1_1_snd: T [512] tf.float32
encw1_1_fst: T [512, 562] tf.float32
encembs: T [50, 10000] tf.float32
-}



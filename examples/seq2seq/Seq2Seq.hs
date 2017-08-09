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
                      (HList '[(T '[512, bs] Float32, T '[512, bs] Float32), (T '[512, bs] Float32, T '[512, bs] Float32)])
encoder prefix input = do
  embs <- parameter (prefix++"embs") embeddingInitializer
  lstm1 <- parameter (prefix++"w1") lstmInitializer
  lstm2 <- parameter (prefix++"w2") lstmInitializer
  -- todo: what to do about sentence length?
  (sFinal,_) <-
    (rnn (timeDistribute (embedding @50 @vocSize embs))
      .--.
     rnn (lstm @512 lstm1)
      .--.
     rnnBackwards (lstm @512 lstm2)
     ) (I (zeros,zeros) :* I (zeros,zeros) :* Unit) input
  return sFinal

decoder :: forall (n :: Nat) (outVocabSize :: Nat) (bs :: Nat).
                 (KnownNat bs, KnownNat outVocabSize, KnownNat n) =>
                 [Char]
                 -> (HList '[(T '[512, bs] Float32, T '[512, bs] Float32), (T '[512, bs] Float32, T '[512, bs] Float32)])
                 -> Tensor '[n, bs] Int32
                 -> Gen (Tensor '[n, outVocabSize, bs] Float32)
decoder prefix thoughtVectors target = do
  embs <- parameter (prefix++"embs") embeddingInitializer
  -- note: for an intra-language translation the embeddings can be shared.
  projs <- parameter (prefix++"proj") denseInitialiser
  lstm1 <- parameter (prefix++"w1") lstmInitializer
  lstm2 <- parameter (prefix++"w2") lstmInitializer
  (_sFinal,outFinal) <-
    (rnn (timeDistribute (embedding @50 @outVocabSize embs))
      .--.
     rnn (timeDistribute (dropout (KeepProb 0.8)))
      .--.
     rnn (recurrentDropout (KeepProb 0.8) (lstm @512 lstm1))
      .--.
     rnnBackwards (lstm @512 lstm2)
      .--.
     rnn (timeDistribute (dense projs))
     ) thoughtVectors target

     -- TODO: should we send the states for all layers? Or just the top one?
  return outFinal

seq2seq :: forall (inVocSize :: Nat) (outVocSize :: Nat) (n :: Nat) (bs :: Nat).
                 KnownNat inVocSize => KnownNat outVocSize => 
                 (KnownNat bs, KnownNat n) =>
                 Tensor '[n, bs] Int32 ->
                 Tensor '[n, bs] Int32 ->
                 Gen (Tensor '[n, outVocSize, bs] Float32)
seq2seq input gold = do
  thought <- encoder @inVocSize "enc" input
  decoder "dec" thought gold

-- TODO: beam search decoder. Perhaps best implemented outside of tensorflow?


-- main :: IO ()
-- main = do
--   writeFile "s2s_model.py" (generate $ compile defaultOptions (mnist @None))
--   putStrLn "done!"

{-> main


-}



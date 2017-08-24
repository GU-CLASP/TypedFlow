{-# LANGUAGE AllowAmbiguousTypes #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}
module Aggr where

import TypedFlow

predict :: forall (vocSize::Nat) batchSize. KnownNat vocSize => KnownNat batchSize => Model '[21,batchSize] Int32 '[batchSize] Int32
predict input gold = do
  embs <- parameter "embs" embeddingInitializer
  lstm1 <- parameter "w1" lstmInitializer
  let drp = dropout (KeepProb 0.9)
  w <- parameter "dense" denseInitialiser
  (_sFi,predictions) <-
    rnn (timeDistribute (embedding @9 @vocSize embs)
          .-.
          timeDistribute drp
          .-.
          (onState drp (lstm @50 lstm1)))
        (I (zeros,zeros) :* Unit) input
  categorical ((dense @vocSize w) (last0 predictions)) gold


main :: IO ()
main = do
  generateFile "lm.py" (compile (defaultOptions {maxGradientNorm = Just 1}) (predict @8 @128))
  putStrLn "done!"

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


{-> main

Parameters (total 12480):
dense_snd: T [8] tf.float32
dense_fst: T [8,50] tf.float32
w1_4_snd: T [50] tf.float32
w1_4_fst: T [50,59] tf.float32
w1_3_snd: T [50] tf.float32
w1_3_fst: T [50,59] tf.float32
w1_2_snd: T [50] tf.float32
w1_2_fst: T [50,59] tf.float32
w1_1_snd: T [50] tf.float32
w1_1_fst: T [50,59] tf.float32
embs: T [9,8] tf.float32
done!
-}




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

lm :: forall (vocSize::Nat) batchSize. KnownNat vocSize => KnownNat batchSize => Model '[21,batchSize] Int32 '[batchSize] Int32
lm input gold = do
  embs <- parameter "embs" embeddingInitializer
  lstm1 <- parameter "w1" lstmInitializer
  let drp = dropout (KeepProb 0.9)
  w <- parameter "dense" denseInitialiser
  (_sFi,predictions) <-
    rnn (timeDistribute (embedding @9 @vocSize embs)
          .-.
          timeDistribute drp
          .-.
          (onState drp (lstm @50 lstm1))
          .-.
          timeDistribute (softmax0 . squeeze0 . dense  w))
        (I (zeros,zeros) :* Unit) input
  categorical (last0 predictions) gold


main :: IO ()
main = do
  generateFile "lm.py" (compile (defaultOptions {maxGradientNorm = Just 1}) (lm @8 @128))
  putStrLn "done!"

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


{-> main

done!
-}




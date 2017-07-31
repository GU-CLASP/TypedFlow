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


agreement :: KnownNat batchSize => Tensor '[20,batchSize] Int32 -> Gen (Tensor '[20,batchSize] Float32)
agreement input = do
  embs <- parameter "embs" embeddingInitializer
  lstm1 <- parameter "w1" lstmInitializer
  lstm2 <- parameter "w2" lstmInitializer
  w <- parameter "dense" denseInitialiser
  (_sFi,out) <- rnn (timeDistribute (embedding @50 @100000 embs)
                     .--.
                     (lstm @150 lstm1)
                     .--.
                     (lstm @150 lstm2)
                     .--.
                     timeDistribute (sigmoid . squeeze0 . dense  w))
                (() |> (zeros,zeros) |> (zeros,zeros) |> (),input)
  return out


(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


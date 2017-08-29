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

predict :: forall (outSize::Nat) (vocSize::Nat) batchSize. KnownNat outSize => KnownNat vocSize => KnownNat batchSize => Model '[21,batchSize] Int32 '[batchSize] Int32
predict input gold = do
  embs <- parameter "embs" embeddingInitializer
  lstm1 <- parameter "w1" lstmInitializer
  drp <- mkDropout (DropProb 0.1)
  rdrp <- mkDropouts (DropProb 0.1)
  w <- parameter "dense" denseInitialiser
  (_sFi,predictions) <-
    rnn (timeDistribute (embedding @9 @vocSize embs)
          .-.
          timeDistribute drp
          .-.
          (onState (rdrp) (lstm @50 lstm1)))
        (I (VecPair zeros zeros) :* Unit) input
  categorical ((dense @outSize w) (last0 predictions)) gold


main :: IO ()
main = do
  generateFile "lm.py" (compile (defaultOptions) (predict @5 @11 @512))
  putStrLn "done!"

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


{-> main


<interactive>:57:1: error:
    • Variable not in scope: main
    • Perhaps you meant ‘min’ (imported from Prelude)
-}




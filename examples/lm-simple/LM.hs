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

predict :: forall len (vocSize::Nat) bs. KnownNat len => KnownNat vocSize => KnownNat bs =>
           Gen (ModelOutput '[len,vocSize,bs] Float32)
predict = do
  input <- placeholder "x"
  gold <- placeholder "y"
  masks <- placeholder "weights"

  embs <- parameter "embs" embeddingInitializer
  lstm1 <- parameter "w1" lstmInitializer
  drp <- mkDropout (DropProb 0.1)
  rdrp <- mkDropouts (DropProb 0.1)
  w <- parameter "dense" denseInitialiser
  (_sFi,predictions) <-
    rnn (timeDistribute (embedding @12 @vocSize embs)
          .-.
          timeDistribute drp
          .-.
          onStates rdrp (lstm @150 lstm1)
          .-.
          timeDistribute (dense @vocSize w)
        )
        (repeatT zeros) input
  timedCategorical @len @vocSize @bs masks predictions gold


main :: IO ()
main = do
  generateFile "lm.py" (compileGen defaultOptions (predict @21 @12 @512))
  putStrLn "done!"

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


{-> main


<interactive>:57:1: error:
    • Variable not in scope: main
    • Perhaps you meant ‘min’ (imported from Prelude)
-}




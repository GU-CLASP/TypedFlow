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

agreement :: KnownNat batchSize => Model '[20,batchSize] Int32 '[batchSize] Int32
agreement input gold = do
  embs <- parameter "embs" embeddingInitializer
  lstm1 <- parameter "w1" lstmInitializer
  lstm2 <- parameter "w2" lstmInitializer
  w <- parameter "dense" denseInitialiser
  (_sFi,predictions) <-
    rnn (timeDistribute (embedding @50 @100000 embs)
          .--.
          (lstm @150 lstm1)
          .--.
          (lstm @150 lstm2)
          .--.
          timeDistribute (sigmoid . squeeze0 . dense  w))
        (() |> (zeros,zeros) |> (zeros,zeros) |> (),input)
  binary (last0 predictions) gold


main :: IO ()
main = do
  writeFile "aggr_model.py" (generate $ compile (agreement @None))
  putStrLn "done!"

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


{-> main

done!
-}




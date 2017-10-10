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
  embs <- parameterDefault "embs"
  lstm1 <- parameterDefault "w1"
  lstm2 <- parameterDefault "w2"
  w <- parameterDefault "dense"
  (_sFi,predictions) <-
    rnn (timeDistribute (embedding @50 @100000 embs)
          .-.
          (lstm @150 lstm1)
          .-.
          (lstm @150 lstm2)
          .-.
          timeDistribute (sigmoid . squeeze0 . dense  w))
        (repeatT zeros) input
  binary (last0 predictions) gold


main :: IO ()
main = do
  generateFile "aggr_model.py" (compile (defaultOptions {maxGradientNorm = Just 1}) (agreement @1024))
  putStrLn "done!"

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


{-> main

done!
-}



-- Local Variables:
-- dante-repl-command-line: ("nix-shell" ".styx/shell.nix" "--pure" "--run" "cabal repl")
-- End:


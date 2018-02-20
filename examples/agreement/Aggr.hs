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
import TypedFlow.Python

agreement :: Gen (Model '[20] Int32 '[] '[] '[] Int32)
agreement = do
  embs <- parameterDefault "embs"
  lstm1 <- parameterDefault "w1"
  lstm2 <- parameterDefault "w2"
  w <- parameterDefault "dense"
  return $ \input gold -> 
    let (_sFi,predictions) = runLayer 
            (iterateCell ((timeDistribute (embedding @50 @100000 embs)
                           .-.
                           (lstm @150 lstm1)
                           .-.
                           (lstm @150 lstm2)
                           .-.
                           timeDistribute (sigmoid . squeeze0 . dense  w))))
                (repeatT zeros, input)
    in binary (last0 predictions) gold


main :: IO ()
main = do
  generateFile "aggr_model.py" (compile @1024 (defaultOptions {maxGradientNorm = Just 1}) agreement)
  putStrLn "done!"

-- >>> main
-- Parameters (total *** Exception: broadcast on gather index not implemented
-- CallStack (from HasCallStack):
--   error, called at ./TypedFlow/Abstract.hs:102:20 in typedflow-0.9-H6KPYKhyHGIKxqNXew8K1n:TypedFlow.Abstract

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


{-> main

done!
-}



-- Local Variables:
-- dante-repl-command-line: ("nix-shell" ".styx/shell.nix" "--pure" "--run" "cabal repl")
-- End:


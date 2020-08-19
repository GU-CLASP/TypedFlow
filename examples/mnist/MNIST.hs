{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE NoStarIsType #-}
module MNIST where

import TypedFlow
import TypedFlow.Python

atShape :: forall s t. T s t -> T s t
atShape x = x

mnist :: Gen (Model '[784] Float32 '[10] '[10] '[] Float32)
mnist = do
  filters1 <- parameterDefault "f1"
  filters2 <- parameterDefault "f2"
  w1 <- parameterDefault "w1"
  w2 <- parameterDefault "w2"
  return $ \input gold ->
    let nn = dense @10 w2                       .
             relu . dense @1024 w1              .
             reshape @'[7 * 7 * 64]             .
             maxPool2D @2 @2                    .
             relu . conv @64 @'[5,5] filters2   .
             maxPool2D @2 @2                    .
             atShape @'[28,28,32]               .
             relu . conv @32 @'[5,5] filters1   .
             reshape @'[28,28,1]
        logits = nn input

    in categoricalDistribution logits gold

main :: IO ()
main = do
  generateFile "mnist_model.py" (compile @100 defaultOptions mnist)
  putStrLn "done!"

-- >>> main
-- Parameters (total *** Exception: broadcast on both convolution filter and data not implemented
-- CallStack (from HasCallStack):
--   error, called at ./TypedFlow/Abstract.hs:250:20 in typedflow-0.9-inplace:TypedFlow.Abstract


-- Local Variables:
-- dante-repl-command-line: ("nix-shell" ".styx/shell.nix" "--pure" "--run" "cabal repl")
-- End:


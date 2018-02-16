{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}
module MNIST where

import TypedFlow
import TypedFlow.Python

(#>) :: forall b c a. (a -> b) -> (b -> c) -> a -> c
(#>) = flip (.)

atShape :: forall s t. T s t -> T s t
atShape x = x

mnist :: Gen (Model '[784] Float32 '[10] '[10] '[] Float32)
mnist = do
  filters1 <- parameterDefault "f1"
  filters2 <- parameterDefault "f2"
  w1 <- parameterDefault "w1"
  w2 <- parameterDefault "w2"
  return $ \input gold ->
    let nn = reshape @'[28,28,1]                #>
             (relu . conv @32 @'[5,5] filters1) #>
             atShape @'[28,28,32]               #>
             maxPool2D @2 @2                    #>
             atShape @'[14,14,32]               #>
             (relu . conv @64 @'[5,5] filters2) #>
             maxPool2D @2 @2                    #>
             flatten3                           #>
             (relu . dense @1024 w1)            #>
             dense @10 w2
        logits = nn input
    in categoricalDistribution logits gold

main :: IO ()
main = do
  generateFile "mnist_model.py" (compile @None defaultOptions mnist)
  putStrLn "done!"

-- >>> main
-- Parameters (total 3274634):
-- w2_bias: T [10] tf.float32
-- w2_w: T [1024,10] tf.float32
-- w1_bias: T [1024] tf.float32
-- w1_w: T [3136,1024] tf.float32
-- f2_biases: T [64] tf.float32
-- f2_filters: T [5,5,32,64] tf.float32
-- f1_biases: T [32] tf.float32
-- f1_filters: T [5,5,1,32] tf.float32
-- done!



-- Local Variables:
-- dante-repl-command-line: ("nix-shell" ".styx/shell.nix" "--pure" "--run" "cabal repl")
-- End:


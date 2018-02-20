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
-- Parameters (total 5301351):
-- dense_bias: T [1] tf.float32
-- dense_w: T [150,1] tf.float32
-- w2_o_bias: T [150] tf.float32
-- w2_o_w: T [300,150] tf.float32
-- w2_c_bias: T [150] tf.float32
-- w2_c_w: T [300,150] tf.float32
-- w2_i_bias: T [150] tf.float32
-- w2_i_w: T [300,150] tf.float32
-- w2_f_bias: T [150] tf.float32
-- w2_f_w: T [300,150] tf.float32
-- w1_o_bias: T [150] tf.float32
-- w1_o_w: T [200,150] tf.float32
-- w1_c_bias: T [150] tf.float32
-- w1_c_w: T [200,150] tf.float32
-- w1_i_bias: T [150] tf.float32
-- w1_i_w: T [200,150] tf.float32
-- w1_f_bias: T [150] tf.float32
-- w1_f_w: T [200,150] tf.float32
-- embs: T [100000,50] tf.float32
-- done!

(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>



-- Local Variables:
-- dante-repl-command-line: ("nix-shell" ".styx/shell.nix" "--pure" "--run" "cabal repl")
-- End:


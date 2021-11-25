{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}


import TypedFlow
import TypedFlow.Python
import qualified GHC.Int as GHC

onFST :: (Tensor s1 t -> Tensor s t) -> HTV t '[s1, s'] -> HTV t '[s, s']
onFST f (VecPair h c) = (VecPair (f h) c)

mkLSTM :: ∀ n x. KnownNat x => KnownNat n => 
        String -> DropProb -> Gen (RnnCell Float32 '[ '[n], '[n]] (Tensor '[x] Float32) (Tensor '[n] Float32))
mkLSTM pName dropProb = do
  params <- parameterDefault pName
  drp1 <- mkDropout dropProb
  rdrp1 <- mkDropout dropProb
  return (timeDistribute drp1 .-. onStates (onFST rdrp1) (lstm params))

model :: forall (vocSize::Nat) (len::Nat). KnownNat len => KnownNat vocSize =>
   Gen (T '[len] Int32 -> T '[len] Int32 -> ModelOutput Float32 '[len,vocSize] '[])
model = do
  embs <- parameterDefault "embs"
  let dropProb = DropProb 0.10
  lstm1 <- mkLSTM @160 "w1" dropProb
  drp <- mkDropout dropProb
  w <- parameterDefault "dense"
  return $ \input gold -> do
    let masks = constant 1 ⊝ cast @Float32 (equal (constant padding) input)
        (_sFi,predictions) =
          simpleRnn (timeDistribute (embedding @12 @vocSize embs) .-.
            lstm1 .-.
            timeDistribute drp .-.
            timeDistribute (dense w))
          (repeatT zeros, input)
      in timedCategorical masks predictions gold

padding :: GHC.Int32
padding = 10

main :: IO ()
main = do
  generateFile "aggr.py" (compile @512 defaultOptions (model @12 @21))
  putStrLn "done!"

-- >>> main
-- Parameters (total 134300):
-- dense_bias: T [12] tf.float32
-- dense_w: T [160,12] tf.float32
-- w1_o_b: T [160] tf.float32
-- w1_o_w: T [172,160] tf.float32
-- w1_c_b: T [160] tf.float32
-- w1_c_w: T [172,160] tf.float32
-- w1_i_b: T [160] tf.float32
-- w1_i_w: T [172,160] tf.float32
-- w1_f_b: T [160] tf.float32
-- w1_f_w: T [172,160] tf.float32
-- embs: T [12,12] tf.float32
-- y: T [512,21] tf.int32
-- x: T [512,21] tf.int32
-- done!


(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


-- Local Variables:
-- dante-repl-command-line: ("nix-shell" ".styx/shell.nix" "--pure" "--run" "cabal repl")
-- End:


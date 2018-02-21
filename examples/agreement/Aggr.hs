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

onFST :: (Tensor s1 t -> Tensor s t) -> HTV t '[s1, s'] -> HTV t '[s, s']
onFST f (VecPair h c) = (VecPair (f h) c)

mkLSTM :: ∀ n x. KnownNat x => KnownNat n => 
        String -> DropProb -> Gen (RnnCell 'B32 '[ '[n], '[n]] (Tensor '[x] Float32) (Tensor '[n] Float32))
mkLSTM pName dropProb = do
  params <- parameterDefault pName
  drp1 <- mkDropout dropProb
  rdrp1 <- mkDropout dropProb
  return (timeDistribute drp1 .-. onStates (onFST rdrp1) (lstm params))

model :: forall (vocSize::Nat) (len::Nat). KnownNat len => KnownNat vocSize =>
   Gen (T '[len] Int32 -> T '[len] Int32 -> ModelOutput Float32 '[vocSize] '[len])
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

padding :: Int
padding = 10

main :: IO ()
main = do
  generateFile "aggr.py" (compile @512 defaultOptions (model @12 @21))
  putStrLn "done!"

-- >>> main
-- <interactive>:6859:2: warning: [-Wdeferred-out-of-scope-variables]
--     • Variable not in scope: main :: IO a0
--     • Perhaps you meant ‘min’ (imported from Prelude)
-- *** Exception: <interactive>:6859:2: error:
--     • Variable not in scope: main :: IO a0
--     • Perhaps you meant ‘min’ (imported from Prelude)
-- (deferred type error)


(|>) :: ∀ a b. a -> b -> (a, b)
(|>) = (,)
infixr |>


-- Local Variables:
-- dante-repl-command-line: ("nix-shell" ".styx/shell.nix" "--pure" "--run" "cabal repl")
-- End:


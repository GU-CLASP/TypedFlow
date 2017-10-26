{-# LANGUAGE AllowAmbiguousTypes #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}

module Main (main) where

import TypedFlow

mkLSTM :: ∀ n x bs w. KnownNat x => KnownNat n => (KnownNat bs) => KnownBits w =>
        String -> Gen (RnnCell w '[ '[n,bs], '[n,bs]] (Tensor '[x,bs] (Flt w)) (Tensor '[n,bs] (Flt w)))
mkLSTM pName = do
  params <- parameterDefault pName
  drp1 <- mkDropout (DropProb 0.05)
  rdrp1 <- mkDropouts (DropProb 0.05)
  return (timeDistribute drp1 .-. onStates rdrp1 (lstm params))

encoder :: forall (lstmSize :: Nat) (vocSize :: Nat) (n :: Nat) (bs :: Nat) w. 
                 KnownNat lstmSize => KnownNat vocSize => (KnownNat bs, KnownNat n) => KnownBits w =>
                 String
                 -> T '[bs] Int32 -- lengths
                 -> Tensor '[n, bs] Int32
                 -> Gen
                      (HTV (Flt w) '[ '[lstmSize, bs], '[lstmSize, bs] ],
                       Tensor '[n, lstmSize, bs] (Flt w))
encoder prefix lens input = do
  embs <- parameterDefault (prefix++"embs")
  lstm1 <- mkLSTM (prefix++"lstm1")
  (sFinal,h) <-
    (rnn (timeDistribute (embedding @vocSize @vocSize embs))
     .--.
     rnnBackwardsWithCull lens lstm1)
    (repeatT zeros) input
  h' <- assign h  -- will be used many times as input to attention model
  return (sFinal,h')

decoder :: forall (lstmSize :: Nat) (n :: Nat) (outVocabSize :: Nat) (bs :: Nat) (d::Nat) w.
                 KnownNat lstmSize => KnownNat d => (KnownNat bs, KnownNat outVocabSize, KnownNat n) => KnownBits w =>
                 String
                 -> T '[bs] Int32 -- ^ lengths
                 -> T '[n, d, bs] (Flt w) -- todo: consider a larger size for the output string
                 -> HTV (Flt w) '[ '[lstmSize, bs], '[lstmSize, bs] ]
                 -> Tensor '[n, bs] Int32
                 -> Gen (Tensor '[n, outVocabSize, bs] (Flt w))
decoder prefix lens hs thoughtVectors targetInput = do
  -- note: for an intra-language translation the embeddings can be shared easily.
  projs <- parameterDefault (prefix++"proj")
  lstm1 <- mkLSTM (prefix++"lstm1")
  embs <- parameterDefault "embs"
  w1 <- parameter (prefix++"att1") glorotUniform
  let attn = uniformAttn (multiplicativeScoring w1) lens hs -- NOTE: attention on the left-part of the input.
  (_sFinal,outFinal) <-
    rnn ((timeDistribute (embedding @outVocabSize @outVocabSize embs)
           .-.
          (attentiveWithFeedback attn lstm1)
           .-.
          timeDistribute (dense projs)))
        (F zeros :* thoughtVectors)
        targetInput
  return outFinal



seq2seq :: forall (vocSize :: Nat) (n :: Nat) (bs :: Nat) w.
                 KnownNat vocSize => (KnownNat bs, KnownNat n) => KnownBits w =>
                 Tensor '[n, bs] Int32 ->
                 Tensor '[bs] Int32 ->
                 Tensor '[n, bs] Int32 ->
                 Gen (Tensor '[n, vocSize, bs] (Flt w))
seq2seq input inputLen output = do
  (VecPair t1 t2,h) <- encoder @256 @vocSize "enc" inputLen input
  decoder "dec" inputLen h (VecPair t1 t2) output

model :: forall w vocSize len batchSize. KnownNat batchSize => KnownNat vocSize => KnownNat len => KnownBits w =>
         Gen (ModelOutput '[len, vocSize, batchSize] (Flt w))
model = do
  sourceInput <- placeholder "src_in"
  sourceLen <- placeholder "src_len"
  targetInput <- placeholder "tgt_in"
  targetOutput <- placeholder "tgt_out"
  masks <- placeholder "tgt_weights"
  y_ <- seq2seq @vocSize @len sourceInput sourceLen targetInput
  timedCategorical masks y_ targetOutput


main :: IO ()
main = generateFile "s2s.py" (compileGen
                               defaultOptions {maxGradientNorm = Just 5}
                               (model @'B32 @15 @22 @256))

-- Local Variables:
-- dante-repl-command-line: ("nix-shell" ".styx/shell.nix" "--pure" "--run" "cabal repl")
-- End:


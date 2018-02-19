{-# LANGUAGE AllowAmbiguousTypes #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}

module Main  where

import TypedFlow

mkLSTM :: ∀ n x w.
       KnownNat x => KnownNat n => KnownBits w
       => String
       -> Gen (RnnCell w '[ '[n], '[n]] (Tensor '[x] (Flt w))
               (Tensor '[n] (Flt w)))
mkLSTM pName = do
  params <- parameterDefault pName
  drp1 <- mkDropout (DropProb 0.05)
  rdrp1 <- mkDropouts (DropProb 0.05)
  return (timeDistribute drp1 .-. onStates rdrp1 (lstm params))

encoder :: forall (lstmSize :: Nat) (vocSize :: Nat) (n :: Nat)  w. 
           KnownNat lstmSize => KnownNat vocSize
        => (KnownNat n) => KnownBits w
        => String
        -> Gen
        (
        T '[] Int32 -- length
        -> Tensor '[n] Int32 -> 
          ((HTV (Flt w) '[ '[lstmSize], '[lstmSize] ], Tensor '[n, lstmSize] (Flt w))))
encoder prefix = do
  embs <- parameterDefault (prefix++"embs")
  lstm1 <- mkLSTM (prefix++"lstm1")
  return $ \len input ->
    runLayer
       (iterateWithCull len (timeDistribute (embedding @vocSize @vocSize embs) .-. lstm1))
       (repeatT zeros, input)

decoder :: forall (lstmSize :: Nat) (n :: Nat) (outVocabSize :: Nat) (d::Nat) w.
                 KnownNat lstmSize => KnownNat d => (KnownNat outVocabSize, KnownNat n) => KnownBits w =>
                 String
                 -> Gen (
                 T '[] Int32 -- ^ length
                 -> T '[n, d] (Flt w) -- todo: consider a larger size for the output string
                 -> HTV (Flt w) '[ '[lstmSize], '[lstmSize] ]
                 -> Tensor '[n] Int32
                 -> Tensor '[n, outVocabSize] (Flt w))
decoder prefix = do
  -- note: for an intra-language translation the embeddings can be shared easily.
  projs <- parameterDefault (prefix++"proj")
  lstm1 <- mkLSTM (prefix++"lstm1")
  embs <- parameterDefault "embs"
  w1 <- parameter (prefix++"att1") glorotUniform
  return $ \ lens hs thoughtVectors targetInput ->
    let attn = uniformAttn (multiplicativeScoring w1) lens hs -- NOTE: attention on the left-part of the input.
        (_sFinal,outFinal) = runLayer 
          (iterateCell ((timeDistribute (embedding @outVocabSize @outVocabSize embs)
                         .-.
                         (attentiveWithFeedback attn lstm1)
                         .-.
                         timeDistribute (dense projs))))
          ((F zeros :* thoughtVectors), targetInput)
    in outFinal


seq2seq :: forall (vocSize :: Nat) (n :: Nat)  w.
                 KnownNat vocSize => (KnownNat n) => KnownBits w
        => Tensor '[n] Int32
        -> Tensor '[] Int32
        -> Tensor '[n] Int32
        -> Gen (Tensor '[n, vocSize] (Flt w))
seq2seq input inputLen output = do
  (VecPair t1 t2,h) <- encoder @256 @vocSize "enc" inputLen input
  decoder "dec" inputLen h (VecPair t1 t2) output

model :: forall w vocSize len batchSize.
  KnownNat batchSize
  => KnownNat vocSize => KnownNat len => KnownBits w
  => Gen (ModelOutput '[len, vocSize, batchSize] (Flt w))
model = do
  sourceInput <- placeholder "src_in"
  sourceLen <- placeholder "src_len"
  targetInput <- placeholder "tgt_in"
  targetOutput <- placeholder "tgt_out"
  masks <- placeholder "tgt_weights"
  y_ <- seq2seq @vocSize @len sourceInput sourceLen targetInput
  timedCategorical masks y_ targetOutput
{-



main :: IO ()
main = generateFile "s2s.py" (compileGen
                               defaultOptions {maxGradientNorm = Just 5}
                               (model @'B32 @15 @22 @256))

-- Local Variables:
-- dante-repl-command-line: ("nix-shell" ".styx/shell.nix" "--pure" "--run" "cabal repl")
-- End:

-}

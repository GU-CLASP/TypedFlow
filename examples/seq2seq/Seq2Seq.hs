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
import TypedFlow.Python

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
    runRnn
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
        (_sFinal,outFinal) = runRnn
          (iterateCell ((timeDistribute (embedding @outVocabSize @outVocabSize embs)
                         .-.
                         (attentiveWithFeedback attn lstm1)
                         .-.
                         timeDistribute (dense projs))))
          ((F zeros :* thoughtVectors), targetInput)
    in outFinal


seq2seq :: forall (vocSize :: Nat) (n :: Nat).
                 KnownNat vocSize => (KnownNat n)
        => Gen (HHTV '[ '[n] ':& Float32, '[n] ':& Int32, '[] ':& Int32, '[n] ':& Int32, '[n] ':& Int32] ->
                ModelOutput Float32 '[vocSize] '[n])
seq2seq  = do
  enc <- encoder @256 @vocSize "enc"
  dec <- decoder "dec"
  return $ \(Uncurry masks :* Uncurry input :* Uncurry inputLen :* Uncurry srcOut :* Uncurry tgtOut :* Unit) ->
    let (VecPair t1 t2,h) = enc inputLen input
        y_ = dec inputLen h (VecPair t1 t2) srcOut
    in timedCategorical masks y_ tgtOut




main :: IO ()
main = generateFile "s2s.py" (compileGen @256
                               defaultOptions {maxGradientNorm = Just 5}
                               (LS (HolderName "tgt_weights") $
                                LS (HolderName "src_in") $
                                LS (HolderName "src_len") $
                                LS (HolderName "src_out") $
                                LS (HolderName "tgt_out") $
                                LZ)
                               (seq2seq @15 @22))

-- Local Variables:
-- dante-repl-command-line: ("nix-shell" ".styx/shell.nix" "--pure" "--run" "cabal repl")
-- End:


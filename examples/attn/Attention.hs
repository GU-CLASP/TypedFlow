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

mkLSTM :: ∀ n x bs. KnownNat x => KnownNat n => (KnownNat bs) =>
        String -> Gen (RnnCell '[ '[n,bs], '[n,bs]] (Tensor '[x,bs] Float32) (Tensor '[n,bs] Float32))
mkLSTM pName = do
  params <- parameter pName lstmInitializer
  drp1 <- mkDropout (DropProb 0.2)
  rdrp1 <- mkDropouts (DropProb 0.2)
  return (timeDistribute drp1 .-. onStates rdrp1 (lstm params))

infixl 0 ⋆
(⋆) :: (a -> b) -> a -> b
(⋆) = ($)
encoder :: forall (lstmSize :: Nat) (vocSize :: Nat) (n :: Nat) (bs :: Nat). 
                 KnownNat lstmSize => KnownNat vocSize => (KnownNat bs, KnownNat n) =>
                 String
                 -> T '[bs] Int32 -- lengths
                 -- -> EmbbeddingP vocSize 50 'B32
                 -> Tensor '[n, bs] Int32
                 -> Gen
                      (FHTV '[ '[lstmSize, bs], '[lstmSize, bs] ],
                       Tensor '[n, lstmSize, bs] Float32)
encoder prefix lens input = do
  embs <- parameter "embs" embeddingInitializer
  lstm1 <- mkLSTM (prefix++"lstm1")
  (sFinal,h) <- rnn  -- TODO: cull
    ⋆ (timeDistribute (embedding @50 @vocSize embs)
       .-.
       lstm1)
    ⋆ (repeatT zeros)
    ⋆ input
  h' <- assign h  -- will be used many times as input to attention model
  return (sFinal,h')

decoder :: forall (lstmSize :: Nat) (n :: Nat) (outVocabSize :: Nat) (bs :: Nat) (d::Nat).
                 KnownNat lstmSize => KnownNat d => (KnownNat bs, KnownNat outVocabSize, KnownNat n) =>
                 String
                 -- -> EmbbeddingP outVocabSize 50 'B32
                 -> T '[bs] Int32 -- ^ lengths
                 -> T '[n, d, bs] Float32 -- todo: consider a larger size for the output string
                 -> FHTV '[ '[lstmSize, bs], '[lstmSize, bs]]
                 -> Tensor '[n, bs] Int32
                 -> Gen (Tensor '[n, outVocabSize, bs] Float32)
decoder prefix lens hs thoughtVectors targetInput = do
  -- note: for an intra-language translation the embeddings can be shared.
  projs <- parameter (prefix++"proj") denseInitialiser
  lstm1 <- mkLSTM (prefix++"lstm1")
  embs <- parameter "embs" embeddingInitializer
  w1 <- parameter (prefix++"att1") glorotUniform
  let attn = uniformAttn lens (luongMultiplicativeScoring w1) hs
  (_sFinal,outFinal) <-
    rnn ⋆ (timeDistribute (embedding @50 @outVocabSize embs)
            .-.
            lstm1
            .-.
            withBypass (timeDistribute' attn)
            .-.
            timeDistribute (dense projs))
        ⋆ thoughtVectors
        ⋆ targetInput

     -- TODO: should we use the states for all layers as
     -- thoughtVectors? Or just the top one?
  return outFinal



seq2seq :: forall (vocSize :: Nat) (n :: Nat) (bs :: Nat).
                 KnownNat vocSize => 
                 (KnownNat bs, KnownNat n) =>
                 Tensor '[n, bs] Int32 ->
                 Tensor '[bs] Int32 ->
                 Tensor '[n, bs] Int32 ->
                 Gen (Tensor '[n, vocSize, bs] Float32)
seq2seq input inputLen outputPlusStart = do
  (thought,h) <- encoder @100 @vocSize "enc" inputLen input
  decoder "dec" inputLen h thought outputPlusStart

model :: forall vocSize len batchSize. KnownNat batchSize => KnownNat vocSize => KnownNat len => Gen (ModelOutput '[len, vocSize, batchSize] Float32)
model = do
  sourceInput <- placeholder "src_in"
  sourceLen <- placeholder "src_len"
  targetInput <- placeholder "tgt_in"
  targetOutput <- placeholder "tgt_out"
  masks <- placeholder "tgt_weights"
  y_ <- seq2seq @vocSize @len sourceInput sourceLen targetInput
  timedCategorical masks y_ targetOutput


main :: IO ()
main = generateFile "s2s.py" (compileGen (defaultOptions -- {maxGradientNorm = Just 1}
                                         ) (model @12 @20 @512))

{-> main

-}



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

encoder :: forall (lstmSize :: Nat) (vocSize :: Nat) (n :: Nat) (bs :: Nat). 
                 KnownNat lstmSize => KnownNat vocSize => (KnownNat bs, KnownNat n) =>
                 String
                 -> T '[bs] Int32 -- lengths
                 -> EmbbeddingP vocSize 50 'B32
                 -> Tensor '[n, bs] Int32
                 -> Gen
                      ((FHTV '[ '[lstmSize, bs], '[lstmSize, bs], '[lstmSize, bs], '[lstmSize, bs]]),
                       Tensor '[n, lstmSize+lstmSize, bs] Float32)
encoder prefix lens embs input = do
  lstm1 <- parameter (prefix++"w1") lstmInitializer
  lstm2 <- parameter (prefix++"w2") lstmInitializer
  drp1 <- mkDropout (DropProb 0.2)
  rdrp1 <- mkDropouts (DropProb 0.2)
  rdrp2 <- mkDropouts (DropProb 0.2)
  -- todo: what to do about sentence length?
  (sFinal,h) <-
    (rnn (timeDistribute (embedding @50 @vocSize embs))
      .--.
     rnn (timeDistribute drp1)
      .--.
     (rnn             (onStates rdrp1 (lstm @lstmSize lstm1))
       .++.
      rnnBackwardCull' lens (onStates rdrp2 (lstm @lstmSize lstm2)))
     ) repeatZeros input
  h' <- assign h  -- will be used many times as input to attention model
  return (sFinal,h')
decoder :: forall (lstmSize :: Nat) (n :: Nat) (outVocabSize :: Nat) (bs :: Nat) (d::Nat).
                 KnownNat lstmSize => KnownNat d => (KnownNat bs, KnownNat outVocabSize, KnownNat n) =>
                 String
                 -> EmbbeddingP outVocabSize 50 'B32
                 -> T '[n, d, bs] Float32 -- todo: consider a larger size for the output string
                 -> (HList '[HTV Float32 '[ '[lstmSize, bs], '[lstmSize, bs]], HTV Float32 '[ '[lstmSize, bs], '[lstmSize, bs]]])
                 -> Tensor '[n, bs] Int32
                 -> Gen (Tensor '[n, outVocabSize, bs] Float32)
decoder prefix embs hs thoughtVectors startTarget = do
  -- note: for an intra-language translation the embeddings can be shared.
  projs <- parameter (prefix++"proj") denseInitialiser
  lstm1 <- parameter (prefix++"w1") lstmInitializer
  lstm2 <- parameter (prefix++"w2") lstmInitializer
  -- w1 <- parameter (prefix++"att1") glorotUniform
  -- w2 <- parameter (prefix++"att2") glorotUniform
  -- let attn = (luongAttention @64 (luongMultiplicativeScoring w1) w2 hs)
  --     initAttn = zeros
  drp1 <- mkDropout (DropProb 0.2)
  drp2 <- mkDropout (DropProb 0.2)
  rdrp1 <- mkDropouts (DropProb 0.2)
  rdrp2 <- mkDropouts (DropProb 0.2)
  (_sFinal,outFinal) <-
    (rnn (timeDistribute (embedding @50 @outVocabSize embs)
          .-.
          timeDistribute drp1
          .-.
          -- addAttentionWithFeedback attn
          
           ((onState rdrp1 (lstm @lstmSize lstm1))
             .-.
             timeDistribute drp2
             .-.
             (onState rdrp2 (lstm @lstmSize lstm2)))
          .-.
          (timeDistribute (dense projs))
         )) ({-I initAttn :* -}thoughtVectors) startTarget

     -- TODO: should we use the states for all layers as
     -- thoughtVectors? Or just the top one?
  return outFinal

seq2seq :: forall (vocSize :: Nat) (n :: Nat) (bs :: Nat).
                 KnownNat vocSize => 
                 (KnownNat bs, KnownNat n) =>
                 Tensor '[n, bs] Int32 ->
                 Tensor '[n, bs] Int32 ->
                 Gen (Tensor '[n, vocSize, bs] Float32)
seq2seq input outputPlusStart = do
  embs <- parameter "embs" embeddingInitializer
  (thought,h) <- encoder @256 @vocSize "enc" embs input
  decoder "dec" embs h thought outputPlusStart

-- TODO: beam search decoder. Perhaps best implemented outside of tensorflow?

trainModel :: forall vocSize len. KnownNat vocSize => KnownNat len => Gen (ModelOutput '[len, vocSize, 128] Float32)
trainModel = do
  sourceInput <- placeholder "src_in"
  targetInput <- placeholder "tgt_in"
  targetOutput <- placeholder "tgt_out"
  masks <- placeholder "tgt_weights"
  y_ <- seq2seq @vocSize @len @128 sourceInput targetInput
  timedCategorical masks y_ targetOutput

main :: IO ()
main = generateFile "model.py" (compileGen (defaultOptions {maxGradientNorm = Just 1}) (trainModel @15295 @20))

{-> main

-}



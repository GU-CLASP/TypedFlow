{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE NoStarIsType #-}
{-|
Module      : TypedFlow.Models.Transformer
Description : Topic models
Copyright   : (c) Jean-Philippe Bernardy, 2020
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental
-}


module TypedFlow.Models.Transformer where
import Prelude hiding (RealFrac(..))
import TypedFlow.TF
import TypedFlow.Abstract
import TypedFlow.Layers
import TypedFlow.Types
import TypedFlow.Types.Proofs ((?>), knownSum')
import GHC.TypeLits

-- Convention for type variables:
-- h = number of heads
-- e = embedding size
-- n = sequence length

average :: forall e. KnownNat e => T '[e] Float32 -> Scalar Float32
average = reduceMeanAll

-- | Normalise a vector. But add a small epsilon to avoid division by zero
normalizer :: forall e. KnownNat e => T '[e] Float32 -> T '[e] Float32
normalizer x = mapT (⊘ (sigma + epsilon)) xmu -- so the norm of result is almost 1
  where mu = average x
        xmu = mapT (⊝ mu) x  -- so the average of xmu is 0
        sigma = sqrt (average (square xmu)) -- the norm of xmu.
        epsilon = 0.001 -- ?

-- Informally:
-- mapT f x = vector y such that y_i = f (x_i) -- (the first axis)

dimAsFloat :: forall e. KnownNat e => Float
dimAsFloat = fromIntegral (knownNatVal (natSat @e))

-- | dot product attention on one key (k)
dotAttention1 :: forall e n. KnownNat e => KnownNat n
  => T '[e,n] Float32 -> T '[n,e] Float32 -> T '[e] Float32 -> T '[e] Float32
dotAttention1 q v k = v ∙ softmax0 (mapT (⊘ normFactor) (q ∙ k))
  where normFactor = constant (sqrt (dimAsFloat @e))

-- | dot product attention for every position
dotAttention :: forall n e. KnownNat n => KnownNat e
  => T '[n,e] Float32 -> T '[n,e] Float32 -> T '[n,e] Float32 -> T '[n,e] Float32
dotAttention v k q = mapT (dotAttention1 (transpose01 q) v) k

-- | h copies of a dense layer (the same for every copy).
multiheadLinearEncoder :: forall h e. KnownNat e => KnownNat h =>
  String -> Gen (T '[e] Float32 -> T '[h,e] Float32)
multiheadLinearEncoder name = do
  wv <- parameterDefault ("w_" ++ name)
  return $ \x -> reshape (wv # x)

multiheadSelfAttentionModule
  :: forall h n e. KnownNat n => KnownNat h => KnownNat e
  => String -> Gen (T '[n,e] Float32 -> T '[n,e] Float32)
multiheadSelfAttentionModule nm = do
  ev <- multiheadLinearEncoder @h ("v" ++ nm)
  eq <- multiheadLinearEncoder @h ("q" ++ nm)
  ek <- multiheadLinearEncoder @h ("k" ++ nm)
  w1 <- parameterDefault ("w1" ++ nm)
  -- w2 <- parameterDefault ("w2" ++ nm)
  return $ \x ->
    let v = transpose01 (mapT ev x)
        q = transpose01 (mapT eq x)
        k = transpose01 (mapT ek x)
        r :: T '[n,h,e] Float32
        r = transpose01 (zipWith3T dotAttention q k v)
        r' = mapT (dense @e w1 . reshape @'[h * e]) r
    in mapT ({-dense w2 . -}normalizer) (r' + x)
       -- x + mapT normalizer r'

multiheadSelfAttentionModuleDecoder
  :: forall h n e. KnownNat n => KnownNat h => KnownNat e
  => String -> Gen (T '[n,e] Float32 -> T '[n,e] Float32  -> T '[n,e] Float32)
multiheadSelfAttentionModuleDecoder nm = do
  ev <- multiheadLinearEncoder @h ("v" ++ nm)
  eq <- multiheadLinearEncoder @h ("q" ++ nm)
  ek <- multiheadLinearEncoder @h ("k" ++ nm)
  w1 <- parameterDefault ("w1" ++ nm)
  -- w2 <- parameterDefault ("w2" ++ nm)
  return $ \x    -- comes from decoder
            y    -- comes from encoder
           ->
    let k = transpose01 (mapT ek y)
        v = transpose01 (mapT ev x)
        q = transpose01 (mapT eq y)
        r :: T '[n,h,e] Float32
        r = transpose01 (zipWith3T dotAttention q k v)
        r' = mapT (dense @e w1 . reshape @'[h * e]) r
    in mapT ({-dense w2 . -}normalizer) (r' + x)
       -- x + mapT normalizer r'


feedForwardModule :: forall e. KnownNat e
  => String -> Gen (T '[e] Float32 -> T '[e] Float32)
feedForwardModule nm = do
  w1 :: DenseP 'B32 e e <- parameterDefault (nm ++ "w1")
  w2 <- parameterDefault (nm ++ "w2")
  return $ \x -> normalizer (x + (w2 # relu (w1 # x)))

encoderModule :: forall h n e. KnownNat n => KnownNat h => KnownNat e => DropProb 
  -> String -> T '[n,e] Float32 -> Gen (T '[n,e] Float32 -> T '[n,e] Float32)
encoderModule dropProb nm positionalTensor = do
  drp <- mkDropout dropProb
  selfAtt <- multiheadSelfAttentionModule @h (nm ++ "mh")
  ff <- feedForwardModule (nm ++ "ff")
  return (mapT ff . selfAtt . (+ positionalTensor) . drp)

positionalModuleSinCos :: forall n e. KnownNat e => KnownNat n => T '[n,e] Float32
positionalModuleSinCos = sin (transpose01 (broadcastT pos) * (broadcastT omega) + broadcastT phase)
  where pos = (cast (range @n @'B32)) :: T '[n] Float32
        phase = cast ((range @e @'B32) `floorMod` constant 2) * (constant pi/2) :: T '[e] Float32
        omega = constant (log 10000) * exp (constant (-2.0 / dimAsFloat @e) * cast (range @e @'B32))
        -- Note I'm not dividing the frequence by 2 because integer
        -- division isn't implemented. Should not have any consequence.

positionalModuleLearned :: KnownNat e => KnownNat n => Gen (T '[n,e] Float32)
positionalModuleLearned = do
  e <- parameterDefault "positional"
  return $ let EmbeddingP x = e in x

encoderStack :: forall h n e. KnownNat h => KnownNat n => KnownNat e
  => DropProb -> Int -> Gen (T '[n,e] Float32 -> T '[n,e] Float32)
encoderStack dropProb n = do
  p <- positionalModuleLearned
  encoders <- mapM (\i -> encoderModule @h dropProb ("enc" ++ show i) p) [1..n]
  return (foldr (.) id encoders) -- n-ary function composition

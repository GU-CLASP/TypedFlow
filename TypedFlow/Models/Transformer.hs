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
Copyright   : (c) Jean-Philippe Bernardy, 2019
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental
-}


module TypedFlow.Model.Transformer where
import Prelude hiding (RealFrac(..))
import TypedFlow.TF
import TypedFlow.Layers
import TypedFlow.Types
import TypedFlow.Types.Proofs ((?>), knownSum')
import TypedFlow.Learn
import GHC.TypeLits
import Data.Monoid ((<>))
import Data.Proxy



normalizer :: forall e. KnownNat e => T '[e] Float32 -> T '[e] Float32
normalizer x = mapT (⊘ (factor + epsilon)) x
  where mu = reduceMeanAll x
        factor = sqrt $ reduceMeanAll $ square (mapT (⊝ mu) x)
        epsilon = 000.1

dotAttention1 :: KnownNat e => KnownNat n => T '[e,n] Float32 -> T '[n,e] Float32 -> T '[e] Float32 -> T '[e] Float32
dotAttention1 q v k = v ∙ softmax0 (q ∙ k)

dotAttention :: forall n e. KnownNat n => KnownNat e
  => T '[n,e] Float32 -> T '[n,e] Float32 -> T '[n,e] Float32 -> T '[n,e] Float32
dotAttention v k q = mapT (dotAttention1 (transpose01 q) v) (k)

multiheadLinearEncoder :: forall h e. KnownNat e => KnownNat h =>
  String -> Gen (T '[e] Float32 -> T '[h,e] Float32)
multiheadLinearEncoder name = do
  wv <- parameterDefault ("w_" ++ name)
  return $ \x -> reshape (dense wv x)


multiheadSelfAttentionModule :: forall h n e. KnownNat n => KnownNat h => KnownNat e => Gen (T '[n,e] Float32 -> T '[n,e] Float32)
multiheadSelfAttentionModule = do
  ev <- multiheadLinearEncoder @h "v"
  eq <- multiheadLinearEncoder @h "q"
  ek <- multiheadLinearEncoder @h "k"
  w1 <- parameterDefault "w1"
  w2 <- parameterDefault "w2"
  return $ \x ->
    let v = transpose01 $ mapT ev x
        q = transpose01 $ mapT eq x
        k = transpose01 $ mapT ek x
        r :: T '[n,h,e] Float32
        r = transpose01 (zipWith3T dotAttention q k v)
        r' = mapT (dense @e w1 . reshape @'[h * e]) r
    in mapT (dense w2 . normalizer) (r' + x)

feedForwardModule :: forall e. KnownNat e => Gen (T '[e] Float32 -> T '[e] Float32)
feedForwardModule = do
  w1 <- parameterDefault "ff1"
  w2 <- parameterDefault "ff2"
  return $ \x -> normalizer (x + (w2 # relu (dense @e w1 x)))




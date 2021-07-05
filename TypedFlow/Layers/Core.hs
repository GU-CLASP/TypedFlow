{-|
Module      : TypedFlow.Layers.Core
Description : Core layers and combinators.
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental
-}
{-# LANGUAGE CPP #-}
#if __GLASGOW_HASKELL__ >= 806
{-# LANGUAGE NoStarIsType #-}
#endif
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE TypeInType #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
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
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE PatternSynonyms #-}

module TypedFlow.Layers.Core
  (
    -- * Dense
    DenseP(..), dense, (#),
    -- * Dropout
    DropProb(..), mkMask, mkDropout, mkDropouts,
    -- * Embedding
    EmbeddingP(..), embedding, 
    -- * Convolutional
    ConvP(..), conv, conv', {-convValid,-} maxPool1D, maxPool2D,
    glu
  )

where
import Prelude hiding (RealFrac(..))
import GHC.TypeLits
import TypedFlow.TF
import TypedFlow.Types
import TypedFlow.Types.Proofs
import TypedFlow.Abstract
import Control.Monad.State (gets)
import Data.Monoid ((<>))
---------------------
-- Linear functions


-- | A dense layer is a linear function form a to b: a transformation matrix and a bias.
data DenseP t a b = DenseP {denseWeights :: Tensor '[a,b] t
                           ,denseBiases  :: Tensor '[b] t}

-----------------------
-- Feed-forward layers

-- | Parameters for the embedding layers
newtype EmbeddingP numObjects embeddingSize t = EmbeddingP (Tensor '[numObjects, embeddingSize] t)

instance (KnownNat numObjects, KnownTyp b, KnownNat embeddingSize) => KnownTensors (EmbeddingP numObjects embeddingSize b) where
  travTensor f s (EmbeddingP p) = EmbeddingP <$> travTensor f s p

instance (KnownNat numObjects, KnownBits b, KnownNat embeddingSize) => ParamWithDefault (EmbeddingP numObjects embeddingSize ('Typ 'Float b)) where
  defaultInitializer = EmbeddingP <$> (noise $ UniformD (-0.05) 0.05)

instance (KnownNat numObjects, KnownBits b, KnownNat embeddingSize) => ParamWithDefault (EmbeddingP numObjects embeddingSize ('Typ 'Cmplx b)) where
  defaultInitializer = EmbeddingP <$> (mkComplex <$> (noise $ UniformD (-0.05) 0.05) <*> (noise $ UniformD (-0.05) 0.05))

-- | embedding layer
embedding :: ∀ embeddingSize numObjects t. KnownNat embeddingSize => KnownNat numObjects =>
             EmbeddingP numObjects embeddingSize t -> Tensor '[] Int32 -> Tensor '[embeddingSize] t
embedding (EmbeddingP param) input = gather param input



instance (KnownNat a, KnownNat b, KnownTyp t) => KnownTensors (DenseP t a b) where
  travTensor f s (DenseP x y) = DenseP <$> travTensor f (s<>"_w") x <*> travTensor f (s<>"_bias") y

instance (KnownNat n, KnownNat m, KnownFloat b) => ParamWithDefault (DenseP b n m) where
  defaultInitializer = DenseP <$> glorotUniform <*> (noise $ TruncatedNormalD 0.1)

-- | Dense layer (Apply a linear function)
(#), dense :: ∀m n t. KnownNat n => KnownNat m => KnownNumeric t => DenseP t n m -> Tensor '[n] t -> Tensor '[m] t
(DenseP weightMatrix bias) # v = (weightMatrix ∙ v) + bias

dense = (#)

-- | A drop probability. (This type is used to make sure one does not
-- confuse keep probability and drop probability)
data DropProb = DropProb Float

-- | Generate a dropout function. The mask applied by the returned
-- function will be constant for any given call to mkDropout.  See
-- 'noise' for the sampling behaviour.
mkDropout :: forall s t. KnownShape s => KnownFloat t => DropProb -> Gen (Tensor s t -> Tensor s t)
mkDropout d = (⊙) <$> mkMask d

-- | Generate a 0-1 mask with given probability, suitable for dropout,
-- or all ones if not in training phase. See 'noise' for the sampling
-- behaviour.
mkMask :: forall s t. KnownShape s => KnownFloat t => DropProb -> Gen (Tensor s t)
mkMask (DropProb dropProb) = do
  let keepProb = 1 - dropProb
  let isTraining = genTrainingPlaceholder
  r <- noise $ UniformD keepProb (1 + keepProb)
  return $ if_ isTraining
               (floor r ⊘ constant (knownAlgebraic @t $ realToFrac keepProb))
               ones

newtype EndoTensor t s = EndoTensor (Tensor s t -> Tensor s t)

-- | Generate a dropout function for an heterogeneous tensor vector.
mkDropouts :: KnownFloat t => KnownLen shapes => All KnownShape shapes => DropProb -> Gen (HTV t shapes -> HTV t shapes)
mkDropouts d = appEndoTensor <$> mkDropouts' typeSList where
   mkDropouts' :: forall shapes t. KnownFloat t => All KnownShape shapes =>
                  SList shapes -> Gen (NP (EndoTensor t) shapes)
   mkDropouts' Unit = return Unit
   mkDropouts' (_ :* rest) = do
     x <- mkDropout d
     xs <- mkDropouts' rest
     return (EndoTensor x :* xs)

   appEndoTensor :: NP (EndoTensor t) s -> HTV t s -> HTV t s
   appEndoTensor Unit Unit = Unit
   appEndoTensor (EndoTensor f :* fs) (F x :* xs) = F (f x) :* appEndoTensor fs xs


------------------------
-- Convolutional layers

data ConvP t outChannels inChannels filterSpatialShape
  = ConvP (T (filterSpatialShape ++ '[inChannels,outChannels]) t)
          (T '[outChannels] t)

instance (KnownNat outChannels,KnownNat inChannels, KnownShape filterSpatialShape, KnownFloat t) =>
  ParamWithDefault (ConvP t outChannels inChannels filterSpatialShape) where
  defaultInitializer = prodHomo @filterSpatialShape @'[inChannels, outChannels] #>
                       prodAssoc @(Product filterSpatialShape) @inChannels @outChannels #>
                       knownAppend @filterSpatialShape @'[inChannels,outChannels] ?>
                       knownProduct @filterSpatialShape ?>
                       ConvP <$> (reshape <$> i) <*> pure (knownAlgebraic @t (constant 0.1))
    where i :: Gen (T '[Product filterSpatialShape*inChannels,outChannels] t)
          i = knownProduct @filterSpatialShape ?> glorotUniform

instance (KnownNat outChannels,KnownNat inChannels, KnownShape filterSpatialShape, KnownAlgebraic t) =>
  KnownTensors (ConvP t outChannels inChannels filterSpatialShape) where
  travTensor f s (ConvP x y) = knownAppend @filterSpatialShape @'[inChannels,outChannels] ?>
          (ConvP <$> travTensor f (s<>"_filters") x <*> travTensor f (s <> "_biases") y)

-- | Size-preserving convolution layer
conv' :: forall s outChannels filterSpatialShape inChannels t.
               KnownShape s => KnownNat inChannels => KnownNat outChannels => KnownShape filterSpatialShape => KnownAlgebraic t
            => Length filterSpatialShape <= 3
            => Length filterSpatialShape ~ Length s
            => ConvP t outChannels inChannels filterSpatialShape
            -> T (s ++ '[inChannels]) t
            -> T (s ++ '[outChannels]) t
conv' (ConvP filters bias) input = mapTT @s (+bias) (convolution @outChannels @filterSpatialShape @inChannels @s input filters)



conv :: forall outChannels filterSpatialShape inChannels s t.
               KnownShape s => KnownNat inChannels => KnownNat outChannels => KnownShape filterSpatialShape => KnownAlgebraic t
            => Length filterSpatialShape <= 3
            => (Length filterSpatialShape + 1) ~ Length s -- The ranks must match, but not necessarily the dimensions
            => (Last s ~ outChannels)
            => ConvP t outChannels inChannels filterSpatialShape
            -> T (Init s ++ '[inChannels]) t
            -> T s t
conv = initLast' @s #>
       incrPos @(Length filterSpatialShape) #>
       lengthInit (typeSList @s) #>
       incrCong @(Length filterSpatialShape) @(Length (Init s)) #>
       knownInit @s ?>
       conv' @(Init s)


-- -- | Convolution layers with no padding (applying the filter only on
-- -- positions where the input is fully defined, aka "VALID" in
-- -- tensorflow.)
-- convValid :: forall outChannels filterSpatialShape inChannels s t.
--                   ((1 + Length filterSpatialShape) ~ Length s,
--                    Length filterSpatialShape <= 3,
--                    KnownLen filterSpatialShape) -- the last dim of s is the batch size
--           => ConvP t outChannels inChannels filterSpatialShape -- ^ Parameters
--           -> T ('[inChannels] ++ AddSpatialDims s filterSpatialShape) ('Typ 'Float t) -- ^ input
--           -> (T ('[outChannels] ++ s) ('Typ 'Float t))
-- convValid (ConvP filters bias) input = convolutionValid input filters + bias

-- | Gated Linear Unit
-- See: Language Modeling with Gated Convolutional Networks
-- https://arxiv.org/pdf/1612.08083.pdf
glu :: forall n t. KnownBits t => KnownNat n => T '[n+n] ('Typ 'Float t) -> T '[n] ('Typ 'Float t)
glu x = plusMono @n @n #> knownPlus @n @n ?>
        let gate, h :: T '[n] ('Typ 'Float t)
            gate = slice0 @0 @n x
            h =  termCancelation @n @n #> slice0 @n @(n+n) x
        in sigmoid gate ⊙ h

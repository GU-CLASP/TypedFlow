{-|
Module      : TypedFlow.Layers.Core
Description : Core layers and combinators.
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental
-}
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

module TypedFlow.Layers.Core where

import Prelude hiding (tanh,Num(..),Floating(..),floor)
import qualified Prelude
import GHC.TypeLits
-- import Text.PrettyPrint.Compact (float)
import TypedFlow.TF
import TypedFlow.Types
import Control.Monad.State (gets)
-- import Data.Type.Equality
-- import Data.Kind (Type,Constraint)
import Data.Monoid ((<>))
---------------------
-- Linear functions


-- type (a ⊸ b) = DenseP Float32 a b

-- | A dense layer is a linear function form a to b: a transformation matrix and a bias.
data DenseP t a b = DenseP {denseWeights :: Tensor '[a,b] (Flt t)
                           ,denseBiases  :: Tensor '[b] (Flt t)}

-----------------------
-- Feed-forward layers

-- | Parameters for the embedding layers
newtype EmbbeddingP numObjects embeddingSize t = EmbbeddingP (Tensor '[numObjects, embeddingSize] ('Typ 'Float t))

instance (KnownNat numObjects, KnownBits b, KnownNat embeddingSize) => KnownTensors (EmbbeddingP numObjects embeddingSize b) where
  travTensor f s (EmbbeddingP p) = EmbbeddingP <$> travTensor f s p

instance (KnownNat numObjects, KnownBits b, KnownNat embeddingSize) => ParamWithDefault (EmbbeddingP numObjects embeddingSize b) where
  defaultInitializer = EmbbeddingP (randomUniform (-0.05) 0.05)

-- | embedding layer
embedding :: ∀ embeddingSize numObjects batchSize t.
             EmbbeddingP numObjects embeddingSize t -> Tensor '[batchSize] Int32 -> Tensor '[embeddingSize,batchSize] ('Typ 'Float t)
embedding (EmbbeddingP param) input = gather @ '[embeddingSize] (transpose param) input

instance (KnownNat a, KnownNat b, KnownBits t) => KnownTensors (DenseP t a b) where
  travTensor f s (DenseP x y) = DenseP <$> travTensor f (s<>"_w") x <*> travTensor f (s<>"_bias") y

instance (KnownNat n, KnownNat m, KnownBits b) => ParamWithDefault (DenseP b n m) where
  defaultInitializer = DenseP glorotUniform (truncatedNormal 0.1)

-- | Apply a linear function
(#) :: DenseP t a b -> T '[a,batchSize] (Flt t) -> Tensor '[b,batchSize] (Flt t)
(DenseP weightMatrix bias) # v = (weightMatrix ∙ v) + bias

-- | Dense layer
dense :: ∀m n batchSize t. DenseP t n m -> Tensor '[n, batchSize] (Flt t) -> Tensor '[m, batchSize] (Flt t)
dense lf t = lf # t

-- | A drop probability. (This type is used to make sure one does not
-- confuse keep probability and drop probability)
data DropProb = DropProb Float

-- | Generate a dropout function. The mask applied by the returned
-- function will be constant for any given call to mkDropout. This
-- behavior allows to use the same mask in the several steps of an
-- RNN.
mkDropout :: forall s t. KnownShape s => KnownBits t => DropProb -> Gen (Tensor s ('Typ 'Float t) -> Tensor s ('Typ 'Float t))
mkDropout (DropProb dropProb) = do
  let keepProb = 1.0 Prelude.- dropProb
  isTraining <- gets genTrainingPlaceholder
  mask <- assign (if_ isTraining
                   (floor (randomUniform keepProb (1 Prelude.+ keepProb)) ⊘ constant keepProb)
                   ones)
  return (mask ⊙)

newtype EndoTensor t s = EndoTensor (Tensor s t -> Tensor s t)

-- | Generate a dropout function for an heterogeneous tensor vector.
mkDropouts :: KnownBits t => KnownLen shapes => All KnownShape shapes => DropProb -> Gen (HTV ('Typ 'Float t) shapes -> HTV ('Typ 'Float t) shapes)
mkDropouts d = appEndoTensor <$> mkDropouts' shapeSList where
   mkDropouts' :: forall shapes t. KnownBits t => All KnownShape shapes =>
                  SList shapes -> Gen (NP (EndoTensor ('Typ 'Float t)) shapes)
   mkDropouts' LZ = return Unit
   mkDropouts' (LS _ rest) = do
     x <- mkDropout d
     xs <- mkDropouts' rest
     return (EndoTensor x :* xs)

   appEndoTensor :: NP (EndoTensor t) s -> HTV t s -> HTV t s
   appEndoTensor Unit Unit = Unit
   appEndoTensor (EndoTensor f :* fs) (F x :* xs) = F (f x) :* appEndoTensor fs xs


------------------------
-- Convolutional layers

data ConvP t outChannels inChannels filterSpatialShape
  = ConvP (T ('[outChannels,inChannels] ++ filterSpatialShape)  ('Typ 'Float t)) (T '[outChannels] ('Typ 'Float t))

instance (KnownNat outChannels,KnownNat inChannels, KnownShape filterSpatialShape, KnownBits t) =>
  ParamWithDefault (ConvP t outChannels inChannels filterSpatialShape) where
  defaultInitializer = ConvP (truncatedNormal 0.1) (constant 0.1)

instance (KnownNat outChannels,KnownNat inChannels, KnownShape filterSpatialShape, KnownBits t) =>
  KnownTensors (ConvP t outChannels inChannels filterSpatialShape) where
  travTensor f s (ConvP x y) = ConvP <$> travTensor f (s<>"_filters") x <*> travTensor f (s <> "_biases") y

-- | Size-preserving convolution layer
conv :: forall outChannels filterSpatialShape inChannels s t.
                  ((1 + Length filterSpatialShape) ~ Length s,
                   Length filterSpatialShape <= 3,
                   KnownLen filterSpatialShape) => -- the last dim of s is the batch size
                  ConvP t outChannels inChannels filterSpatialShape ->
                  T ('[inChannels] ++ s) ('Typ 'Float t) -> (T ('[outChannels] ++ s) ('Typ 'Float t))
conv (ConvP filters bias) input = convolution input filters + bias


-- | 2 by 2 maxpool layer.
maxPool2D :: forall stridex (stridey::Nat) batch height width channels t.
             (KnownNat stridex, KnownNat stridey) =>
             T '[channels,width*stridex,height*stridex,batch] (Flt t) -> T '[channels,width,height,batch] (Flt t)
maxPool2D (T value) = T (funcall "tf.nn.max_pool" [value
                                                  ,showShape @'[1,stridex,stridey,1]
                                                  ,showShape @'[1,stridex,stridey,1]
                                                  ,named "padding" (str "SAME") ])


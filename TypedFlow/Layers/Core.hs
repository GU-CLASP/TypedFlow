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

module TypedFlow.Layers.Core
  (
    -- * Dense
    DenseP(..), dense, (#),
    -- * Dropout
    DropProb(..), mkDropout, mkDropouts,
    -- * Embedding
    EmbeddingP(..), embedding, 
    -- * Convolutional
    ConvP(..), conv, {-convValid,-} maxPool1D, maxPool2D)

where

import Prelude hiding (tanh,Num(..),Floating(..),floor)
import qualified Prelude
import GHC.TypeLits
-- import Text.PrettyPrint.Compact (float)
import TypedFlow.TF
import TypedFlow.Types
import TypedFlow.Python (assign)
import TypedFlow.Abstract
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
newtype EmbeddingP numObjects embeddingSize t = EmbeddingP (Tensor '[numObjects, embeddingSize] ('Typ 'Float t))

instance (KnownNat numObjects, KnownBits b, KnownNat embeddingSize) => KnownTensors (EmbeddingP numObjects embeddingSize b) where
  travTensor f s (EmbeddingP p) = EmbeddingP <$> travTensor f s p

instance (KnownNat numObjects, KnownBits b, KnownNat embeddingSize) => ParamWithDefault (EmbeddingP numObjects embeddingSize b) where
  defaultInitializer = EmbeddingP (randomUniform (-0.05) 0.05)

-- | embedding layer
embedding :: ∀ embeddingSize numObjects t. KnownNat embeddingSize => KnownNat numObjects =>
             EmbeddingP numObjects embeddingSize t -> Tensor '[] Int32 -> Tensor '[embeddingSize] ('Typ 'Float t)
embedding (EmbeddingP param) input = gather param input



instance (KnownNat a, KnownNat b, KnownBits t) => KnownTensors (DenseP t a b) where
  travTensor f s (DenseP x y) = DenseP <$> travTensor f (s<>"_w") x <*> travTensor f (s<>"_bias") y

instance (KnownNat n, KnownNat m, KnownBits b) => ParamWithDefault (DenseP b n m) where
  defaultInitializer = DenseP glorotUniform (truncatedNormal 0.1)

-- | Dense layer (Apply a linear function)
(#), dense :: ∀m n t. KnownNat n => KnownNat m => KnownBits t => DenseP t n m -> Tensor '[n] (Flt t) -> Tensor '[m] (Flt t)
(DenseP weightMatrix bias) # v = (weightMatrix ∙ v) + bias

dense = (#)

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
mkDropouts d = appEndoTensor <$> mkDropouts' typeSList where
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
  = ConvP (T (filterSpatialShape ++ '[inChannels,outChannels])  ('Typ 'Float t)) (T '[outChannels] ('Typ 'Float t))

instance (KnownNat outChannels,KnownNat inChannels, KnownShape filterSpatialShape, KnownBits t) =>
  ParamWithDefault (ConvP t outChannels inChannels filterSpatialShape) where
  defaultInitializer = prodHomo @filterSpatialShape @'[inChannels, outChannels] $
                       prodAssoc @(Product filterSpatialShape) @inChannels @outChannels $
                       knownAppend @filterSpatialShape @'[inChannels,outChannels] $
                       knownProduct @filterSpatialShape $
                       ConvP (reshape i) (constant 0.1)
    where i :: T '[Product filterSpatialShape*inChannels,outChannels] (Flt t)
          i = knownProduct @filterSpatialShape glorotUniform

instance (KnownNat outChannels,KnownNat inChannels, KnownShape filterSpatialShape, KnownBits t) =>
  KnownTensors (ConvP t outChannels inChannels filterSpatialShape) where
  travTensor f s (ConvP x y) = knownAppend @filterSpatialShape @'[inChannels,outChannels] $
          ConvP <$> travTensor f (s<>"_filters") x <*> travTensor f (s <> "_biases") y

-- | Size-preserving convolution layer
conv' :: forall outChannels filterSpatialShape inChannels s t.
               KnownShape s => KnownNat inChannels => KnownNat outChannels => KnownShape filterSpatialShape => KnownBits t
            => Length filterSpatialShape <= 3
            => Length filterSpatialShape ~ Length s
            => ConvP t outChannels inChannels filterSpatialShape
            -> T (s ++ '[inChannels]) ('Typ 'Float t)
            -> T (s ++ '[outChannels]) ('Typ 'Float t)
conv' (ConvP filters bias) input = mapTT @s (+bias) (convolution @outChannels @filterSpatialShape @inChannels @s input filters)



conv :: forall outChannels filterSpatialShape inChannels s t.
               KnownShape s => KnownNat inChannels => KnownNat outChannels => KnownShape filterSpatialShape => KnownBits t
            => Length filterSpatialShape <= 3
            => (Length filterSpatialShape+1) ~ Length s
            => (Last s ~ outChannels)
            => ConvP t outChannels inChannels filterSpatialShape
            -> T (Init s ++ '[inChannels]) ('Typ 'Float t)
            -> T s ('Typ 'Float t)
conv = initLast @s $
        incrPos @(Length filterSpatialShape) $
        lengthInit @s $
        incrCong @(Length filterSpatialShape) @(Length (Init s)) $
        knownInit @s $ 
        conv' @outChannels @filterSpatialShape @inChannels @(Init s)

-- warning: [-Wdeferred-type-errors]
--     • Could not deduce: Length filterSpatialShape ~ Length (Init s)
--         arising from a use of ‘conv’
--       from the context: (KnownShape s,
--                          KnownNat inChannels,
--                          KnownNat outChannels,
--                          KnownShape filterSpatialShape,
--                          KnownBits t,
--                          Length filterSpatialShape <= 3,
--                          (Length filterSpatialShape + 1) ~ Length s,
--                          Last s ~ outChannels)
--         bound by the type signature for:
--                    conv' :: (KnownShape s, KnownNat inChannels, KnownNat outChannels,
--                              KnownShape filterSpatialShape, KnownBits t,
--                              Length filterSpatialShape <= 3,
--                              (Length filterSpatialShape + 1) ~ Length s,
--                              Last s ~ outChannels) =>
--                             ConvP t outChannels inChannels filterSpatialShape
--                             -> T (Init s ++ '[inChannels]) ('Typ 'Float t)
--                             -> T s ('Typ 'Float t)
--         at /tmp/dante9507pzG.hs:(168,1)-(175,34)
--       or from: (Init s ++ '[Last s]) ~ s
--         bound by a type expected by the context:
--                    (Init s ++ '[Last s]) ~ s =>
--                    ConvP t outChannels inChannels filterSpatialShape
--                    -> T (Init s ++ '[inChannels]) ('Typ 'Float t)
--                    -> T s ('Typ 'Float t)
--         at /tmp/dante9507pzG.hs:(176,9)-(179,67)
--       or from: 0 < (Length filterSpatialShape + 1)
--         bound by a type expected by the context:
--                    (0 < (Length filterSpatialShape + 1)) =>
--                    ConvP t outChannels inChannels filterSpatialShape
--                    -> T (Init s ++ '[inChannels]) ('Typ 'Float t)
--                    -> T s ('Typ 'Float t)
--         at /tmp/dante9507pzG.hs:(177,9)-(179,67)
--       or from: (Length (Init s) + 1) ~ Length s
--         bound by a type expected by the context:
--                    (Length (Init s) + 1) ~ Length s =>
--                    ConvP t outChannels inChannels filterSpatialShape
--                    -> T (Init s ++ '[inChannels]) ('Typ 'Float t)
--                    -> T s ('Typ 'Float t)
--         at /tmp/dante9507pzG.hs:(178,9)-(179,67)
--       NB: ‘Length’ is a type function, and may not be injective
--     • In the second argument of ‘($)’, namely
--         ‘conv @outChannels @filterSpatialShape @inChannels @(Init s)’
--       In the second argument of ‘($)’, namely
--         ‘lengthInit @s
--          $ conv @outChannels @filterSpatialShape @inChannels @(Init s)’
--       In the second argument of ‘($)’, namely
--         ‘incrPos @(Length filterSpatialShape)
--          $ lengthInit @s
--            $ conv @outChannels @filterSpatialShape @inChannels @(Init s)’
--     • Relevant bindings include
--         conv' :: ConvP t outChannels inChannels filterSpatialShape
--                  -> T (Init s ++ '[inChannels]) ('Typ 'Float t)
--                  -> T s ('Typ 'Float t)
--           (bound at /tmp/dante9507pzG.hs:176:1)

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



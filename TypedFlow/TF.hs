{-# LANGUAGE InstanceSigs #-}
{-|
Module      : TypedFlow.TF
Description : Binding to tensorflow functions
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental

This module provides direct access to the most commonly used
TensorFlow functions. Higher-level functions are not defined here.
-}

{-# LANGUAGE LambdaCase #-}
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

module TypedFlow.TF (
  -- * Variables, Parameters
  -- ** Parameters
  parameter',
  parameter,
  parameterDefault,
  ParamWithDefault(..),
  -- getParameters,
  -- ** Persistent variables
  persistent,
  modifyPersistent,
  -- ** Placeholders and outputs
  placeholder,
  peekAt,
  -- peekAtMany,
  -- * Operations
  -- ** Constants
  zeros,
  ones,
  eye,
  constant,
  -- ** indexwise unary operators
  round, sigmoid, relu, floor, square,
  -- ** Indexwise binary operators
  addN, (⊕), (⊝), (⊙), (⊘), equal,
  minT, maxT,
  -- ** Products
  (∙), (·), matmul,
  -- ** Reducers
  reduceMeanAll, reduceSumAll, reduceMinAll, reduceMaxAll,
  reduceSum, reduceMean, reduceMin, reduceMax,
  -- argmax,
  argmax0, argmax1,
  softmax0, softmax1,
  -- ** Gradients
  -- grad,
  -- clipByGlobalNorm,
  clipByValue,
  -- ** Indexing
  last0, nth0, nth0', lookupT, gather, range,
  -- ** Split and concatenate
  slice, slice0, slice1,
  litStack0,
  stack0, unstack0,
  stack1,
  concatT, concat0, concat1,
  consT0, snocT0, tailT0, initT0,
  -- ** Reshaping
  expandDim,
  expandDim0, squeeze0,
  expandDim1, 
  flatten2, flatten3, flatten12, flattenN2,
  inflate2, inflate3, inflate12,
  reshape, flattenAll, inflateAll,
  -- ** Transposition
  transposeN, transposeN', transpose01, transposeN01,
  -- ** Sequences
  sequenceMask,
  -- ** Convolutions
  convolution,
  -- ** Misc
  norm, normalize,
  stopGradient,
  cast,
  oneHot0, oneHot1,
  -- ** Testing conditions
  if_, where_, lessThan,
  -- * Contrib
  -- ** Mapping
  mapT, zipWithT, zipWith3T,
  mapTT, zipWithTT,
  -- ** Losses
  sigmoidCrossEntropyWithLogits,
  softmaxCrossEntropyWithLogits,
  sparseSoftmaxCrossEntropyWithLogits,
  -- ** Initializers
  noise,
  Distribution(..),
  varianceScaling, glorotUniform,

  -- ** Heterogeneous vectors
  repeatT, KnownTensors(..),

  -- ** Heterogeneous heterogeneous vectors
  repeatHT
  ) where

import Prelude hiding (RealFrac(..))
import GHC.TypeLits
import Data.Proxy
import TypedFlow.Types
import TypedFlow.Types.Proofs
import Control.Monad (when)
import Control.Monad.State
import TypedFlow.Abstract

-- | Repeat a flexible-shape constant vector to form a heterogeneous tensor vector.
repeatT :: forall (ss :: [Shape]) t. All KnownShape ss => KnownLen ss =>
           (forall s. KnownShape s => T s t) -> HTV t ss
repeatT f = zs (typeSList @ss)
  where zs :: forall (s :: [Shape]). All KnownShape s => SList s -> HTV t s
        zs Unit = Unit
        zs (_ :* n) = F f :* zs n

-- | Repeat a flexible-shape constant vector to form a heterogeneous tensor vector.
repeatHT :: forall ss. All KnownPair ss => KnownLen ss =>
           (forall s t. KnownShape s => KnownTyp t => T s t) -> HHTV ss
repeatHT f = zs (typeSList @ss)
  where zs :: forall s. All KnownPair s => SList s -> HHTV s
        zs Unit = Unit
        zs (_ :* n) = Uncurry f :* zs n

-- | Declare a parameter to optimize. The shape of parameter should
-- not depend on dimensions which can change between runs, such as the
-- batch size.
parameter' :: ∀ (shape :: Shape) t. (KnownTyp t,KnownShape shape) => String -> T shape t -> Gen (T shape t)
parameter' = persistent True


newParameter :: MonadState GState m => VarInfo -> m ()
newParameter p =   modify $ \GState{..} -> GState{genParams = p:genParams,..}

-- TODO: use a different type for persistent?
-- | Declare variable which persists between calls to session.run.
persistent :: ∀ (shape :: Shape) t. (KnownTyp t,KnownShape shape) => Bool -> String -> T shape t -> Gen (T shape t)
persistent trainable name initial = do
  result <- T . Variable <$> GPVariable trainable name initial
  when trainable (newParameter (VarInfo name (typeSShape @shape) (typeSTyp @t) result))
  peekAt name result
  return result

placeholder :: ∀ (shape :: Shape) t. (KnownTyp t,KnownShape shape) => String -> Gen (T shape t)
placeholder n = do
  x <- GPPlaceholder typeSShape typeSTyp n
  peekAt n (T (Variable x))
  return (T (Variable x))

-- | Modify a mutable tensor. Attention: for the assignment to happen,
-- the resulting tensor must be evaluated!
modifyPersistent :: (KnownShape s,KnownTyp t) => T s t -> T s t -> Gen (T s t)
modifyPersistent v x = (GPModify v x)


-- | Name a tensor so that it is made available for session.run.
peekAt :: forall s t. (KnownShape s,KnownTyp t) => String -> Tensor s t -> Gen ()
peekAt name v =  modify $ \GState{..} -> GState{genPeeks = p:genPeeks,..}
  where p :: VarInfo
        p = (VarInfo name (typeSShape @s) (typeSTyp @t) v)

-- type family AddSpatialDims xs ys where
--   AddSpatialDims '[x] '[] = '[x]
--   AddSpatialDims (x ': xs) (y ': ys) = (x+(y-1)) ': AddSpatialDims xs ys

-- -- | Convolution operation with no padding (applying the filter only on positions where the input is fully defined)
-- convolutionValid :: forall outputChannels filterSpatialShape inChannels s t.
--                KnownLen filterSpatialShape
--             => Length filterSpatialShape <= 3
--             => ((1 + Length filterSpatialShape) ~ Length s) -- the last dim of s is the batch size
--             => T (inChannels ': AddSpatialDims s filterSpatialShape) t -- ^ input tensor (batched)
--             -> T ('[outputChannels,inChannels] ++ filterSpatialShape) t -- ^ filters
--             -> T (outputChannels ': s) t
-- convolutionValid = untypedConvolution "VALID"

-- poolNC :: forall dim s inputSpatialShape channels batchSize t.
--                   (inputSpatialShape ~ Take dim s, '[batchSize] ~ Drop dim s) =>
--                   T ('[channels] ++ s) t ->
--                   Vec dim  -> String -> String -> 
--                   T ('[channels] ++ s) t
-- poolNC (T input) windowShape poolingType padding =
--    T (funcall "tf.nn.pool" [input,list (map float (vecToList windowShape)),text poolingType,text padding,named "data_format" (text "NWC")])

-- Difficulty: relate windowSize, inputSpatialShape, outputSpatialShape




---------------------------
-- Contrib
data VarianceScaleMode = VSFanIn | VSFanOut | VSAvg
data Distrib = NormalDistr | UniformDistr

-- | Random tensor with variance scaling according to deeplearning lore.
varianceScaling :: forall inDim outDim t. KnownNat inDim => (KnownNat outDim, KnownBits t) =>
   Float -> VarianceScaleMode -> Distrib -> Gen (Tensor '[inDim,outDim] ('Typ 'Float t))
varianceScaling factor mode distr = noise $ case distr of
                                   UniformDistr -> UniformD (-limit) limit
                                   NormalDistr -> TruncatedNormalD limit
  where
    fan_in = fromIntegral (natVal (Proxy @inDim))
    fan_out = fromIntegral (natVal (Proxy @outDim))
    n = max 1 $ case mode of
                  VSFanIn -> fan_in
                  VSFanOut -> fan_out
                  VSAvg -> (fan_in + fan_out) / 2
    limit = sqrt ((case distr of NormalDistr -> 1.3; UniformDistr -> 3) * factor / n)


glorotUniform :: forall inDim outDim t. KnownNat inDim => (KnownNat outDim, KnownBits t) => Gen (Tensor '[outDim,inDim] ('Typ 'Float t))
glorotUniform = varianceScaling 1 VSAvg UniformDistr

-- | 'cons' an element and an array (in the first dimension)
consT0 :: forall n s t. KnownTyp t => KnownShape s => KnownNat n =>  T s t -> T (n ': s) t -> T (n+1 ': s) t
consT0 x xs = plusComm @1 @n #> concat0 (expandDim0 x) xs

-- | 'snoc' an element and an array (in the first dimension)
snocT0 :: forall n s t. KnownTyp t => KnownShape s => KnownNat n =>  KnownLen s => T (n ': s) t -> T s t -> T (n+1 ': s) t
snocT0 xs x = concat0 xs (expandDim0 x)

tailT0 :: forall n s t. KnownTyp t => KnownShape s => KnownNat n =>  T (n+1 ': s) t -> T (n ': s) t
tailT0 xs = incrPos @n              #> -- 0 < n+1
            plusMinusAssoc @n @1 @1 #> -- (n+1) - 1 = -- n+ (1 - 1)
            slice0 @1 @(n+1) xs

initT0 :: forall n s t. KnownTyp t => KnownShape s => KnownNat n =>  T (n+1 ': s) t -> T (n ': s) t
initT0 xs = plusMono @n @1 #> -- n <= n+1
            slice0 @0 @n xs

----------------
-- Helpers

-- | Product of a matrix of weights with a vector.
(∙) :: (KnownNumeric t, KnownNat cols, KnownNat rows, KnownTyp t) => Tensor '[cols, rows] t -> Tensor '[cols] t -> Tensor '[rows] t
m ∙ v = squeeze0 (matmul (expandDim0 v) m)
infixl 7 ∙

-- | Dot product between two vectors.
(·) :: ∀ n t. (KnownNumeric t, KnownNat n) =>
  Tensor '[n] t -> Tensor '[n] t -> Tensor '[] t
x · y = reduceSum0 (x ⊙ y)
infixl 7 ·


norm :: KnownBits t => KnownNat n
     => T '[n] (Flt t) -> Scalar (Flt t)
norm = sqrt . reduceSumAll . square

normalize :: (KnownNat n, KnownBits t) =>
                   T '[n] (Flt t) -> T '[n] (Flt t)
normalize v = mapT (/ (norm v + epsilon)) v
  where epsilon = 1.0e-8

-------------------------
-- Generic parameters

-- | Create a parameter and initialize it with a suitable default for its type. Control the exact initializer using 'parameter'.
parameterDefault :: forall p. ParamWithDefault p => String -> Gen p
parameterDefault name = parameter name defaultInitializer

-- | Create a parameter.
parameter :: forall p. KnownTensors p => String -> Gen p -> Gen p
parameter s p = do
  x <- p
  travTensor parameter' s x

-- flattenHTV :: KnownTyp t => All KnownShape xs => HTV t xs -> Tensor '[Sum (Ap (FMap CProduct) xs)] t
-- flattenHTV Unit = zeros
-- flattenHTV (F x :* xs) = concat0 (flattenAll x) (flattenHTV xs)

-- class CProduct (xs :: [Nat])
-- instance Fun CProduct where type Ap CProduct xs = Product xs

-- inflateHTV :: ∀ xs s t. (All KnownShape xs, KnownLen s, KnownLen xs) =>
--           Tensor '[Sum (Ap (FMap CProduct) xs)] t -> Gen (HTV t xs)
-- inflateHTV (T x) = do
--   v <- newVar
--   gen (v <> text " = " <> funcall "tf.split" [x, showShape' (prodshape @xs shapeSList), text "axis=0"])
--   return (mkArr @xs 0 shapeSList  v)
--   where mkArr :: forall zs. All KnownShape zs => Int -> SList zs -> DOC -> HTV t zs
--         mkArr _ LZ _ = Unit
--         mkArr i (LS _ n) v = F (unsafeReshape (T (v <> brackets (int i)) )):* mkArr (succ i) n v
--         prodshape :: forall zs. All KnownShape zs => SList zs -> [Integer]
--         prodshape LZ = []
--         prodshape (LS xx xs) = product (shapeToList' (shapeSListProxy xx)) : prodshape xs


-- -- | Gradient of wrt. given parameters.
-- grad' :: KnownLen xs => T s Float32 -> HHTV xs -> Gen (HHTV xs)
-- grad' (T y) vars = do
--  v <- newVar
--  v <-- funcall "tf.gradients" [y, list (htoList (hmap (\(Uncurry (T x)) -> K x) vars)) ]
--  return (mkArr 0 shapeSList v)
--   where mkArr :: forall xs. Int -> SList xs -> DOC -> HHTV xs
--         mkArr _ LZ _ = Unit
--         mkArr i (LS _ n) v = Uncurry (T (v <> brackets (int i))) :* mkArr (succ i) n v

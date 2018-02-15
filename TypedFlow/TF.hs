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
  getParameters,
  -- ** Persistent variables
  persistent,
  modifyPersistent,
  -- ** Placeholders and outputs
  placeholder,
  peekAt,
  peekAtMany,
  -- * Operations
  -- ** Constants
  zeros,
  ones,
  eye,
  constant,
  -- ** indexwise unary operators
  round, sqrt, sigmoid, tanh, log, relu, floor, negate, square,
  -- ** Indexwise binary operators
  add, addN, (+), (/), (⊕), (⊝), (⊙), (⊘), equal,
  -- ** Products
  (∙), (·), matmul,
  -- ** Reducers
  reduceMeanAll, reduceSumAll, reduceMaxAll,
  reduceSum, reduceMean, reduceMax,
  argmax, argmax0, argmax1,
  softmax0, softmax1,
  -- ** Gradients
  grad,
  clipByGlobalNorm,
  clipByValue,
  -- ** Indexing
  last0, nth0, nth0', gather,
  -- ** Split and concatenate
  split0, slice, slice1,
  stack0, unstack0, stackN,
  stack1,
  concatT, concat0, concat1,
  -- ** Reshaping
  expandDim,
  expandDim0, squeeze0,
  expandDim1, squeeze1,
  flatten2, flatten3, flatten12, flattenN2,
  inflate2, inflate3, inflate12,
  broadcast0, broadcastN,
  reshape, flattenAll, inflateAll,
  -- ** Transposition
  transpose, transposeN, transposeN', transpose01, transposeN01,
  -- ** Sequences
  reverseSequences, sequenceMask,
  -- ** Convolutions
  convolution, 
  -- ** Misc
  cast,
  oneHot, oneHot0, oneHot1,
  -- ** Testing conditions
  if_, where_,
  -- * Contrib
  -- ** Mapping
  mapT, mapTN, zipWithT, zipWithTN,
  consT0, snocT0,
  -- ** Losses
  sigmoidCrossEntropyWithLogits,
  softmaxCrossEntropyWithLogits,
  sparseSoftmaxCrossEntropyWithLogits,
  -- ** Initializers
  truncatedNormal, randomUniform, randomOrthogonal, varianceScaling, glorotUniform,

  -- ** Heterogeneous vectors
  repeatT, KnownTensors(..), LastEqual
  ) where

import Prelude hiding (tanh,Num(..),Floating(..),round,floor,(/),sqrt)
import qualified Prelude
import Prelude ((-))
import Text.PrettyPrint.Compact hiding (Last, All,Product,Sum)
import GHC.TypeLits
import Data.Proxy
import TypedFlow.Types
import TypedFlow.Python
import Control.Monad (when)
import TypedFlow.Abstract

-- | Repeat a flexible-shape constant vector to form a heterogeneous tensor vector.
repeatT :: forall (ss :: [Shape]) t. All KnownShape ss => KnownLen ss =>
           (forall s. KnownShape s => T s t) -> HTV t ss
repeatT f = zs (typeSList @ss)
  where zs :: forall (s :: [Shape]). All KnownShape s => SList s -> HTV t s
        zs LZ = Unit
        zs (LS _ n) = F f :* zs n

-- TODO: use a different type for persistent?
-- | Declare variable which persists between calls to session.run.
persistent :: ∀ (shape :: Shape) t. (KnownTyp t,KnownShape shape) => Bool -> String -> T shape t -> Gen (T shape t)
persistent trainable name (T initial) = do
  v <- newVar
  when trainable (newParameter (ParamInfo name (shapeToList @shape) (typVal @t) (T v)))
  v <-- funcall "tf.Variable" [initial, named "name" (string (show (name))), named "trainable" (bool trainable)]
  return (T v)


-- | Declare a parameter to optimize. The shape of parameter should
-- not depend on dimensions which can change between runs, such as the
-- batch size.
parameter' :: ∀ (shape :: Shape) t. (KnownTyp t,KnownShape shape) => String -> T shape t -> Gen (T shape t)
parameter' = persistent True

-- | Name a tensor so that it is made available for session.run.
peekAt :: String -> Tensor s t -> Gen ()
peekAt p (T v) = peekAtAny p v

peekAtMany :: String -> HTV t xs -> Gen ()
peekAtMany p htv = peekAtAny p (list $ htoList $ hmap (\(F (T x)) -> K x) htv)


-- | Modify a mutable tensor. Attention: for the assignment to happen,
-- the resulting tensor must be evaluated!
modifyPersistent :: T s t -> T s t -> T s t
modifyPersistent (T ref) (T value) = T (funcall "tf.assign" [ref,value])

-- TODO: get the parameters from the genParams field
-- | Return a list of parameters.
getParameters :: Gen UntypedExpression
getParameters = do
  v <- newVar
  v <-- text "tf.trainable_variables()"
  return v

-- TODO: get the parameters from the genParams field


-- TODO: gradient wrt. a HTV
-- | Gradient of wrt. given parameters.
grad :: T s Float32 -> UntypedExpression -> UntypedExpression
grad (T y) vars = funcall "tf.gradients" [y, vars]

-- -- | Gradient of wrt. given parameters.
-- grad' :: KnownLen xs => T s Float32 -> HHTV xs -> Gen (HHTV xs)
-- grad' (T y) vars = do
--  v <- newVar
--  v <-- funcall "tf.gradients" [y, list (htoList (hmap (\(Uncurry (T x)) -> K x) vars)) ]
--  return (mkArr 0 shapeSList v)
--   where mkArr :: forall xs. Int -> SList xs -> DOC -> HHTV xs
--         mkArr _ LZ _ = Unit
--         mkArr i (LS _ n) v = Uncurry (T (v <> brackets (int i))) :* mkArr (succ i) n v


-- | Clip a gradient
clipByGlobalNorm :: Float -> UntypedExpression -> UntypedExpression
clipByGlobalNorm maxNorm x = funcall "tf.clip_by_global_norm" [x,float maxNorm] <> brackets (int 0)
 -- clip_by_global_norm returns a couple (clipped grads, global_norm)

-- | Clip a tensor
clipByValue :: Float -> Float -> T s (Flt t) -> T s (Flt t)
clipByValue lo hi (T x) = T (funcall "tf.clip_by_value" [x, float lo, float hi])

-- | Placeholder (to fill)
placeholder :: ∀t s. (KnownShape s, KnownTyp t) => String -> Gen (T s t)
placeholder n = do
  let name = text n
  name <-- funcall "tf.placeholder" [showTyp @t, named "shape" (showShape @s), named "name" (text (show n))]
  peekAt n (T name)
  return (T name)



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

-- | Softmax along the first dimension
softmax0 :: T (n ': s) ('Typ 'Float w) -> T (n ': s) ('Typ 'Float w)
softmax0 = unOp "tf.nn.softmax"

-- | Softmax along the second dimension
softmax1 :: forall n m s w. KnownLen s => T (m ': n ': s) ('Typ 'Float w) -> T (m ': n ': s) ('Typ 'Float w)
softmax1 (T x) = T (funcall "tf.nn.softmax" [x, named "dim" (showShapeLen @s)])

-- | Argmax along dimension @n@
argmax :: forall n u m s t. (KnownLen s, KnownPeano n,KnownBits u) => Tensor (Take n s ++ (m ': Drop n s)) t -> Tensor s ('Typ 'Int u)
argmax (T t) = T (funcall "tf.argmax" [t, named "axis" (integer ((listLen @ s) - peanoInt @n)) , named "output_type" (showTyp @('Typ 'Int u))])

-- | Argmax along the first dimension
argmax0 :: forall u n s t. (KnownLen s, KnownBits u) => T (n ': s) t -> T s ('Typ 'Int u)
argmax0 = argmax @Dim0

-- | Argmax along the second dimension
argmax1 :: forall u m n s t. (KnownLen s, KnownBits u) => T (m ': n ': s) t -> T (m ': s) ('Typ 'Int u)
argmax1 = argmax @Dim1

-- | Cast the element type.
cast :: forall u s t. KnownTyp u => T s t -> T s u
cast (T t) = T (funcall "tf.cast" [t, showTyp @ u])


-- | (dense) softmax cross entropy with logits.
softmaxCrossEntropyWithLogits :: Tensor '[numClasses,batchSize] Float32 -- ^ labels
                              -> Tensor '[numClasses,batchSize] Float32 -- ^ logits
                              -> Tensor '[batchSize] Float32
softmaxCrossEntropyWithLogits (T labels) (T logits) =
  T (funcall "tf.nn.softmax_cross_entropy_with_logits" [named "labels" labels,named "logits" logits])

-- | Computes sigmoid cross entropy given logits. Measures the
-- probability error in discrete classification tasks in which each
-- class is independent and not mutually exclusive. For instance, one
-- could perform multilabel classification where a picture can contain
-- both an elephant and a dog at the same time. See
-- https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
sigmoidCrossEntropyWithLogits :: Tensor s (Flt w) -- ^ labels
                              -> Tensor s (Flt w) -- ^ logits
                              -> Tensor s (Flt w)
sigmoidCrossEntropyWithLogits (T labels) (T logits) =
  T (funcall "tf.nn.sigmoid_cross_entropy_with_logits" [named "labels" labels,named "logits" logits])

-- | sparse softmax cross entropy with logits.
sparseSoftmaxCrossEntropyWithLogits :: Tensor s Int32                   -- ^ desired labels
                                    -> Tensor (numClasses ': s) (Flt t) -- ^ predictions
                                    -> Tensor s (Flt t)
sparseSoftmaxCrossEntropyWithLogits (T labels) (T logits) =
  T (funcall "tf.nn.sparse_softmax_cross_entropy_with_logits" [named "labels" labels,named "logits" logits])

-- | One hot vector along axis @n@
oneHot :: forall n numClasses s w t. KnownNat numClasses => KnownBits t =>
  (KnownLen s, KnownPeano n) => Tensor s ('Typ 'Int w) -> Tensor (Take n s ++ (numClasses ': Drop n s)) (Flt t)
oneHot (T x) = T (funcall "tf.one_hot" [x, named "depth" (showDim @numClasses), named "axis" (integer (listLen @s - peanoInt @n)), named "dtype" (showTyp @(Flt t))])

-- | One hot vector along axis 0
oneHot0 :: forall numClasses w s t. KnownLen s => KnownNat numClasses => KnownBits t => Tensor s ('Typ 'Int w) -> Tensor (numClasses ': s) (Flt t)
oneHot0 = oneHot @Dim0

-- | One hot vector along axis 1
oneHot1 :: forall numClasses w s m t. KnownLen s => KnownNat numClasses => KnownBits t => Tensor (m ': s) ('Typ 'Int w) -> Tensor (m ': numClasses ': s) (Flt t)
oneHot1 = oneHot @Dim1

-- | Generate a random tensor where each individual element is picked
-- in a normal distribution with given standard deviation.
truncatedNormal :: forall s w. KnownShape s => KnownBits w => Float -> T s ('Typ 'Float w)
truncatedNormal stddev = T (funcall "tf.truncated_normal" [showShape @s, named "stddev" (float stddev), named "dtype" (showTyp @(Flt w))])

-- | Generate a random tensor where each individual element is picked
-- in a uniform distribution with given bounds.
randomUniform :: forall s t. (KnownShape s, KnownTyp t) => Float -> Float -> T s t
randomUniform low high = T (funcall "tf.random_uniform" [showShape @s
                                                        ,named "minval" (float low)
                                                        ,named "maxval" (float high)
                                                        ,named "dtype" (showTyp @t)])


-- | Generate an orthorgonal matrix. If the output has more dimensions
-- than 2 the matrix is reshaped.
randomOrthogonal :: forall n s t. (KnownBits t, KnownNat n, KnownShape s) => T (n ':s) ('Typ 'Float t)
randomOrthogonal = T (funcall' (funcall "tf.orthogonal_initializer" [named "dtype" (showTyp @('Typ 'Float t))])
                               [named "shape" (showShape @(n ': s))])

---------------------------
-- Contrib
data VarianceScaleMode = VSFanIn | VSFanOut | VSAvg
data Distrib = NormalDistr | UniformDistr

-- | Random tensor with variance scaling according to deeplearning lore.
varianceScaling :: forall inDim outDim t. KnownNat inDim => (KnownNat outDim, KnownBits t) =>
   Float -> VarianceScaleMode -> Distrib -> Tensor '[inDim,outDim] ('Typ 'Float t)
varianceScaling factor mode distr = case distr of
                                   UniformDistr -> randomUniform (-limit) limit
                                   NormalDistr -> truncatedNormal limit
  where
    fan_in = fromIntegral (natVal (Proxy @inDim))
    fan_out = fromIntegral (natVal (Proxy @outDim))
    n = max 1 $ case mode of
                  VSFanIn -> fan_in
                  VSFanOut -> fan_out
                  VSAvg -> (fan_in Prelude.+ fan_out) Prelude./ 2
    limit = Prelude.sqrt ((case distr of NormalDistr -> 1.3; UniformDistr -> 3) Prelude.* factor Prelude./ n)


glorotUniform :: forall inDim outDim t. KnownNat inDim => (KnownNat outDim, KnownBits t) => Tensor '[inDim,outDim] ('Typ 'Float t)
glorotUniform = varianceScaling 1 VSAvg UniformDistr

-- | 'cons' an element and an array (in the first dimension)
consT0 :: forall n s t. KnownLen s => T s t -> T (n ': s) t -> T (n+1 ': s) t
consT0 x xs = plusComm @1 @n $ concat0 (expandDim0 x) xs

-- | 'snoc' an element and an array (in the first dimension)
snocT0 :: forall n s t. KnownLen s => T (n ': s) t -> T s t -> T (n+1 ': s) t
snocT0 xs x = concat0 xs (expandDim0 x)

----------------
-- Helpers

-- matvecmulBatch :: ∀ s cols rows t. (KnownLen s) =>  Tensor (cols ': rows ': s) t -> Tensor (cols ': s) t -> Tensor (rows ': s) t
-- matvecmulBatch m v = squeeze0 (matmul m (expandDim0 v))

-- | Product of a matrix of weights with a (batched) vector .
(∙) :: Tensor '[cols, rows] t -> Tensor '[cols,batchSize] t -> Tensor '[rows,batchSize] t
m ∙ v = matmul v (transpose m)
infixl 7 ∙

-- | Dot product between two batched vectors.
(·) :: ∀ cols batchSize t. Tensor '[cols,batchSize] t -> Tensor '[cols,batchSize] t -> Tensor '[batchSize] t
x · y = reduceSum0 (x ⊙ y)
infixl 7 ·



-- apparently tensorflow (python?) is not aware of 2-argument
-- functions; so we do this... thing.
lambda2 :: (T s t -> T s1 t1 -> T s' t') -> Gen UntypedExpression
lambda2 f = do
  v <- newVar
  let T body = f (T (v <> brackets (int 0))) (T (v <> brackets (int 1)))
  return (text "lambda " <> v <> text ": " <> body)

-- | Selection of a tensor (note: this is a strict operation)
if_ :: Scalar TFBool -> T s t -> T s t -> T s t
if_ (T c) (T x) (T y) = T (funcall "tf.cond" [-- named "pred" -- names have changed between TF 1.1 and TF 1.3
                                              c,
                                              -- named "true_fn"
                                              (lambda0 x),
                                              -- named "false_fn"
                                              (lambda0 y),
                                              named "strict" (bool True)])
  where lambda0 z = text "lambda: " <> z

-- | (where_ c x y)[i] = if c[i] then x[i] else y[i]
where_ :: T s TFBool -> T s t -> T s t -> T s t
where_ (T c) (T x) (T y) = T (funcall "tf.where" [c, x, y])

-------------------------
-- Generic parameters

-- | Create a parameter and initialize it with a suitable default for its type. Control the exact initializer using 'parameter'.
parameterDefault :: forall p. ParamWithDefault p => String -> Gen p
parameterDefault name = parameter name defaultInitializer

-- | Create a parameter.
parameter :: forall p. KnownTensors p => String -> p -> Gen p
parameter = travTensor parameter'

class KnownTensors p where
  -- | traverse all the tensors over tuples of tensors
  travTensor :: (forall s t. (KnownTyp t, KnownShape s) => String -> T s t -> Gen (T s t)) -> String -> p -> Gen p 

instance (KnownTyp t, KnownShape shape) => KnownTensors (T shape t) where
  travTensor f = f

instance (KnownTyp t, All KnownShape ys) => KnownTensors (HTV t ys) where
  travTensor f s = ttr 0
    where ttr :: forall xs. All KnownShape xs => Int -> HTV t xs -> Gen (HTV t xs)
          ttr _ Unit = return Unit
          ttr n (F x :* xs) = do
            x' <- f (s <> "_" <> show n) x
            xs' <- ttr (n Prelude.+ 1) xs
            return (F x' :* xs')

instance (KnownTensors p, KnownTensors q) => KnownTensors (p,q) where
  travTensor f s (x,y) = (,) <$> travTensor f (s<>"_fst") x <*> travTensor f (s<>"_snd") y

instance (KnownTensors p1, KnownTensors p2, KnownTensors p3) => KnownTensors (p1,p2,p3) where
  travTensor f s (x,y,z) = (,,) <$> travTensor f (s<>"_1") x <*> travTensor f (s<>"_2") y <*> travTensor f (s<>"_3") z

instance (KnownTensors p1, KnownTensors p2, KnownTensors p3, KnownTensors p4) => KnownTensors (p1,p2,p3,p4) where
  travTensor f s (x,y,z,w) = (,,,) <$> travTensor f (s<>"_1") x <*> travTensor f (s<>"_2") y <*> travTensor f (s<>"_3") z <*> travTensor f (s<>"_4") w

class KnownTensors p => ParamWithDefault p where
  defaultInitializer :: p

-- | Flatten all the dimensions of the tensor
flattenAll :: forall s t. KnownShape s => Tensor s t -> Tensor '[Product s] t
flattenAll = knownProduct @s reshape


flattenHTV :: KnownTyp t => All KnownShape xs => HTV t xs -> Tensor '[Sum (Ap (FMap CProduct) xs)] t
flattenHTV Unit = zeros
flattenHTV (F x :* xs) = concat0 (flattenAll x) (flattenHTV xs)

inflateAll :: forall s t. KnownShape s => Tensor '[Product s] t -> Tensor s t
inflateAll = knownProduct @s reshape

class CProduct (xs :: [Nat])
instance Fun CProduct where type Ap CProduct xs = Product xs

inflateHTV :: ∀ xs s t. (All KnownShape xs, KnownLen s, KnownLen xs) =>
          Tensor '[Sum (Ap (FMap CProduct) xs)] t -> Gen (HTV t xs)
inflateHTV (T x) = do
  v <- newVar
  gen (v <> text " = " <> funcall "tf.split" [x, showShape' (prodshape @xs shapeSList), text "axis=0"])
  return (mkArr @xs 0 shapeSList  v)
  where mkArr :: forall zs. All KnownShape zs => Int -> SList zs -> DOC -> HTV t zs
        mkArr _ LZ _ = Unit
        mkArr i (LS _ n) v = F (unsafeReshape (T (v <> brackets (int i)) )):* mkArr (succ i) n v

        prodshape :: forall zs. All KnownShape zs => SList zs -> [Integer]
        prodshape LZ = []
        prodshape (LS xx xs) = product (shapeToList' (shapeSListProxy xx)) : prodshape xs



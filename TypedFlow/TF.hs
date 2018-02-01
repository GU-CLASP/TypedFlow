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
  AddSpatialDims, convolution, convolutionValid,
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
  repeatT, flattenHTV, inflateHTV, KnownTensors(..), LastEqual
  ) where

import Prelude hiding (tanh,Num(..),Floating(..),round,floor,(/),sqrt)
import qualified Prelude
import Prelude ((-))
import Text.PrettyPrint.Compact hiding (Last, All,Product,Sum)
import GHC.TypeLits
import Data.Proxy
import TypedFlow.Types
import Control.Monad (when)

-- | Repeat a flexible-shape constant vector to form a heterogeneous tensor vector.
repeatT :: forall (ss :: [Shape]) t. All KnownShape ss => KnownLen ss =>
           (forall s. KnownShape s => T s t) -> HTV t ss
repeatT f = zs (shapeSList @ss)
  where zs :: forall (s :: [Shape]). All KnownShape s => SList s -> HTV t s
        zs LZ = Unit
        zs (LS _ n) = F f :* zs n

-- | Zeros
zeros :: ∀ t (shape :: Shape). KnownShape shape => KnownTyp t => (T shape t)
zeros = T (funcall "tf.zeros" [showShape @shape, named "dtype" (showTyp @t)])

-- | Ones
ones :: ∀ t (shape :: Shape). KnownShape shape => KnownTyp t => (T shape t)
ones = T (funcall "tf.ones" [showShape @shape, named "dtype" (showTyp @t)])

-- | Identity matrix in dimensions m,n (extended with zeros if m ≠ n), and repeated on shape s.
eye :: ∀ m n s t. KnownShape s => KnownNat m => KnownNat n => KnownTyp t => (T (m ': n ': s) t)
eye = T (funcall "tf.eye" [showDim @n,
                            named "num_columns" (showDim @m),
                            named "batch_shape" (showShape @s),
                            named "dtype" (showTyp @t)])


-- | Constant
constant :: forall s t w. KnownShape s => KnownBits w => KnownKind t => HostType t -> T s ('Typ t w)
constant c = T (funcall "tf.constant" [pretty c, named "shape" (showShape @s), named "dtype" (showTyp @('Typ t w))])

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

-- | Internal. Use 'reduceMeanAll', etc. instead.
reduceAll :: String -> Tensor s t -> Tensor '[] t
reduceAll op = unOp ("tf.reduce_" ++ op)

-- | Mean value of the input tensor.
reduceMeanAll, reduceSumAll, reduceMaxAll :: ∀ (s :: Shape) t. Tensor s t -> Tensor '[] t
reduceMaxAll = reduceAll "max"
reduceMeanAll = reduceAll "mean"
reduceSumAll = reduceAll "sum"

-- | Internal. Use 'reduceSum', etc. instead.
reduce :: ∀ n s t. (KnownLen s,KnownPeano n) => String -> T s t -> T (Take n s ++ Drop ('Succ n) s) t
reduce op (T x) = T (funcall ("tf.reduce_" ++ op) [x, text "axis=" <> integer (listLen @ s - peanoInt @n - 1)])

-- | Reduce along a given dimension
reduceSum, reduceMean, reduceMax :: ∀n s t. (KnownLen s,KnownPeano n) => T s t -> T (Take n s ++ Drop ('Succ n) s) t
reduceSum = reduce @n "sum"
reduceMean = reduce @n "mean"
reduceMax = reduce @n "max"


-- | Sum along the first dimension
reduceSum0 :: ∀ s' n t. KnownLen s' => Tensor (n ': s') t -> Tensor s' t
reduceSum0 = reduceSum @Dim0

-- | Add two tensors, broacasting along shape @s@
add :: ∀ s d t. Tensor (d++s) t -> Tensor d t -> Tensor (d++s) t -- note ++s for for 'broadcasting'
add = binOp "tf.add"

addN :: ∀ s t. KnownTyp t => KnownShape s => [Tensor s t] -> Tensor s t
addN [] = zeros
addN ts = T (funcall "tf.add_n" [list [x | T x <- ts]])

-- | Add two tensors, broacasting along shape @s@
(+) :: ∀ (d :: Shape) (s :: Shape) t. Tensor (d ++ s) t -> Tensor d t -> Tensor (d ++ s) t
(+) = add @s @d
infixl 6 +

-- | Divide tensors, broacasting along shape @s@
(/) :: ∀ (d :: Shape) (s :: Shape) t. Tensor (d ++ s) t -> Tensor d t -> Tensor (d ++ s) t
(/) = binOp "tf.divide"
infixl 7 /

-- | Indexwise equality test.
equal :: Tensor d t -> Tensor d t -> Tensor d TFBool
equal = binOp "tf.equal"

-- | Indexwise operator
(⊕), (⊝), (⊙), (⊘) :: ∀ (s :: Shape) t. Tensor s t -> Tensor s t -> Tensor s t
(⊝) = binOp "tf.subtract"
(⊙) = binOp "tf.multiply"
(⊘) = binOp "tf.divide"
(⊕) = binOp "tf.add"

infixl 7 ⊙,⊘
infixl 6 ⊕,⊝

-- | Matrix multiplication (note that shape @s@ is preserved)
matmul :: Tensor (o ': n ': s) t -> Tensor (m ': o ': s) t -> Tensor (m ': n ': s) t
matmul = binOp "tf.matmul"

round, sigmoid, tanh, log, relu, floor, sqrt, square
   :: ∀ s t. Tensor s ('Typ 'Float t) -> Tensor s ('Typ 'Float t)
sigmoid = unOp "tf.sigmoid"
tanh = unOp "tf.tanh"
log = unOp "tf.log"
relu = unOp "tf.nn.relu"
round = unOp "tf.round"
floor = unOp "tf.floor"
sqrt = unOp "tf.sqrt"
square = unOp "tf.square"

negate :: ∀ s t. T s t -> T s t
negate = unOp "-"

-- | Split a tensor on the first dimension
split0 :: ∀ n m batchShape t. (KnownNat n, KnownNat m, KnownLen batchShape) =>
          Tensor ((n + m) ': batchShape) t -> Gen (Tensor (n ': batchShape) t, Tensor (m ': batchShape) t)
split0 (T x) = do
  v1 <- newVar
  v2 <- newVar
  gen (v1 <> text "," <> v2 <> text " = " <> funcall "tf.split" [x, list [showDim @ n, showDim @ m], text "axis=" <> showShapeLen @batchShape])
  return (T v1, T v2)

-- | Concatenate tensors on dimension @n@
concatT :: ∀ n d1 d2 s t. (KnownPeano n, KnownLen s, (d1+d2) ~ At n s) =>
    T (Take n s ++ (d1 ': Drop ('Succ n) s)) t -> T (Take n s ++ (d2 ': Drop ('Succ n) s)) t -> T s t
concatT (T x) (T y) = T (funcall "tf.concat" [list [x,y], named "axis" (integer (listLen @s - peanoInt @n - 1))])

-- | Concatenate tensors on the first dimension
concat0 :: ∀ ys d1 d2 t. (KnownLen ys) => T (d1 ': ys) t -> T (d2 ': ys) t -> T ((d1 + d2) ': ys) t
concat0 = concatT @Dim0

-- | Concatenate tensors on the second dimension
concat1 :: ∀ n ys d1 d2 t. (KnownLen ys) =>  T (n ': d1 ': ys) t -> T (n ': d2 ': ys) t -> T (n ': (d1 + d2) ': ys) t
concat1 = concatT @Dim1

-- | Add an extra dimension at axis (@n@) of size 1.
expandDim :: forall n s t. (KnownLen s, KnownPeano n) => Tensor s t -> Tensor (Take n s ++ (1 ': Drop n s)) t
expandDim (T x) = (T (funcall "tf.expand_dims" [x, named "axis" (integer (listLen @s - peanoInt @n))]))

-- | Add an extra dimension at axis (0) of size 1.
expandDim0 :: ∀ s t. KnownLen s => Tensor s t -> Tensor (1 ': s) t
expandDim0 = expandDim @Dim0

-- | Add an extra dimension at axis (1) of size 1.
expandDim1 :: ∀ n s t. KnownShape s => Tensor (n ': s) t -> Tensor (n ': 1 ': s) t
expandDim1 = expandDim @Dim1

-- -- | Tile a tensor along the first dimension
-- tile0 :: forall m n s t. KnownLen s => (KnownNat m) => Tensor (n ': s) t -> Tensor ((m * n) ': s) t
-- tile0 (T x) = T (funcall "tf.tile" [x, list (genericReplicate (listLen @s) (int 1) ++ [integer (natVal (Proxy @m))])])
-- Probably less efficient than a broadcast; so I leave it out for now.

broadcast0 :: forall n s t. KnownTyp t => KnownNat n => KnownShape s => Tensor s t -> Tensor (n ': s) t
broadcast0 x = binOp "tf.add" (zeros @t @(n ': s)) x
 -- this is some "hack to force the shape to that we want."

broadcastN :: forall n s t. KnownTyp t => KnownNat n => KnownShape s => Tensor s t -> Tensor (s ++ '[n]) t
broadcastN x = knownAppend @s @'[n] $
  binOp "tf.add" (zeros @t @(s ++ '[n])) x


-- -- | Replicate a tensor
-- replicateT :: ∀ n s t. (KnownNat n, KnownLen s) => T s t -> T (n ': s) t
-- replicateT = tile @n . expandDim0

-- | Remove a dimension if its size is 1.
squeeze :: ∀ s0 s1 t. KnownLen s1 => Tensor (s0 ++ (1 ': s1)) t -> Tensor (s0 ++ s1) t
squeeze (T x) = T (funcall "tf.squeeze" [x, text "axis=" <> integer (listLen @ s1)])

-- | Remove the first dimension if its size is 1.
squeeze0 :: ∀ s t. KnownLen s => Tensor (1 ': s) t -> Tensor s t
squeeze0 = squeeze @ '[]

-- | Remove the second dimension if its size is 1.
squeeze1 :: ∀ n s t. KnownLen s => Tensor (n ': 1 ': s) t -> Tensor (n ': s) t
squeeze1 = squeeze @ '[n]

reshape :: ∀ s2 s1 t. KnownShape s2 => Product s1 ~ Product s2 => Tensor s1 t -> Tensor s2 t
reshape = unsafeReshape

unsafeReshape :: ∀ s2 s1 t. KnownShape s2 => Tensor s1 t -> Tensor s2 t
unsafeReshape (T t) = T (funcall "tf.reshape" [t, showShapeMinus @s2])

-- | Reshape a tensor so that the first two dimensions are collapsed
flatten2 :: ∀ m n s t. (KnownNat m, KnownNat n, KnownShape s) => Tensor (m ': n ': s) t -> Tensor (m*n ': s) t
flatten2 = prodAssoc @m @n @(Product s) reshape

-- | Reshape a tensor so that the last two dimensions are collapsed
flattenN2 :: ∀ s m n t. (KnownNat m, KnownNat n, KnownShape s) => Tensor (s ++ '[m,n]) t -> Tensor (s ++ '[m*n]) t
flattenN2  = prodHomo @s @'[m,n] $
             prodHomo @s @'[m*n] $
             knownAppend @s @'[m*n] $
             reshape

-- | Reshape a tensor so that the first three dimensions are collapsed
flatten3 :: ∀ m n o s t. (KnownNat m, KnownNat n, KnownNat o, KnownShape s) => Tensor (m ': n ': o ': s) t -> Tensor (m*n*o ': s) t
flatten3  =  -- (m * (n * (o * Product s)))
             prodAssoc @m @n @(o * Product s) $
             -- (m * n) * (o * Product s)
             prodAssoc @(m * n) @o @(Product s) $
             -- ((m * n) * o) * Product s
             reshape
-- | Reshape a tensor so that the first two dimensions are collapsed
flatten12 :: ∀ m n o s t. KnownNat o => (KnownNat m, KnownNat n, KnownShape s) => Tensor (o ': m ': n ': s) t -> Tensor (o ': m*n ': s) t
flatten12 = prodAssoc @m @n @(Product s) reshape

-- | Reshape a tensor so that the first dimension is expanded into two.
inflate2 :: ∀ m n s t. (KnownNat m, KnownNat n, KnownShape s) => Tensor (m*n ': s) t -> Tensor (m ': n ': s) t
inflate2 = prodAssoc @m @n @(Product s) reshape

-- | Reshape a tensor so that the first dimension is expanded into three.
inflate3 :: ∀ m n o s t. (KnownNat m, KnownNat n, KnownNat o, KnownShape s) => Tensor (m*n*o ': s) t -> Tensor (m ': n ': o ': s) t
inflate3 = -- (m * (n * (o * Product s)))
           prodAssoc @m @n @(o * Product s) $
           -- (m * n) * (o * Product s)
           prodAssoc @(m * n) @o @(Product s) $
           -- ((m * n) * o) * Product s
           reshape

-- | Reshape a tensor so that the first two dimensions are collapsed
inflate12 :: ∀ m n o s t. KnownNat o => (KnownNat m, KnownNat n, KnownShape s) => Tensor (o ': m*n ': s) t -> Tensor (o ': m ': n ': s) t
inflate12 = prodAssoc @m @n @(Product s) reshape



-- | Access the last element in a tensor (in the 0th dimension)
last0 :: ∀ n s t. KnownNat n => KnownLen s => T (n ': s) t -> Tensor s t
last0 = nth0 (natVal (Proxy @n) - 1)

-- | Access the nth element in a tensor (in the 0th dimension)
nth0 :: ∀ n s t. KnownLen s => Integer -> T (n ': s) t -> Tensor s t
nth0 i (T x) = T (x <> list (replicate (fromIntegral (listLen @s)) (text ":") ++ [integer i]))

-- | Access the nth element in a tensor (in the 0th dimension), with a static index
nth0' :: ∀ n m s t. KnownNat n => KnownLen s => n < m => T (m ': s) t -> Tensor s t
nth0' (T x) = T (x <> list (replicate (fromIntegral (listLen @s)) (text ":") ++ [integer (natVal (Proxy @n))]))

-- | Take a slice at dimension n from i to j.
slice :: forall n i j s t. KnownNat j => KnownNat i => (i < j, j <= At n s, KnownPeano n, KnownLen s) =>
         Tensor s t -> Tensor (Take n s ++ ((j-i) ': Drop ('Succ n) s)) t
slice (T x) = T (x <> list (replicate (fromIntegral (listLen @s - peanoInt @n - 1)) (text ":") ++ [integer (natVal (Proxy @i)) <> text ".." <> integer (natVal (Proxy @j))]))

slice1 :: forall i j m n s t. KnownNat j => KnownNat i => (i < j, j <= m, KnownLen s) =>
         Tensor (n ': m ': s) t -> Tensor (n ': (j-i) ': s) t
slice1 = slice @Dim1 @i @j

-- | Split a tensors into @n@ tensors along the first dimension
unstack0 :: ∀ s (n::Nat) t. (KnownLen s, KnownNat n) => Tensor (n ': s) t -> Gen (V n (T s t))
unstack0 (T x) = do
  v <- newVar
  v <-- funcall "tf.unstack" [x, text "axis=" <> integer (listLen @ s)]
  return $ V $ [ T $ v <> brackets (integer i)| i <- [0..n Prelude.- 1] ]
        where n = natVal (Proxy @ n)

-- | Concatenate @n@ tensors along the first dimension
stack0 :: ∀ s (n::Nat) t. (KnownLen s) => V n (T s t) -> Tensor (n ': s) t
stack0 (V xs) = T (funcall "tf.stack" [list [x | T x <- xs], text "axis=" <> integer (listLen @ s)])

-- | Concatenate @n@ tensors along the second dimension
stack1 :: ∀ s (n::Nat) m t. (KnownLen s) => V n (T (m ': s) t) -> Tensor (m ': n ': s) t
stack1 (V xs) = T (funcall "tf.stack" [list [x | T x <- xs], text "axis=" <> integer (listLen @ s)])

-- | Concatenate @n@ tensors along the last dimension
stackN :: ∀ s (n::Nat) t. V n (T s t) -> Tensor (s ++ '[n]) t
stackN (V xs) = T (funcall "tf.stack" [list [x | T x <- xs], text "axis=0"])

-- | Transposition. See the type for the permutation of dimensions.
transpose :: ∀ s t. T (Reverse s) t -> T s t
transpose = unOp "tf.transpose"

-- | Transposition. See the type for the permutation of dimensions.
transposeN :: ∀ s n t. KnownLen s => T (n ': s) t -> T (s ++ '[n]) t
transposeN (T x) = T (funcall "tf.transpose" [x, named "perm" (list (map integer (listLen @s:[0.. listLen @s-1])))])

-- | Transposition. See the type for the permutation of dimensions.
transposeN' :: ∀ s n t. KnownLen s => T (s ++ '[n]) t -> T (n ': s) t
transposeN' (T x) = T (funcall "tf.transpose" [x, named "perm" (list (map integer ([1.. listLen @s]++[0])))])

-- | Transposition. See the type for the permutation of dimensions.
transpose01 :: ∀ s m n t. KnownLen s => T (m ': n ': s) t -> T (n ': m ': s) t
transpose01 (T x) = T (funcall "tf.transpose" [x, named "perm" (list (map integer ([0..l-1] ++ [l Prelude.+ 1,l])))])
  where l = listLen @s

-- | Transposition. See the type for the permutation of dimensions.
transposeN01 :: ∀ s m n t. T (s ++ [m,n]) t -> T (s ++ [n,m]) t
transposeN01 (T x) = T (funcall "tf.transpose" [x, named "perm" (list (map integer [1,0]))])

class LastEqual x xs
instance                   LastEqual x (x ': '[])
instance LastEqual x (y2 ': xs) => LastEqual x (y ': (y2 ': xs))

-- | Reverse sequences. See https://www.tensorflow.org/api_docs/python/tf/reverse_sequence
reverseSequences :: forall bs n x t. KnownLen x => LastEqual bs x => T '[bs] Int32 -> T (n ': x) t -> T (n ': x) t
reverseSequences (T seqLengths) (T input) =
  T (funcall "tf.reverse_sequence" [input, seqLengths, named "seq_axis" (showShapeLen @x),named "batch_axis" (int 0)])

-- | Generate a mask of given length for each sequence.
sequenceMask :: forall maxlen bs. KnownNat maxlen => Tensor '[bs] Int32 -> Tensor '[maxlen,bs] TFBool
sequenceMask (T x) = T (funcall "tf.sequence_mask" [x, named "maxlen" (showDim @maxlen)])


-- | @(gather x ix)[k] = x[ix[k]]@. See https://www.tensorflow.org/api_docs/python/tf/gather
gather :: ∀s n indexShape t. T (s ++ '[n]) t -> T indexShape Int32 -> T (s ++ indexShape) t
gather = binOp "tf.gather"


untypedConvolution :: forall outputChannels filterSpatialShape inChannels s t y.
               KnownLen filterSpatialShape
            => Length filterSpatialShape <= 3
            => ((1 + Length filterSpatialShape) ~ Length s) -- the last dim of s is the batch size
            => String
            -> T (inChannels ': y) t -- ^ input tensor (batched)
            -> T ('[outputChannels,inChannels] ++ filterSpatialShape) t -- ^ filters
            -> T ('[outputChannels] ++ s) t
untypedConvolution padding (T input) (T filters) = T (funcall "tf.nn.convolution"
                                                      [input,filters
                                                      ,named "padding" (text (show padding)) 
                                                      ,named "data_format" (text (show dataFormat))])
  where dataFormat = case listLen @ filterSpatialShape of
          1 -> "NWC"
          2 -> "NHWC"
          3 -> "NDHWC"
          _ -> error "convolution: more than 3 spatial dimensions are not supported!"

-- | Size-preserving convolution operation.
convolution :: forall outputChannels filterSpatialShape inChannels s t.
               KnownLen filterSpatialShape
            => Length filterSpatialShape <= 3
            => ((1 + Length filterSpatialShape) ~ Length s) -- the last dim of s is the batch size
            => T ('[inChannels] ++ s) t -- ^ input tensor (batched)
            -> T ('[outputChannels,inChannels] ++ filterSpatialShape) t -- ^ filters
            -> T ('[outputChannels] ++ s) t
convolution = untypedConvolution "SAME"

type family AddSpatialDims xs ys where
  AddSpatialDims '[x] '[] = '[x]
  AddSpatialDims (x ': xs) (y ': ys) = (x+(y-1)) ': AddSpatialDims xs ys

-- | Convolution operation with no padding (applying the filter only on positions where the input is fully defined)
convolutionValid :: forall outputChannels filterSpatialShape inChannels s t.
               KnownLen filterSpatialShape
            => Length filterSpatialShape <= 3
            => ((1 + Length filterSpatialShape) ~ Length s) -- the last dim of s is the batch size
            => T (inChannels ': AddSpatialDims s filterSpatialShape) t -- ^ input tensor (batched)
            -> T ('[outputChannels,inChannels] ++ filterSpatialShape) t -- ^ filters
            -> T (outputChannels ': s) t
convolutionValid = untypedConvolution "VALID"

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

-- mapT' :: forall s t r u n. KnownLen r => KnownLen s => KnownNat n => (T s t -> T r u) ->  T (n ': s) t -> Gen (T (n ': r) u)
-- mapT' f t = do
--   xs <- unstack t
--   return (stack (fmap f xs))

-- | Map a function along the first dimension of a tensor
mapT :: forall s t r u n. KnownTyp u => KnownLen r => KnownLen s => (T s t -> T r u) ->  T (n ': s) t -> Gen (T (n ': r) u)
mapT f x = do
  x' <- mapTN @n f (transposeN @s @n x)
  return (transposeN' @r x')

-- | Map a function along the last dimension of a tensor
mapTN :: forall n s t r u. KnownTyp u => (T s t -> T r u) ->  T (s ++ '[n]) t -> Gen(T (r ++ '[n]) u)
mapTN f t = do
  fn <- lambda f
  return (T (funcall "tf.map_fn" [fn, fromTensor t, named "dtype" (showTyp @u)]))

-- TODO: separate harmless and harmful effects. (the big question: are assignments harmful?)

zipWithT :: forall (s :: [Nat]) (t :: Typ) (s1 :: [Nat]) (t1 :: Typ) (s2 :: Shape) (n :: Nat) (t2 :: Typ).
            KnownNat n => (KnownLen s, KnownLen s2, KnownLen s1) => KnownTyp t2 =>
                  (T s t -> T s1 t1 -> T s2 t2)
                  -> Tensor (n ': s) t
                  -> Tensor (n ': s1) t1
                  -> Gen (Tensor (n ': s2) t2)
zipWithT f x y = do
  -- xs <- unstack x
  -- ys <- unstack y
  -- return (stack (f <$> xs <*> ys))
  x' <- zipWithTN @n f (transposeN @s @n x) (transposeN @s1 @n y)
  return (transposeN' @s2 x')

zipWithTN :: forall (n :: Nat) (s :: [Nat]) (t :: Typ) (s1 :: [Nat]) (t1 :: Typ) (s2 :: Shape) (t2 :: Typ).
            KnownTyp t2 =>
                  (T s t -> T s1 t1 -> T s2 t2)
                  -> Tensor (s ++ '[n]) t
                  -> Tensor (s1 ++ '[n]) t1
                  -> Gen (Tensor (s2 ++ '[n]) t2)
zipWithTN f (T t) (T u) =  do
  fn <- lambda2 f
  return (T (funcall "tf.map_fn" [fn, tuple [t,u], named "dtype" (showTyp @t2)]))


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



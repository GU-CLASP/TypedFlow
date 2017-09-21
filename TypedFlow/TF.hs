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

module TypedFlow.TF where

import Prelude hiding (tanh,Num(..),Floating(..))
import qualified Prelude
import Prelude ((-))
import Text.PrettyPrint.Compact hiding (Last, All)
import GHC.TypeLits
import Data.Proxy
import TypedFlow.Types


-- | Repeat a flexible-shape constant vector to form a heterogeneous tensor vector.
repeatT :: forall (ss :: [Shape]) t. All KnownShape ss => KnownLen ss => (forall s. KnownShape s => T s t) -> HTV t ss
repeatT f = zs (shapeSList @ss)
  where zs :: forall (s :: [Shape]). All KnownShape s => SList s -> HTV t s
        zs LZ = Unit
        zs (LS _ n) = F f :* zs n

-- | Zeros
zeros :: ∀ t (shape :: Shape). KnownShape shape => (T shape t)
zeros = T (funcall "tf.zeros" [showShape @shape])

-- | Ones
ones :: ∀ t (shape :: Shape). KnownShape shape => (T shape t)
ones = T (funcall "tf.ones" [showShape @shape])

-- | Constant
constant :: forall s w. KnownShape s => Float -> T s ('Typ 'Float w)
constant c = T (funcall "tf.constant" [float c, named "shape" (showShape @s)])


-- | Declare a parameter to optimize. The shape of parameter should
-- not depend on dimensions which can change between runs, such as the
-- batch size.
parameter' :: ∀ (shape :: Shape) t. (KnownTyp t,KnownShape shape) => String -> T shape t -> Gen (T shape t)
parameter' name (T initial) = do
  v <- newVar
  newParameter (ParamInfo name (shapeToList @shape) (typVal @t) (T v))
  v <-- funcall "tf.Variable" [initial, named "name" (string (show (name)))]
  return (T v)

-- TODO: get the parameters from the genParams field
-- | Return a list of parameters.
getParameters :: Gen UntypedExpression
getParameters = do
  v <- newVar
  v <-- text "tf.trainable_variables()"
  return v

-- TODO: gradient wrt. a HTV
-- | Gradient of wrt. given parameters.
grad :: T s Float32 -> UntypedExpression -> UntypedExpression
grad (T y) vars = funcall "tf.gradients" [y, vars]

-- | Clip a gradient
clipByGlobalNorm :: Float -> UntypedExpression -> UntypedExpression
clipByGlobalNorm maxNorm x = funcall "tf.clip_by_global_norm" [x,float maxNorm] <> brackets (int 0)
 -- clip_by_global_norm returns a couple (clipped grads, global_norm)

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
reduceMeanAll, reduceSumAll :: ∀ (s :: Shape) t. Tensor s t -> Tensor '[] t
reduceMeanAll = reduceAll "mean"
reduceSumAll = reduceAll "sum"

-- | Internal. Use 'reduceSum', etc. instead.
reduce :: ∀ s s' n t. KnownLen s' => String -> Tensor (s ++ (n ': s')) t -> Tensor (s ++ s') t
reduce op (T x) = T (funcall ("tf.reduce_" ++ op) [x, text "axis=" <> integer (listLen @ s')])

-- | Sum along a given dimension
reduceSum, reduceMean :: ∀ s s' n t. KnownLen s' => Tensor (s ++ (n ': s')) t -> Tensor (s ++ s') t
reduceSum = reduce @s @s' @n "sum"
reduceMean = reduce @s @s' @n "mean"

-- | Sum along the first dimension
reduceSum0 :: ∀ s' n t. KnownLen s' => Tensor (n ': s') t -> Tensor s' t
reduceSum0 = reduceSum @'[]

-- | Add two tensors, broacasting along shape @s@
add :: ∀ s d t. Tensor (d++s) t -> Tensor d t -> Tensor (d++s) t -- note ++s for for 'broadcasting'
add = binOp "tf.add"

-- add_n :: ∀ s t. [Tensor s t] -> Tensor s t
-- add_n = error "add_n not implemented"

-- | Add two tensors, broacasting along shape @s@
(+) :: ∀ (d :: Shape) (s :: Shape) t. Tensor (d ++ s) t -> Tensor d t -> Tensor (d ++ s) t
(+) = add @s @d

-- | Indexwise equality test.
equal :: Tensor d t -> Tensor d t -> Tensor d TFBool
equal = binOp "tf.equal"

-- | Indexwise sum.
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

round, sigmoid, tanh, log, relu, floor :: ∀ s t. Tensor s ('Typ 'Float t) -> Tensor s ('Typ 'Float t)
sigmoid = unOp "tf.sigmoid"
tanh = unOp "tf.tanh"
log = unOp "tf.log"
relu = unOp "tf.nn.relu"
round = unOp "tf.round"
floor = unOp "tf.floor"

negate :: ∀ s t. T s t -> T s t
negate = unOp "-"

-- | Split a tensor on the first dimension
split0 :: ∀ m n batchShape t. (KnownNat n, KnownNat m, KnownLen batchShape) =>
          Tensor ((n + m) ': batchShape) t -> Gen (Tensor (n ': batchShape) t, Tensor (m ': batchShape) t)
split0 (T x) = do
  v1 <- newVar
  v2 <- newVar
  gen (v1 <> text "," <> v2 <> text " = " <> funcall "tf.split" [x, list [showDim @ n, showDim @ m], text "axis=" <> showShapeLen @batchShape])
  return (T v1, T v2)

-- | Concatenate tensors on dimension @n@
concatT :: ∀ n d1 d2 s t.
    (KnownPeano n, KnownLen s, (d1+d2) ~ At n s) =>
    T (Take n s ++ (d1 ': Drop ('Succ n) s)) t -> T (Take n s ++ (d2 ': Drop ('Succ n) s)) t -> T s t
concatT (T x) (T y) = T (funcall "tf.concat" [list [x,y], named "axis" (integer (listLen @s - peanoInt @n - 1))])

-- | Concatenate tensors on the first dimension
concat0 :: ∀ ys d1 d2 t. (KnownShape ys) =>  T (d1 ': ys) t -> T (d2 ': ys) t -> T ((d1 + d2) ': ys) t
concat0 = concatT @Dim0
  -- let T x = t
  --     T y = u
  -- in (T (funcall "tf.concat" [list [x,y], text "axis=" <> integer (listLen @ ys)]))

-- | Concatenate tensors on the second dimension
concat1 :: ∀ n ys d1 d2 t. (KnownShape ys) =>  T (n ': d1 ': ys) t -> T (n ': d2 ': ys) t -> T (n ': (d1 + d2) ': ys) t
concat1 = concatT @Dim1

-- expandDim :: ∀ s0 s t. KnownLen s => Tensor (s0 ++ s) t -> Tensor (s0 ++ (1 ': s)) t
-- expandDim (T x) = (T (funcall "tf.expand_dims" [x, text "axis=" <> integer (listLen @ s)]))

-- | Add an extra dimension at axis (@n@) of size 1.
expandDim :: forall n s t. (KnownLen s, KnownPeano n) => Tensor s t -> Tensor (Take n s ++ (1 ': Drop n s)) t
expandDim (T x) = (T (funcall "tf.expand_dims" [x, named "axis" (integer (listLen @s - peanoInt @n))]))

-- | Add an extra dimension at axis (0) of size 1.
expandDim0 :: ∀ s t. KnownLen s => Tensor s t -> Tensor (1 ': s) t
expandDim0 = expandDim @Dim0

-- | Add an extra dimension at axis (1) of size 1.
expandDim1 :: ∀ n s t. KnownShape s => Tensor (n ': s) t -> Tensor (n ': 1 ': s) t
expandDim1 = expandDim @Dim1

-- | Tile a tensor along the first dimension
tile :: forall m n s t. (KnownNat m) => Tensor (n ': s) t -> Tensor ((m * n) ': s) t
tile (T x) = T (funcall "tf.tile" [x, integer (natVal (Proxy @m))])

-- | Replicate a tensor
replicateT :: ∀ n s t. (KnownNat n, KnownLen s) => T s t -> T (n ': s) t
replicateT = tile @n . expandDim0

-- | Remove a dimension if its size is 1.
squeeze :: ∀ s0 s1 t. KnownLen s1 => Tensor (s0 ++ (1 ': s1)) t -> Tensor (s0 ++ s1) t
squeeze (T x) = T (funcall "tf.squeeze" [x, text "axis=" <> integer (listLen @ s1)])

-- | Remove the first dimension if its size is 1.
squeeze0 :: ∀ s t. KnownLen s => Tensor (1 ': s) t -> Tensor s t
squeeze0 = squeeze @ '[]

-- | Remove the second dimension if its size is 1.
squeeze1 :: ∀ n s t. KnownLen s => Tensor (n ': 1 ': s) t -> Tensor (n ': s) t
squeeze1 = squeeze @ '[n]

-- | Reshape a tensor so that the first two dimensions are collapsed
flatten2 :: ∀ m n s t. (KnownNat m, KnownNat n, KnownShape s) => Tensor (m ': n ': s) t -> Tensor (m*n ': s) t
flatten2 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m*n ': s)])

-- | Reshape a tensor so that the last two dimensions are collapsed
flattenN2 :: ∀ s m n t. (KnownNat m, KnownNat n, KnownShape s) => Tensor (s ++ '[m,n]) t -> Tensor (s ++ '[m*n]) t
flattenN2 (T t) = knownShapeApp @s @'[m*n] $ T (funcall "tf.reshape" [t, showShapeMinus @(s ++ '[m*n])])

-- | Reshape a tensor so that the first three dimensions are collapsed
flatten3 :: ∀ m n o s t. (KnownNat m, KnownNat n, KnownNat o, KnownShape s) => Tensor (m ': n ': o ': s) t -> Tensor (m*n*o ': s) t
flatten3 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m*n*o ': s)])

-- | Reshape a tensor so that the first dimension is expanded into two.
inflate2 :: ∀ m n s t. (KnownNat m, KnownNat n, KnownShape s) => Tensor (m*n ': s) t -> Tensor (m ': n ': s) t
inflate2 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m ': n ': s)])

-- | Reshape a tensor so that the first dimension is expanded into three.
inflate3 :: ∀ m n o s t. (KnownNat m, KnownNat n, KnownNat o, KnownShape s) => Tensor (m*n*o ': s) t -> Tensor (m ': n ': o ': s) t
inflate3 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m ': n ': o ': s)])

-- | Access the last element in a tensor (in the 0th dimension)
last0 :: ∀ n s t. KnownNat n => KnownLen s => T (n ': s) t -> Tensor s t
last0 = nth0 (natVal (Proxy @n) - 1)

-- | Access the nth element in a tensor (in the 0th dimension)
nth0 :: ∀ n s t. KnownNat n => KnownLen s => Integer -> T (n ': s) t -> Tensor s t
nth0 i (T x) = T (x <> list (replicate (fromIntegral (listLen @s)) (text ":") ++ [integer i]))

-- | Take a slice at dimension n from i to j.
slice :: forall n i j s t. KnownNat j => KnownNat i => (i < j, j <= At n s, KnownPeano n, KnownLen s) =>
         Tensor s t -> Tensor (Take n s ++ ((j-i) ': Drop ('Succ n) s)) t
slice (T x) = T (x <> list (replicate (fromIntegral (listLen @s - peanoInt @n - 1)) (text ":") ++ [integer (natVal (Proxy @i)) <> text ".." <> integer (natVal (Proxy @j))]))

slice1 :: forall i j m n s t. KnownNat j => KnownNat i => (i < j, j <= m, KnownLen s) =>
         Tensor (n ': m ': s) t -> Tensor (n ': (j-i) ': s) t
slice1 = slice @Dim1 @i @j

-- | Split a tensors into @n@ tensors along the first dimension
unstack :: ∀ s (n::Nat) t. (KnownLen s, KnownNat n) => Tensor (n ': s) t -> Gen (V n (T s t))
unstack (T x) = do
  v <- newVar
  v <-- funcall "tf.unstack" [x, text "axis=" <> integer (listLen @ s)]
  return $ V $ [ T $ v <> brackets (integer i)| i <- [0..n Prelude.- 1] ]
        where n = natVal (Proxy @ n)

-- | Concatenate @n@ tensors along the first dimension
stack :: ∀ s (n::Nat) t. (KnownLen s) => V n (T s t) -> Tensor (n ': s) t
stack (V xs) = T (funcall "tf.stack" [list [x | T x <- xs], text "axis=" <> integer (listLen @ s)])

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

-- | Size-preserving convolution operation.
convolution :: forall outputChannels filterSpatialShape inChannels s t.
               KnownLen filterSpatialShape => ((1 + Length filterSpatialShape) ~ Length s) -- the last dim of s is the batch size
            => T ('[inChannels] ++ s) t -- ^ input tensor (batched)
            -> T ('[outputChannels,inChannels] ++ filterSpatialShape) t -- ^ filters
            -> T ('[outputChannels] ++ s) t
convolution (T input) (T filters) = T (funcall "tf.nn.convolution" [input,filters
                                                                   ,named "padding" (text (show "SAME")) -- otherwise the shape s changes
                                                                   ,named "data_format" (text (show dataFormat))])
  where dataFormat = case listLen @ filterSpatialShape of
          1 -> "NWC"
          2 -> "NHWC"
          3 -> "NDHWC"
          _ -> error "convolution: more than 3 spatial dimensions are not supported!"

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

-- | sparse softmax cross entropy with logits.
sparseSoftmaxCrossEntropyWithLogits :: Tensor s Int32                   -- ^ desired labels
                                    -> Tensor (numClasses ': s) Float32 -- ^ predictions
                                    -> Tensor s Float32
sparseSoftmaxCrossEntropyWithLogits (T labels) (T logits) =
  T (funcall "tf.nn.sparse_softmax_cross_entropy_with_logits" [named "labels" labels,named "logits" logits])

-- | One hot vector along axis @n@
oneHot :: forall n numClasses s w. KnownNat numClasses => (KnownLen s, KnownPeano n) => Tensor s ('Typ 'Int w) -> Tensor (Take n s ++ (numClasses ': Drop n s)) Float32
oneHot (T x) = T (funcall "tf.one_hot" [x, named "depth" (showDim @numClasses), named "axis" (integer (listLen @s - peanoInt @n))])

-- | One hot vector along axis 0
oneHot0 :: forall numClasses w batchSize. KnownNat numClasses => Tensor '[batchSize] ('Typ 'Int w) -> Tensor '[numClasses,batchSize] Float32
oneHot0 = oneHot @Dim0

-- | One hot vector along axis 1
oneHot1 :: forall numClasses w batchSize m. KnownNat numClasses => Tensor '[m,batchSize] ('Typ 'Int w) -> Tensor '[m,numClasses,batchSize] Float32
oneHot1 = oneHot @Dim1

-- | Generate a random tensor where each individual element is picked
-- in a normal distribution with given standard deviation.
truncatedNormal :: forall s w. KnownShape s => Float -> T s ('Typ 'Float w)
truncatedNormal stddev = T (funcall "tf.truncated_normal" [showShape @s, named "stddev" (float stddev)])

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
varianceScaling sc0 mode distr = case distr of
                                   UniformDistr -> randomUniform (-p) p
                                   NormalDistr -> truncatedNormal p
  where
    fan_in = fromIntegral (natVal (Proxy @inDim))
    fan_out = fromIntegral (natVal (Proxy @outDim))
    sc = sc0 / max 1 (case mode of
                         VSFanIn -> fan_in
                         VSFanOut -> fan_out
                         VSAvg -> (fan_in Prelude.+ fan_out) / 2)
    p = Prelude.sqrt $ (/ sc) $ case distr of
                                  NormalDistr -> 1
                                  UniformDistr -> 3


glorotUniform :: forall inDim outDim t. KnownNat inDim => (KnownNat outDim, KnownBits t) => Tensor '[inDim,outDim] ('Typ 'Float t)
glorotUniform = varianceScaling 1 VSAvg UniformDistr

----------------
-- Helpers
matvecmulBatch :: ∀ s cols rows t. (KnownLen s) =>  Tensor (cols ': rows ': s) t -> Tensor (cols ': s) t -> Tensor (rows ': s) t
matvecmulBatch m v = squeeze0 (matmul m (expandDim0 v))

matvecmul :: Tensor (cols ': rows ': '[]) t -> Tensor (cols ': batchSize ': '[]) t -> Tensor (rows ': batchSize ': '[]) t
matvecmul m v = matmul v (transpose m)

-- | Product of a matrix of weight with a (batched) vector .
(∙) :: Tensor '[cols, rows] t -> Tensor '[cols,batchSize] t -> Tensor '[rows,batchSize] t
m ∙ v = matvecmul m v
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

-------------------------
-- Generic parameters

parameterDefault :: forall p. ParamWithDefault p => String -> Gen p
parameterDefault name = parameter name defaultInitializer

class Parameter p where
  -- | parameterize over tuples of tensors
  parameter :: String -> p -> Gen p

instance (KnownTyp t, KnownShape shape) => Parameter (T shape t) where
  parameter = parameter'

instance (Parameter p, Parameter q) => Parameter (p,q) where
  parameter s (x,y) = (,) <$> parameter (s<>"_fst") x <*> parameter (s<>"_snd") y

instance (Parameter p1, Parameter p2, Parameter p3) => Parameter (p1,p2,p3) where
  parameter s (x,y,z) = (,,) <$> parameter (s<>"_1") x <*> parameter (s<>"_2") y <*> parameter (s<>"_3") z

instance (Parameter p1, Parameter p2, Parameter p3, Parameter p4) => Parameter (p1,p2,p3,p4) where
  parameter s (x,y,z,w) = (,,,) <$> parameter (s<>"_1") x <*> parameter (s<>"_2") y <*> parameter (s<>"_3") z <*> parameter (s<>"_4") w

class Parameter p => ParamWithDefault p where
  defaultInitializer :: p

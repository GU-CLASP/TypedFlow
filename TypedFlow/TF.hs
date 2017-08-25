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
import Text.PrettyPrint.Compact hiding (Last)
import GHC.TypeLits
import Data.Proxy
import TypedFlow.Types

zeros :: ∀ t (shape :: Shape). KnownShape shape => (T shape t)
zeros = T (funcall "tf.zeros" [showShape @shape])

ones :: ∀ t (shape :: Shape). KnownShape shape => (T shape t)
ones = T (funcall "tf.ones" [showShape @shape])

-- | Declare a parameter to optimize.
parameter' :: ∀ (shape :: Shape) t. (KnownTyp t,KnownShape shape) => String -> T shape t -> Gen (T shape t)
parameter' name (T initial) = do
  v <- newVar
  newParameter (ParamInfo name (shapeToList @shape) (typVal @t) (T v))
  v <-- funcall "tf.Variable" [initial, named "name" (string (show (name)))]
  return (T v)

-- TODO: gather the parameters in Haskell
getParameters :: Gen UntypedExpression
getParameters = do
  v <- newVar
  v <-- text "tf.trainable_variables()"
  return v

grad :: T s Float32 -> UntypedExpression -> UntypedExpression
grad (T y) vars = funcall "tf.gradients" [y, vars]

clipByGlobalNorm :: Float -> UntypedExpression -> UntypedExpression
clipByGlobalNorm maxNorm x = funcall "tf.clip_by_global_norm" [x,float maxNorm]

placeholder :: ∀t s. (KnownShape s, KnownTyp t) => String -> Gen (T s t)
placeholder n = do
  let name = text n
  name <-- funcall "tf.placeholder" [showTyp @t, named "shape" (showShape @s)]
  return (T name)

reduceAll :: String -> Tensor s t -> Tensor '[] t
reduceAll op = unOp ("tf.reduce_" ++ op)

reduceMeanAll :: ∀ (s :: Shape) t. Tensor s t -> Tensor '[] t
reduceMeanAll = reduceAll "mean"

reduce :: ∀ s s' n t. KnownLen s' => String -> Tensor (s ++ (n ': s')) t -> Tensor (s ++ s') t
reduce op (T x) = T (funcall ("tf.reduce_" ++ op) [x, text "axis=" <> integer (shapeLen @ s')])

reduceSum, reduceMean :: ∀ s s' n t. KnownLen s' => Tensor (s ++ (n ': s')) t -> Tensor (s ++ s') t
reduceSum = reduce @s @s' @n "sum"
reduceMean = reduce @s @s' @n "mean"

reduceSum0 :: ∀ s' n t. KnownLen s' => Tensor (n ': s') t -> Tensor s' t
reduceSum0 = reduceSum @'[]

add :: ∀ s d t. Tensor (d++s) t -> Tensor d t -> Tensor (d++s) t -- note ++s for for 'broadcasting'
add = binOp "tf.add"

-- add_n :: ∀ s t. [Tensor s t] -> Tensor s t
-- add_n = error "add_n not implemented"

(+) :: ∀ (d :: Shape) (s :: Shape) t. Tensor (d ++ s) t -> Tensor d t -> Tensor (d ++ s) t
(+) = add @s @d

(⊕) :: ∀  (s :: Shape) t. Tensor s t -> Tensor s t -> Tensor s t
(⊕) = binOp "tf.add"

(⊝) :: ∀ (s :: Shape) t. Tensor s t -> Tensor s t -> Tensor s t
(⊝) = binOp "tf.subtract"

multiply :: Tensor d t -> Tensor d t -> Tensor d t
multiply = binOp "tf.multiply"

equal :: Tensor d t -> Tensor d t -> Tensor d TFBool
equal = binOp "tf.equal"

(⊙) :: ∀ (d :: Shape) t. Tensor d t -> Tensor d t -> Tensor d t
(⊙) = multiply

matmul :: Tensor (o ': n ': s) t -> Tensor (m ': o ': s) t -> Tensor (m ': n ': s) t
matmul = binOp "tf.matmul"

round, sigmoid, tanh, log, relu :: ∀ s t. Tensor s ('Typ 'Float t) -> Tensor s ('Typ 'Float t)
sigmoid = unOp "tf.sigmoid"
tanh = unOp "tf.tanh"
log = unOp "tf.log"
relu = unOp "tf.nn.relu"
round = unOp "tf.round"

split0 :: ∀ m n batchShape t. (KnownNat n, KnownNat m, KnownLen batchShape) =>
          Tensor ((n + m) ': batchShape) t -> Gen (Tensor (n ': batchShape) t, Tensor (m ': batchShape) t)
split0 (T x) = do
  v1 <- newVar
  v2 <- newVar
  gen (v1 <> text "," <> v2 <> text " = " <> funcall "tf.split" [x, list [showDim @ n, showDim @ m], text "axis=" <> showShapeLen @batchShape])
  return (T v1, T v2)

concatT :: ∀ n d1 d2 s t.
    (KnownPeano n, KnownLen s, (d1+d2) ~ At n s) =>
    T (Take n s ++ (d1 ': Drop (Succ n) s)) t -> T (Take n s ++ (d2 ': Drop ('Succ n) s)) t -> T s t
concatT (T x) (T y) = T (funcall "tf.concat" [list [x,y], named "axis" (integer (shapeLen @s - peanoInt @n - 1))])

concat0 :: ∀ ys d1 d2 t. (KnownShape ys) =>  T (d1 ': ys) t -> T (d2 ': ys) t -> T ((d1 + d2) ': ys) t
concat0 = concatT @Dim0
  -- let T x = t
  --     T y = u
  -- in (T (funcall "tf.concat" [list [x,y], text "axis=" <> integer (shapeLen @ ys)]))

concat1 :: ∀ n ys d1 d2 t. (KnownShape ys) =>  T (n ': d1 ': ys) t -> T (n ': d2 ': ys) t -> T (n ': (d1 + d2) ': ys) t
concat1 = concatT @Dim1

-- expandDim :: ∀ s0 s t. KnownLen s => Tensor (s0 ++ s) t -> Tensor (s0 ++ (1 ': s)) t
-- expandDim (T x) = (T (funcall "tf.expand_dims" [x, text "axis=" <> integer (shapeLen @ s)]))

expandDim :: forall n s t. (KnownLen s, KnownPeano n) => Tensor s t -> Tensor (Take n s ++ (1 ': Drop n s)) t
expandDim (T x) = (T (funcall "tf.expand_dims" [x, named "axis" (integer (shapeLen @s - peanoInt @n))]))

expandDim0 :: ∀ s t. KnownLen s => Tensor s t -> Tensor (1 ': s) t
expandDim0 = expandDim @Dim0

expandDim1 :: ∀ n s t. KnownShape s => Tensor (n ': s) t -> Tensor (n ': 1 ': s) t
expandDim1 = expandDim @Dim1

tile :: forall m n s t. (KnownNat m) => Tensor (n ': s) t -> Tensor ((m * n) ': s) t
tile (T x) = T (funcall "tf.tile" [x, integer (natVal (Proxy @m))])

replicateT :: ∀ n s t. (KnownNat n, KnownLen s) => T s t -> T (n ': s) t
replicateT = tile @n . expandDim0

squeeze :: ∀ s0 s1 t. KnownLen s1 => Tensor (s0 ++ (1 ': s1)) t -> Tensor (s0 ++ s1) t
squeeze (T x) = T (funcall "tf.squeeze" [x, text "axis=" <> integer (shapeLen @ s1)])

squeeze0 :: ∀ s t. KnownLen s => Tensor (1 ': s) t -> Tensor s t
squeeze0 = squeeze @ '[]

squeeze1 :: ∀ n s t. KnownLen s => Tensor (n ': 1 ': s) t -> Tensor (n ': s) t
squeeze1 = squeeze @ '[n]

-- | Reshape a tensor so that the first two dimensions are collapsed
flatten2 :: ∀ m n s t. (KnownNat m, KnownNat n, KnownShape s) => Tensor (m ': n ': s) t -> Tensor (m*n ': s) t
flatten2 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m*n ': s)])

-- | Reshape a tensor so that the first three dimensions are collapsed
flatten3 :: ∀ m n o s t. (KnownNat m, KnownNat n, KnownNat o, KnownShape s) => Tensor (m ': n ': o ': s) t -> Tensor (m*n*o ': s) t
flatten3 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m*n*o ': s)])

inflate2 :: ∀ m n s t. (KnownNat m, KnownNat n, KnownShape s) => Tensor (m*n ': s) t -> Tensor (m ': n ': s) t
inflate2 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m ': n ': s)])

inflate3 :: ∀ m n o s t. (KnownNat m, KnownNat n, KnownNat o, KnownShape s) => Tensor (m*n*o ': s) t -> Tensor (m ': n ': o ': s) t
inflate3 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m ': n ': o ': s)])

-- | Access the last element in a tensor (in the 0th dimension)
last0 :: ∀ n s t. KnownNat n => KnownLen s => T (n ': s) t -> Tensor s t
last0 (T x) = T (x <> list (replicate (fromIntegral (shapeLen @s)) (text ":") ++ [integer (natVal (Proxy @n) - 1)]))

-- | Take a slice at dimension n from i to j.
slice :: forall n i j s t. KnownNat j => KnownNat i => (i < j, j <= At n s, KnownPeano n, KnownLen s) =>
         Tensor s t -> Tensor (Take n s ++ ((j-i) ': Drop ('Succ n) s)) t
slice (T x) = T (x <> list (replicate (fromIntegral (shapeLen @s - peanoInt @n - 1)) (text ":") ++ [integer (natVal (Proxy @i)) <> text ".." <> integer (natVal (Proxy @j))]))

slice1 :: forall i j m n s t. KnownNat j => KnownNat i => (i < j, j <= m, KnownLen s) =>
         Tensor (n ': m ': s) t -> Tensor (n ': (j-i) ': s) t
slice1 = slice @Dim1 @i @j

unstack :: ∀ s (n::Nat) t. (KnownLen s, KnownNat n) => Tensor (n ': s) t -> Gen (V n (T s t))
unstack (T x) = do
  v <- newVar
  v <-- funcall "tf.unstack" [x, text "axis=" <> integer (shapeLen @ s)]
  return $ V $ [ T $ v <> brackets (integer i)| i <- [0..n Prelude.- 1] ]
        where n = natVal (Proxy @ n)

stack :: ∀ s (n::Nat) t. (KnownLen s) => V n (T s t) -> Tensor (n ': s) t
stack (V xs) = T (funcall "tf.stack" [list [x | T x <- xs], text "axis=" <> integer (shapeLen @ s)])

stackN :: ∀ s (n::Nat) t. V n (T s t) -> Tensor (s ++ '[n]) t
stackN (V xs) = T (funcall "tf.stack" [list [x | T x <- xs], text "axis=0"])

transpose :: ∀ s t. T (Reverse s) t -> T s t
transpose = unOp "tf.transpose"

transposeN :: ∀ s n t. KnownLen s => T (n ': s) t -> T (s ++ '[n]) t
transposeN (T x) = T (funcall "tf.transpose" [x, named "perm" (list (map integer (shapeLen @s:[0.. shapeLen @s-1])))])

transposeN' :: ∀ s n t. KnownLen s => T (s ++ '[n]) t -> T (n ': s) t
transposeN' (T x) = T (funcall "tf.transpose" [x, named "perm" (list (map integer ([1.. shapeLen @s]++[0])))])


gather :: ∀s n indexShape t. T (s ++ '[n]) t -> T indexShape Int32 -> T (s ++ indexShape) t
gather = binOp "tf.gather"

negate :: ∀ s t. T s t -> T s t
negate = unOp "-"

convolution :: forall outputChannels filterSpatialShape inChannels s t.
                KnownLen filterSpatialShape =>
                  ((1 + Length filterSpatialShape) ~ Length s) => -- the last dim of s is the batch size
                  T ('[inChannels] ++ s) t ->
                  T ('[outputChannels,inChannels] ++ filterSpatialShape) t ->
                  T ('[outputChannels] ++ s) t
convolution (T input) (T filters) = T (funcall "tf.nn.convolution" [input,filters
                                                                   ,named "padding" (text (show "SAME")) -- otherwise the shape s changes
                                                                   ,named "data_format" (text (show dataFormat))])
  where dataFormat = case shapeLen @ filterSpatialShape of
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



softmax0 :: T (n ': s) ('Typ 'Float w) -> T (n ': s) ('Typ 'Float w)
softmax0 = unOp "tf.nn.softmax"

softmax1 :: forall n m s w. KnownLen s => T (m ': n ': s) ('Typ 'Float w) -> T (m ': n ': s) ('Typ 'Float w)
softmax1 (T x) = T (funcall "tf.nn.softmax" [x, named "dim" (showShapeLen @s)])

argmax :: forall n u m s t. (KnownLen s, KnownPeano n,KnownBits u) => Tensor (Take n s ++ (m ': Drop n s)) t -> Tensor s ('Typ 'Int u)
-- argmax :: forall s0 u n s t. (KnownLen s,KnownBits u) => T (s0 ++ (n ': s)) ('Typ 'Float t) -> T (s0 ++ s) ('Typ 'Int u)
argmax (T t) = T (funcall "tf.argmax" [t, named "axis" (showShapeLen @ s), named "output_type" (showTyp @('Typ 'Int u))])

argmax0 :: forall u n s t. (KnownLen s, KnownBits u) => T (n ': s) t -> T s ('Typ 'Int u)
argmax0 = argmax @Dim0

argmax1 :: forall u m n s t. (KnownLen s, KnownBits u) => T (m ': n ': s) t -> T (m ': s) ('Typ 'Int u)
argmax1 = argmax @Dim1

cast :: forall u s t. KnownTyp u => T s t -> T s u
cast (T t) = T (funcall "tf.cast" [t, showTyp @ u])


softmaxCrossEntropyWithLogits :: Tensor '[numClasses,batchSize] Float32 -> Tensor '[numClasses,batchSize] Float32 -> Tensor '[batchSize] Float32
softmaxCrossEntropyWithLogits (T labels) (T logits) =
  T (funcall "tf.nn.softmax_cross_entropy_with_logits" [named "labels" labels,named "logits" logits])

oneHot :: forall n numClasses s w. KnownNat numClasses => (KnownLen s, KnownPeano n) => Tensor s ('Typ 'Int w) -> Tensor (Take n s ++ (numClasses ': Drop n s)) Float32
oneHot (T x) = T (funcall "tf.one_hot" [x, named "depth" (showDim @numClasses), named "axis" (integer (shapeLen @s - peanoInt @n))])

oneHot0 :: forall numClasses w batchSize. KnownNat numClasses => Tensor '[batchSize] ('Typ 'Int w) -> Tensor '[numClasses,batchSize] Float32
oneHot0 = oneHot @Dim0

oneHot1 :: forall numClasses w batchSize m. KnownNat numClasses => Tensor '[m,batchSize] ('Typ 'Int w) -> Tensor '[m,numClasses,batchSize] Float32
oneHot1 = oneHot @Dim1

truncatedNormal :: forall s w. KnownShape s => Float -> T s ('Typ 'Float w)
truncatedNormal stddev = T (funcall "tf.truncated_normal" [showShape @s, named "stddev" (float stddev)])

randomUniform :: forall s t. (KnownShape s, KnownTyp t) => Float -> Float -> T s t
randomUniform low high = T (funcall "tf.random_uniform" [showShape @s
                                                        ,named "minval" (float low)
                                                        ,named "maxval" (float high)
                                                        ,named "dtype" (showTyp @t)])

randomOrthogonal :: forall n s t. (KnownBits t, KnownNat n, KnownShape s) => T (n ':s) ('Typ 'Float t)
randomOrthogonal = T (funcall' (funcall "tf.orthogonal_initializer" [named "dtype" (showTyp @('Typ 'Float t))])
                               [named "shape" (showShape @(n ': s))])

constant :: forall s w. KnownShape s => Float -> T s ('Typ 'Float w)
constant c = T (funcall "tf.constant" [float c, named "shape" (showShape @s)])

---------------------------
-- Contrib
data VarianceScaleMode = VSFanIn | VSFanOut | VSAvg
data Distrib = NormalDistr | UniformDistr


varianceScaling :: forall inDim outDim t. KnownNat inDim => (KnownNat outDim, KnownBits t) => Float -> VarianceScaleMode -> Distrib -> Tensor '[inDim,outDim] ('Typ 'Float t)
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

(∙) :: Tensor '[cols, rows] t -> Tensor '[cols,batchSize] t -> Tensor '[rows,batchSize] t
x ∙ y = matvecmul x y

(·) :: ∀ cols batchSize t. Tensor '[cols,batchSize] t -> Tensor '[cols,batchSize] t -> Tensor '[batchSize] t
x · y = reduceSum0 (x ⊙ y)

mapT' :: forall s t r u n. KnownLen r => KnownLen s => KnownNat n => (T s t -> T r u) ->  T (n ': s) t -> Gen (T (n ': r) u)
mapT' f t = do
  xs <- unstack t
  return (stack (fmap f xs))

mapT :: forall s t r u n. KnownTyp u => KnownLen r => KnownLen s => (T s t -> T r u) ->  T (n ': s) t -> Gen (T (n ': r) u)
mapT f x = do
  x' <- mapTN @n f (transposeN @s @n x)
  return (transposeN' @r x')

mapTN :: forall n s t r u. KnownTyp u => (T s t -> T r u) ->  T (s ++ '[n]) t -> Gen(T (r ++ '[n]) u)
mapTN f t = do
  fn <- lambda f
  return (T (funcall "tf.map_fn" [fn, fromTensor t, named "dtype" (showTyp @u)]))

-- TODO: separate harmless and harmful effects. (the big question: are assignments harmful?)

zipWithT :: forall (s :: [Nat]) (t :: Typ) (s1 :: [Nat]) (t1 :: Typ) (s2 :: Shape) (n :: Nat) (t2 :: Typ).
            (KnownLen s, KnownLen s2, KnownLen s1) => KnownTyp t2 =>
                  (T s t -> T s1 t1 -> T s2 t2)
                  -> Tensor (n ': s) t
                  -> Tensor (n ': s1) t1
                  -> Gen (Tensor (n ': s2) t2)
zipWithT f x y = do
  x' <- zipWithTN @n f (transposeN @s @n x) (transposeN @s1 @n y)
  return (transposeN' @s2 x')

zipWithTN :: forall (n :: Nat) (s :: [Nat]) (t :: Typ) (s1 :: [Nat]) (t1 :: Typ) (s2 :: Shape) (t2 :: Typ).
            KnownTyp t2 =>
                  (T s t -> T s1 t1 -> T s2 t2)
                  -> Tensor (s ++ '[n]) t
                  -> Tensor (s1 ++ '[n]) t1
                  -> Gen (Tensor (s2 ++ '[n]) t2)
zipWithTN f (T t) (T u) =  do
  -- xs <- unstack t
  -- ys <- unstack u
  -- return (stack (f <$> xs <*> ys))
  fn <- lambda2 f
  return (T (funcall "tf.map_fn" [fn, tuple [t,u], named "dtype" (showTyp @t2)]))


-- apparently tensorflow (python?) is not aware of 2-argument
-- functions; so we do this... thing.
lambda2 :: (T s t -> T s1 t1 -> T s' t') -> Gen UntypedExpression
lambda2 f = do
  v <- newVar
  let T body = f (T (v <> brackets (int 0))) (T (v <> brackets (int 1)))
  return (text "lambda " <> v <> text ": " <> body)


-------------------------
-- Generic parameters

class Parameter p where
  parameter :: String -> p -> Gen p

instance (KnownTyp t, KnownShape shape) => Parameter (T shape t) where
  parameter = parameter'

instance (Parameter p, Parameter q) => Parameter (p,q) where
  parameter s (x,y) = (,) <$> parameter (s<>"_fst") x <*> parameter (s<>"_snd") y

instance (Parameter p1, Parameter p2, Parameter p3) => Parameter (p1,p2,p3) where
  parameter s (x,y,z) = (,,) <$> parameter (s<>"_1") x <*> parameter (s<>"_2") y <*> parameter (s<>"_3") z

instance (Parameter p1, Parameter p2, Parameter p3, Parameter p4) => Parameter (p1,p2,p3,p4) where
  parameter s (x,y,z,w) = (,,,) <$> parameter (s<>"_1") x <*> parameter (s<>"_2") y <*> parameter (s<>"_3") z <*> parameter (s<>"_4") w

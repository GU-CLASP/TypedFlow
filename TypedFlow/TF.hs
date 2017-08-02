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
parameter' :: ∀ (shape :: Shape) t. String -> T shape t -> Gen (T shape t)
parameter' name (T initial) = do
  v <- newVar
  v <-- T (funcall "tf.Variable" [initial, named "name" (string (show (name)))])
  return (T v)

placeholder :: ∀t s. (KnownShape s, KnownTyp t) => String -> Gen (T s t)
placeholder n = do
  let name = text n
  name <-- T (funcall "tf.placeholder" [showTyp @t, named "shape" (showShape @s)])
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



sigmoid, tanh, log, relu :: ∀ s. Tensor s Float32 -> Tensor s Float32
sigmoid = unOp "tf.sigmoid"
tanh = unOp "tf.tanh"
log = unOp "tf.log"
relu = unOp "tf.nn.relu"

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

expandDim' :: forall n s t. (KnownLen s, KnownPeano n) => Tensor s t -> Tensor (Take n s ++ (1 ': Drop n s)) t
expandDim' (T x) = (T (funcall "tf.expand_dims" [x, text "axis=" <> integer (shapeLen @s - peanoInt @n)]))

expandDim0 :: ∀ s t. KnownLen s => Tensor s t -> Tensor (1 ': s) t
expandDim0 = expandDim' @Dim0

expandDim1 :: ∀ n s t. KnownShape s => Tensor (n ': s) t -> Tensor (n ': 1 ': s) t
expandDim1 = expandDim' @Dim1

tile :: forall m n s t. (KnownNat m) => Tensor (n ': s) t -> Tensor ((m * n) ': s) t
tile (T x) = T (funcall "tf.tile" [x, integer (natVal (Proxy @m))])

replicateT :: ∀ n s t. (KnownNat n, KnownLen s) => T s t -> T (n ': s) t
replicateT = tile @n . expandDim0

-- TODO: same trick as expandDim0
squeeze :: ∀ s0 s1 t. KnownLen s1 => Tensor (s0 ++ (1 ': s1)) t -> Tensor (s0 ++ s1) t
squeeze (T x) = T (funcall "tf.squeeze" [x, text "axis=" <> integer (shapeLen @ s1)])

squeeze0 :: ∀ s t. KnownLen s => Tensor (1 ': s) t -> Tensor s t
squeeze0 = squeeze @ '[]

squeeze1 :: ∀ n s t. KnownLen s => Tensor (n ': 1 ': s) t -> Tensor (n ': s) t
squeeze1 = squeeze @ '[n]

-- | Reshape a tensor so that the first two dimensions are collapsed
linearize2 :: ∀ m n s t. (KnownNat m, KnownNat n, KnownShape s) => Tensor (m ': n ': s) t -> Tensor (m*n ': s) t
linearize2 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m*n ': s)])

-- | Reshape a tensor so that the first three dimensions are collapsed
linearize3 :: ∀ m n o s t. (KnownNat m, KnownNat n, KnownNat o, KnownShape s) => Tensor (m ': n ': o ': s) t -> Tensor (m*n*o ': s) t
linearize3 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m*n*o ': s)])

arrange2 :: ∀ m n s t. (KnownNat m, KnownNat n, KnownShape s) => Tensor (m*n ': s) t -> Tensor (m ': n ': s) t
arrange2 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m ': n ': s)])

arrange3 :: ∀ m n o s t. (KnownNat m, KnownNat n, KnownNat o, KnownShape s) => Tensor (m*n*o ': s) t -> Tensor (m ': n ': o ': s) t
arrange3 (T t) = T (funcall "tf.reshape" [t, showShapeMinus @(m ': n ': o ': s)])

-- | Access the last element in a tensor (in the 0th dimension)
last0 :: ∀ n s t. KnownNat n => KnownLen s => T (n ': s) t -> Tensor s t
last0 (T x) = T (x <> list (replicate (fromIntegral (shapeLen @s)) (text ":") ++ [integer (natVal (Proxy @n) - 1)]))

unstack :: ∀ s (n::Nat) t. (KnownLen s, KnownNat n) => Tensor (n ': s) t -> Gen (V n (T s t))
unstack (T x) = do
  v <- newVar
  v <-- T (funcall "tf.unstack" [x, text "axis=" <> integer (shapeLen @ s)])
  return $ V $ [ T $ v <> brackets (integer i)| i <- [0..n Prelude.- 1] ]
        where n = natVal (Proxy @ n)

stack :: ∀ s (n::Nat) t. (KnownLen s) => V n (T s t) -> Tensor (n ': s) t
stack (V xs) = T (funcall "tf.stack" [(list [x | T x <- xs]), text "axis=" <> integer (shapeLen @ s)])

transpose :: ∀ s t. T (Reverse s) t -> T s t
transpose = unOp "tf.transpose"

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

round :: T s ('Typ 'Float w) -> T s ('Typ 'Float w)
round = unOp "tf.round"


argmax0 :: forall n s. KnownLen s => T (n ': s) Float32 -> T s Int64
argmax0 (T t) = T (funcall "tf.argmax" [t, showShapeLen @ s])


cast :: forall u s t. KnownTyp u => T s t -> T s u
cast (T t) = T (funcall "tf.cast" [t, showTyp @ u])


softmaxCrossEntropyWithLogits :: Tensor '[numClasses,batchSize] Float32 -> Tensor '[numClasses,batchSize] Float32 -> Tensor '[batchSize] Float32
softmaxCrossEntropyWithLogits (T labels) (T logits) =
  T (funcall "tf.nn.softmax_cross_entropy_with_logits" [named "labels" labels,named "logits" logits])


oneHot :: forall numClasses w batchSize. KnownNat numClasses =>
          Tensor '[batchSize] ('Typ 'Int w) -> Tensor '[numClasses,batchSize] Float32
oneHot (T indices) = T (funcall "tf.one_hot" [indices, named "depth" (showDim @numClasses), named "axis" (int 0)])

truncatedNormal :: forall s w. KnownShape s => Float -> T s ('Typ 'Float w)
truncatedNormal stddev = T (funcall "tf.truncated_normal" [showShape @s, named "stddev" (float stddev)])

randomUniform :: forall s t. (KnownShape s, KnownTyp t) => Float -> Float -> T s t
randomUniform low high = T (funcall "tf.random_uniform" [showShape @s
                                                        ,named "minval" (float low)
                                                        ,named "maxval" (float high)
                                                        ,named "dtype" (showTyp @t)])

randomOrthogonal :: forall n s t. (KnownBits t, KnownNat n, KnownShape s) => T (n ':s) ('Typ 'Float t)
randomOrthogonal = T (funcall' (funcall "tf.orthogonal_initializer" [named "shape" (showShape @(n ': s))])
                               [named "dtype" (showTyp @('Typ 'Float t))])

constant :: forall s w. KnownShape s => Float -> T s ('Typ 'Float w)
constant c = T (funcall "tf.constant" [float c, named "shape" (showShape @s)])


---------------------------
-- Contrib


glorotUniform :: forall a b t. (KnownNat a, KnownNat b, KnownTyp t) => Tensor '[a,b] t
glorotUniform = randomUniform low high
  where
    low, high, fan_in, fan_out :: Float
    low = -4.0 Prelude.* Prelude.sqrt(6.0/(fan_in Prelude.+ fan_out)) -- use 4 for sigmoid, 1 for tanh activation 
    high = 4.0 Prelude.* Prelude.sqrt(6.0/(fan_in Prelude.+ fan_out))
    fan_in = fromIntegral (natVal (Proxy @ a))
    fan_out = fromIntegral (natVal (Proxy @ b))


-------------------------
-- Generic parameters

class Parameter p where
  parameter :: String -> p -> Gen p

instance Parameter (T shape t) where
  parameter = parameter'

instance (Parameter p, Parameter q) => Parameter (p,q) where
  parameter s (x,y) = (,) <$> parameter (s<>"_fst") x <*> parameter (s<>"_snd") y

instance (Parameter p1, Parameter p2, Parameter p3) => Parameter (p1,p2,p3) where
  parameter s (x,y,z) = (,,) <$> parameter (s<>"_1") x <*> parameter (s<>"_2") y <*> parameter (s<>"_3") z

instance (Parameter p1, Parameter p2, Parameter p3, Parameter p4) => Parameter (p1,p2,p3,p4) where
  parameter s (x,y,z,w) = (,,,) <$> parameter (s<>"_1") x <*> parameter (s<>"_2") y <*> parameter (s<>"_3") z <*> parameter (s<>"_4") w

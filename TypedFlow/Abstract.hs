{-|
Module      : TypedFlow.Abstract
Description : Abstract Tensor representations
Copyright   : (c) Jean-Philippe Bernardy, 2018
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental

This module provides an abstract representation of tensor
operations. It is not normally imported directly by users.
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

module TypedFlow.Abstract where

import TypedFlow.Python
import Prelude hiding (tanh,Num(..),Floating(..),round,floor,(/),sqrt)
import qualified Prelude
import Prelude ((-))
import GHC.TypeLits
import Data.Proxy
import TypedFlow.Types hiding (T)
import Data.Type.Equality
import Unsafe.Coerce
import Data.Kind (Type,)
import qualified Data.IntMap as I
import System.Mem.StableName
import Data.IORef
import System.IO.Unsafe
import TypedFlow.Types (T(..))

permToFun :: Permutation s t -> Int -> Int
permToFun = \case
  PermId -> \x -> x
  PermTrans a b -> permToFun b . permToFun a
  PermSwap -> \case
    0 -> 1
    1 -> 0
    x -> x
  PermSkip p -> \case
    0 -> 0
    x -> permToFun p (x-1) Prelude.+ 1

appAssocS :: SList' f a -> SList' f b -> SList' f c -> ((a ++ b) ++ c) :~: (a ++ (b ++ c))
appAssocS = unsafeCoerce Refl

broadcastPerm :: Proxy n -> Permutation s t -> Permutation (s ++ '[n]) (t ++ '[n])
broadcastPerm _ PermId = PermId
broadcastPerm n (PermSkip p) = PermSkip (broadcastPerm n p)
broadcastPerm _ PermSwap = PermSwap
broadcastPerm n (PermTrans p q) = PermTrans (broadcastPerm n p) (broadcastPerm n q)

proxyCons :: Proxy x -> Proxy xs -> Proxy (x ': xs)
proxyCons _ _ = Proxy

broadcast :: forall n s t. KnownNat n => Proxy n -> T s t -> T (n ': s) t
broadcast n tensor
  | finished tensor = SimpleBroadcast (Proxy @'[]) n (Proxy @s)  tensor
  | otherwise = case tensor of
  (Where cond x y) -> Where (broadcast n cond) (broadcast n x) (broadcast n y)
  (SimpleBroadcast s0 m s1 x) -> SimpleBroadcast (proxyCons n s0) m s1 (broadcast n x)
  T _ -> error "panic: broadcast constant should be finished!"
  Share x -> Share (broadcast n x)
  ArgMax s0 m s1 x -> ArgMax (LS n s0) m s1 (broadcast n x)
  SoftMax s0 m s1 x -> SoftMax (LS n s0) m s1 (broadcast n x)
  Unbroadcast p x -> case testEqual p n of
     Nothing -> error "panic: Unbroadcast of wrong kind found!"
     Just Refl -> x
  BinOp op s0 s1 s2 s3 x y -> BinOp op (LS n s0) s1 s2 s3 (broadcast n x) (broadcast n y)
  -- MatMul m p o s x y -> MatMul m p o (LS n s) (broadcast n x) (broadcast n y)
  Gather is s0 m s1 x ix
    | finished ix -> Gather is (LS n s0) m s1 (broadcast n x) ix
    | otherwise -> error "broadcast on gather index not implemented"
  Transpose t x -> Transpose (PermSkip t) (broadcast n x)
  -- ReduceBy op s0 m s1 x -> ReduceBy op (LS n s0) m s1 (broadcast n x)
  ReshapeTo s x -> ReshapeTo (LS n s) (broadcast n x)
  Stack s0 m s1 xs -> Stack (LS n s0) m s1 (fmap (broadcast n) xs)
  Concat s0 m o s1 x y -> Concat (LS n s0) m o s1 (broadcast n x) (broadcast n y) 
  -- Index ix s0 m s1 x  -> Index ix (LS n s0) m s1 (broadcast n x)
  Convolution bs inChans outChans filterShape x filters
    | finished filters ->
      prodAssocS n bs (productS (sl filterShape outChans)) $
      prodAssocS n bs (productS (sl filterShape inChans)) $
      knownSList (sl filterShape outChans)  $
      knownSList (sl filterShape inChans)  $
      reshapeFrom (shapeProxySList (LS (proxyMul n bs) (filterShape `sl` outChans))) $
      Convolution (proxyMul n bs) inChans outChans filterShape (reshapeAuto (broadcast n x)) filters
    | otherwise -> error "broadcast on convolution filter not implemented"

proxyMul :: forall n m. Proxy n -> Proxy m -> Proxy (n*m)
proxyMul _ _ = Proxy

testEqual :: KnownNat m => KnownNat n => Proxy m -> Proxy n -> Maybe (m :~: n)
testEqual m n = if natVal m == natVal n then Just (unsafeCoerce Refl) else Nothing

noBroadcast :: a -> a
noBroadcast = id -- FIXME: check

prodAssocS :: forall (x :: Nat) (y :: Nat) (z :: Nat) k (proxy :: Nat -> Type) . proxy x -> proxy y -> proxy z -> (((x * y) * z) ~ (x * (y * z)) => k) -> k
prodAssocS _ _ _ = prodAssoc @x @y @z

productS :: SList s -> Proxy (Product s)
productS _ = Proxy

finished :: T s t -> Bool
finished rec = _

app :: SList' f xs -> SList' f ys -> SList' f (xs ++ ys)
app LZ x = x
app (LS x xs) ys = LS x (app xs ys)

sl :: forall x xs f. SList' f xs -> f x -> SList' f (xs ++ '[x])
sl xs x = app xs (LS x LZ) 


perm210 :: Permutation (n ': m ': o ': s) (m ': o ': n ': s)
perm210 = PermSwap `PermTrans` (PermSkip PermSwap)

perm021 :: Permutation (m ': o ': n ': s) (n ': m ': o ': s) 
perm021 = inversePerm perm210

-- >>> map (permToFun perm210) [0..5::Int]
-- [2,0,1,3,4,5]

inversePerm :: Permutation a b -> Permutation b a
inversePerm PermId = PermId
inversePerm (PermSkip x) = PermSkip (inversePerm x)
inversePerm PermSwap = PermSwap
inversePerm (PermTrans x y) = PermTrans (inversePerm y) (inversePerm x)

atShape :: SList s -> T s t -> T s t
atShape _ x = x

reshapeAuto :: forall s s0 t. KnownLen s => Product s ~ Product s0 => T s0 t -> T s t
reshapeAuto = ReshapeTo (shapeSList @s)

reshapeTo :: forall s s0 t. KnownLen s => Product s ~ Product s0 => SList s -> T s0 t -> T s t
reshapeTo s = ReshapeTo s

reshapeFrom :: forall s s0 t. KnownLen s => Product s ~ Product s0 => Proxy s0 -> T s0 t -> T s t
reshapeFrom _ = ReshapeTo (shapeSList @s)

type SNMap k v = IORef (I.IntMap [(StableName k,v)])

bc :: KnownNat n => Proxy n -> T s t -> T (n : s) t
bc x y = memo (memo broadcast x) y

memo :: (a -> b) -> a -> b
memo f = unsafePerformIO (
  do { tref <- newIORef (I.empty)
     ; return (applyStable f tref)
     })

lk :: StableName k -> I.IntMap [(StableName k,v)] -> Maybe v
lk sn m = do
  x <- I.lookup (hashStableName sn) m
  lookup sn x

applyStable :: (a -> b) -> SNMap a b -> a -> b
applyStable f tbl arg = unsafePerformIO (
  do { sn <- makeStableName arg
     ; lkp <- lk sn <$> readIORef tbl
     ; case lkp of
         Just result -> return result
         Nothing ->
           do { let res = f arg
              ; modifyIORef tbl (I.insertWith (++) (hashStableName sn) [(sn,res)])
              ; return res
              }})


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


-- | Internal. Use 'reduceMeanAll', etc. instead.
reduceAll :: forall s t. String -> Tensor s t -> Tensor '[] t
reduceAll op x = UnOp (Simple1Op ("tf.reduce_" ++ op)) LZ (Proxy @s) (Proxy @'[]) x

-- | Mean value of the input tensor.
reduceMeanAll, reduceSumAll, reduceMaxAll :: ∀ (s :: Shape) t. KnownShape s => Tensor s t -> Tensor '[] t
reduceMaxAll = reduceAll "max"
reduceMeanAll = reduceAll "mean"
reduceSumAll = reduceAll "sum"

-- | Internal. Use 'reduceSum', etc. instead.
reduce :: ∀ n s t. (KnownLen s,KnownPeano n) => String -> T s t -> T (Take n s ++ Drop ('Succ n) s) t
reduce op x = UnOp (Axis1Op ("tf.reduce_" ++ op) (listLen @s)) LZ (Proxy @s)  (Proxy @(Take n s ++ Drop ('Succ n) s)) x

-- | Reduce along a given dimension
reduceSum, reduceMean, reduceMax :: ∀n s t. (KnownLen s,KnownPeano n) => T s t -> T (Take n s ++ Drop ('Succ n) s) t
reduceSum = reduce @n "sum"
reduceMean = reduce @n "mean"
reduceMax = reduce @n "max"


-- | Sum along the first dimension
reduceSum0 :: ∀ s' n t. KnownLen s' => Tensor (n ': s') t -> Tensor s' t
reduceSum0 = reduceSum @Dim0


-- | Add two tensors, broacasting along shape @s@
add :: forall s t. T s t -> T s t -> T s t
add = (⊕)

addN :: ∀ s t. KnownTyp t => KnownShape s => [Tensor s t] -> Tensor s t
addN [] = zeros
addN ts = foldr1 add ts

-- | Add two tensors, broacasting along shape @s@
(+) :: T s t -> T s t -> T s t
(+) = add
infixl 6 +

-- | Divide tensors, broacasting along shape @s@
(/) :: forall s t. T s ('Typ 'Float t) -> T s ('Typ 'Float t) -> T s ('Typ 'Float t)
(/) = BinOp (Simple2Op "tf.divide") LZ (Proxy @s) (Proxy @s) (Proxy @s)
infixl 7 /


-- | Indexwise equality test.
equal :: forall s t. Tensor s t -> Tensor s t -> Tensor s TFBool
equal = BinOp (Simple2Op "tf.equal") LZ (Proxy @s) (Proxy @s) (Proxy @s)

-- | Indexwise operator
(⊕), (⊝), (⊙), (⊘) :: ∀ (s :: Shape) t. Tensor s t -> Tensor s t -> Tensor s t
(⊝) = binOp "tf.subtract"
(⊙) = binOp "tf.multiply"
(⊘) = binOp "tf.divide"
(⊕) = binOp "tf.add"

infixl 7 ⊙,⊘
infixl 6 ⊕,⊝


-- | Matrix multiplication (note that shape @s@ is preserved)
matmul :: forall m n o t. T '[n,o] t -> T '[o,m] t -> T '[n,m] t
matmul = BinOp (Simple2Op "tf.matmul") LZ (Proxy @'[n,o]) (Proxy @[o,m]) (Proxy @[n,m])

unOp :: forall s t. String -> T s t -> T s t
unOp op = UnOp (Simple1Op op) LZ (Proxy @s) (Proxy @s)

binOp :: forall s t. String -> T s t -> T s t -> T s t
binOp op = BinOp (Simple2Op op) LZ (Proxy @s) (Proxy @s) (Proxy @s)

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


-- | Take a slice at dimension n from i to j.
slice :: forall n i j s t. KnownNat j => KnownNat i => (i < j, j <= At n s, KnownPeano n, KnownLen s) =>
         Tensor s t -> Tensor (Take n s ++ ((j-i) ': Drop ('Succ n) s)) t
slice = UnOp (SliceOp (natVal (Proxy @i)) (natVal (Proxy @j))) LZ (Proxy @s) (Proxy @(Take n s ++ ((j-i) ': Drop ('Succ n) s)))


slice1 :: forall i j m n s t. KnownNat j => KnownNat i => (i < j, j <= m, KnownLen s) =>
         Tensor (n ': m ': s) t -> Tensor (n ': (j-i) ': s) t
slice1 = slice @Dim1 @i @j

-- | Concatenate tensors on dimension @n@
concatT :: ∀ n d1 d2 s t. (KnownPeano n, KnownLen s, (d1+d2) ~ At n s) =>
    T (Take n s ++ (d1 ': Drop ('Succ n) s)) t -> T (Take n s ++ (d2 ': Drop ('Succ n) s)) t -> T s t
concatT = BinOp (Axis2Op "tf.concat" (peanoInt @n)) LZ
  (Proxy @(Take n s ++ (d1 ': Drop ('Succ n) s)))
  (Proxy @(Take n s ++ (d2 ': Drop ('Succ n) s)))
  (Proxy @s)

-- | Concatenate tensors on the first dimension
concat0 :: ∀ ys d1 d2 t. (KnownLen ys) => T (d1 ': ys) t -> T (d2 ': ys) t -> T ((d1 + d2) ': ys) t
concat0 = concatT @Dim0

-- | Concatenate tensors on the second dimension
concat1 :: ∀ n ys d1 d2 t. (KnownLen ys) =>  T (n ': d1 ': ys) t -> T (n ': d2 ': ys) t -> T (n ': (d1 + d2) ': ys) t
concat1 = concatT @Dim1

-- | Add an extra dimension at axis (@n@) of size 1.
expandDim :: forall n s t. (KnownLen s, KnownPeano n) => Tensor s t -> Tensor (Take n s ++ (1 ': Drop n s)) t
expandDim = UnOp (Axis1Op "tf.expand_dims" (peanoInt @n)) LZ (Proxy @s) (Proxy @(Take n s ++ (1 ': Drop n s)))

-- | Add an extra dimension at axis (0) of size 1.
expandDim0 :: ∀ s t. KnownLen s => Tensor s t -> Tensor (1 ': s) t
expandDim0 = expandDim @Dim0

-- | Add an extra dimension at axis (1) of size 1.
expandDim1 :: ∀ n s t. KnownShape s => Tensor (n ': s) t -> Tensor (n ': 1 ': s) t
expandDim1 = expandDim @Dim1

reshape :: ∀ s2 s1 t. KnownShape s2 => Product s1 ~ Product s2 => Tensor s1 t -> Tensor s2 t
reshape = unsafeReshape


unsafeReshape :: ∀ s2 s1 t. KnownShape s2 => Tensor s1 t -> Tensor s2 t
unsafeReshape = ReshapeTo (shapeSList @s2)

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
nth0 i = UnOp (IndexOp 0 i) LZ (Proxy @(n ': s)) (Proxy @s)

-- | Access the nth element in a tensor (in the 0th dimension), with a static index
nth0' :: ∀ n m s t. KnownNat n => KnownLen s => n < m => T (m ': s) t -> Tensor s t
nth0' = nth0 (natVal (Proxy @n))


-- -- | Split a tensors into @n@ tensors along the first dimension
-- unstack0 :: ∀ s (n::Nat) t. (KnownLen s, KnownNat n) => Tensor (n ': s) t -> Gen (V n (T s t))
-- unstack0 (T x) = do
--   v <- newVar
--   v <-- funcall "tf.unstack" [x, text "axis=" <> integer (listLen @ s)]
--   return $ V $ [ T $ v <> brackets (integer i)| i <- [0..n Prelude.- 1] ]
--         where n = natVal (Proxy @ n)

stackT :: ∀ s0 s (n::Nat) t. (KnownLen s0) => V n (T (s0 ++ s) t) -> Tensor (s0 ++ (n ': s)) t
stackT = Stack (shapeSList @s0) (Proxy @n) (Proxy @s)

-- | Concatenate @n@ tensors along the first dimension
stack0 :: ∀ s (n::Nat) t. (KnownLen s) => V n (T s t) -> Tensor (n ': s) t
stack0 = stackT @'[]

-- | Concatenate @n@ tensors along the second dimension
stack1 :: ∀ s (n::Nat) m t. (KnownLen s) => V n (T (m ': s) t) -> Tensor (m ': n ': s) t
stack1 = stackT @'[m]

-- -- | Concatenate @n@ tensors along the last dimension
-- stackN :: ∀ s (n::Nat) t. V n (T s t) -> Tensor (s ++ '[n]) t
-- stackN = stackT @s @'[]

-- -- | Transposition. See the type for the permutation of dimensions.
-- transpose :: ∀ s t. T (Reverse s) t -> T s t
-- transpose = unOp "tf.transpose"

permN :: SList s -> Permutation (n ': s) (s ++ '[n])
permN LZ = PermId
permN (LS _n s) = PermSwap `PermTrans` PermSkip (permN s)

permN01 :: SList s -> Proxy m -> Proxy n -> Permutation (s ++ [m,n]) (s ++ [n,m])
permN01 LZ _ _ = PermSwap
permN01 (LS _n s) m n = PermSkip (permN01 s m n)

-- | Transposition. See the type for the permutation of dimensions.
transposeN :: ∀ s n t. KnownLen s => T (n ': s) t -> T (s ++ '[n]) t
transposeN  = Transpose (permN (shapeSList @s))

-- | Transposition. See the type for the permutation of dimensions.
transposeN' :: ∀ s n t. KnownLen s => T (s ++ '[n]) t -> T (n ': s) t
transposeN' = Transpose (inversePerm (permN (shapeSList @s)))

-- | Transposition. See the type for the permutation of dimensions.
transpose01 :: ∀ s m n t. KnownLen s => T (m ': n ': s) t -> T (n ': m ': s) t
transpose01 = Transpose PermSwap

-- | Transposition. See the type for the permutation of dimensions.
transposeN01 :: ∀ s m n t. KnownLen s => T (s ++ [m,n]) t -> T (s ++ [n,m]) t
transposeN01 = Transpose (permN01 (shapeSList @s) (Proxy @m) (Proxy @n))

-- TODO: re-implement
-- -- | Generate a mask of given length for each sequence.
-- sequenceMask :: forall maxlen bs. KnownNat maxlen => Tensor '[bs] Int32 -> Tensor '[maxlen,bs] TFBool
-- sequenceMask (T x) = T (funcall "tf.sequence_mask" [x, named "maxlen" (showDim @maxlen)])


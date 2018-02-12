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

data TF (s :: Shape) (t :: Typ) where
  SimpleBroadcast :: Proxy s0 -> Proxy m ->  Proxy s1 -> TF (s0 ++  s1) t -> TF (s0 ++ (m ': s1)) t
  T :: UntypedExpression -> TF s t
  BinOp :: String -> TF s t -> TF s t -> TF s t
  Unbroadcast :: KnownNat n => Proxy n -> TF (n ': s) t -> TF s t
  ReduceBy :: String -> SList s0 -> Proxy m -> SList s1 -> TF (s0 ++ (m ': s1)) t -> TF (s0 ++ s1)t
  ReshapeFrom :: Product s ~ Product s0 => SList s0 -> TF s0 t -> TF s t
  Transpose :: Permutation s0 s -> TF s0 t -> TF s t
  Share :: TF s t -> TF s t
  Stack :: SList s0 -> Proxy m -> SList s1 -> V m (TF (s0 ++ s1)t) -> TF (s0 ++ (m ': s1))t
  Index :: Int -> SList s0 -> Proxy m ->  SList s1 -> TF (s0 ++ (m ': s1))t -> TF (s0 ++ s1) t
  Concat :: SList s0 -> Proxy m -> Proxy o -> SList s1 -> TF (s0 ++ (m ': s1))t -> TF (s0 ++ (o ': s1))t -> TF (s0 ++ ((m+o) ': s1))t
  Gather :: SList indexShape -> SList s0 -> Proxy m -> SList s1 -> TF (s0 ++ (m ': s1)) t -> TF indexShape ('Typ 'Int w0) -> TF (s0 ++ indexShape ++ s1) t
  MatMul :: KnownLen s => Proxy m -> Proxy n ->  Proxy o -> SList s -> TF (s ++ '[n,o]) t -> TF (s ++ [o,m]) t -> TF (s ++ [n,m]) t
  ArgMax :: SList s0 -> Proxy m -> Proxy s1 -> TF (s0 ++ (m ': s1)) t' -> TF (s0 ++ s1) ('Typ 'Int w)
  SoftMax :: SList s0 -> Proxy m ->  Proxy s1 -> TF (s0 ++ (m ': s1)) t -> TF (s0 ++ (m ': s1)) t
  Where :: TF s TFBool  -> TF s t -> TF s t -> TF s t
  Convolution :: Proxy bs -> Proxy inChannels -> Proxy outChannels -> SList filterSpatialShape
            -> TF (bs ': filterSpatialShape ++ '[inChannels]) t -- ^ input tensor (batched)
            -> TF (filterSpatialShape ++ '[inChannels,outChannels]) t -- ^ filters
            -> TF (bs ': filterSpatialShape ++ '[outChannels]) t

appAssocS :: SList' f a -> SList' f b -> SList' f c -> ((a ++ b) ++ c) :~: (a ++ (b ++ c))
appAssocS = unsafeCoerce Refl

broadcastPerm :: Proxy n -> Permutation s t -> Permutation (s ++ '[n]) (t ++ '[n])
broadcastPerm _ PermId = PermId
broadcastPerm n (PermSkip p) = PermSkip (broadcastPerm n p)
broadcastPerm _ PermSwap = PermSwap
broadcastPerm n (PermTrans p q) = PermTrans (broadcastPerm n p) (broadcastPerm n q)

proxyCons :: Proxy x -> Proxy xs -> Proxy (x ': xs)
proxyCons _ _ = Proxy

broadcast :: forall n s t. KnownNat n => Proxy n -> TF s t -> TF (n ': s) t
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
  BinOp op x y -> BinOp op (broadcast n x) (broadcast n y)
  MatMul m p o s x y -> MatMul m p o (LS n s) (broadcast n x) (broadcast n y)
  Gather is s0 m s1 x ix
    | finished ix -> Gather is (LS n s0) m s1 (broadcast n x) ix
    | otherwise -> error "broadcast on gather index not implemented"
  Transpose t x -> Transpose (PermSkip t) (broadcast n x)
  ReduceBy op s0 m s1 x -> ReduceBy op (LS n s0) m s1 (broadcast n x)
  ReshapeFrom s x -> ReshapeFrom (LS n s) (broadcast n x)
  Stack s0 m s1 xs -> Stack (LS n s0) m s1 (fmap (broadcast n) xs)
  Concat s0 m o s1 x y -> Concat (LS n s0) m o s1 (broadcast n x) (broadcast n y) 
  Index ix s0 m s1 x  -> Index ix (LS n s0) m s1 (broadcast n x)
  Convolution bs inChans outChans filterShape x filters
    | finished filters ->
      prodAssocS n bs (productS (sl filterShape outChans)) $
      prodAssocS n bs (productS (sl filterShape inChans)) $
      knownSList (sl filterShape inChans)  $
      ReshapeFrom (LS (proxyMul n bs) (filterShape `sl` outChans)) $
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

finished :: TF s t -> Bool
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

atShape :: SList s -> TF s t -> TF s t
atShape _ x = x

reshapeAuto :: forall s s0 t. KnownLen s0 => Product s ~ Product s0 => TF s0 t -> TF s t
reshapeAuto = ReshapeFrom (shapeSListProxy (Proxy @s0))

reshapeTo :: forall s s0 t. KnownLen s0 => Product s ~ Product s0 => SList s -> TF s0 t -> TF s t
reshapeTo _ = ReshapeFrom (shapeSListProxy (Proxy @s0))

type SNMap k v = IORef (I.IntMap [(StableName k,v)])

bc :: KnownNat n => Proxy n -> TF s t -> TF (n : s) t
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


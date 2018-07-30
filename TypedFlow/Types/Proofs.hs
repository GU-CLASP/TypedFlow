{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module TypedFlow.Types.Proofs where


import System.IO.Unsafe
import Prelude hiding (RealFrac(..))
import GHC.TypeLits
import Data.Proxy
import TypedFlow.Types hiding (T)
import Data.Type.Equality
import Unsafe.Coerce
import Data.Kind (Type,)

testEqual :: KnownNat m => KnownNat n => Proxy m -> Proxy n -> Maybe (m :~: n)
testEqual m n = if natVal m == natVal n then Just (unsafeCoerce Refl) else Nothing

productS :: forall s. SShape s -> Sat KnownNat (Product s)
productS s = knownSShape s $ knownProduct @s $ Sat

plusComm' :: forall x y. (x + y) :~: (y + x)
plusComm' = unsafeCoerce Refl

plusComm :: forall x y k. ((x + y) ~ (y + x) => k) -> k
plusComm k = case plusComm' @x @y of
  Refl -> k

plusAssoc' :: forall x y z. (x + y) + z :~: x + (y + z)
plusAssoc' = unsafeCoerce Refl

plusAssoc :: forall x y z k. (((x + y) + z) ~ (x + (y + z)) => k) -> k
plusAssoc k = case plusAssoc' @x @y @z of
  Refl -> k

plusAssocS :: forall x y z k px py pz. px x -> py y -> pz z -> (((x + y) + z) ~ (x + (y + z)) => k) -> k
plusAssocS _ _ _ k = case plusAssoc' @x @y @z of
  Refl -> k

prodAssoc' :: forall x y z. (x * y) * z :~: x * (y * z)
prodAssoc' = unsafeCoerce Refl

prodAssoc :: forall (x::Nat) (y::Nat) (z::Nat) k. (((x * y) * z) ~ (x * (y * z)) => k) -> k
prodAssoc k = case prodAssoc' @x @y @z of
  Refl -> k

prodAssocS :: forall x y z k px py pz. px x -> py y -> pz z -> (((x * y) * z) ~ (x * (y * z)) => k) -> k
prodAssocS _ _ _ k = case prodAssoc' @x @y @z of
  Refl -> k


data DimExpr (a :: Nat) (x :: Nat) (b :: Nat) where
  ANat :: Sat KnownNat x -> DimExpr a x (a * x)
  (:*:) :: DimExpr a x b -> DimExpr b y c -> DimExpr a (x*y) c

knownOutputDim :: forall a x b. Sat KnownNat a -> DimExpr a x b -> Sat KnownNat b
knownOutputDim a (ANat x) = satMul a x
knownOutputDim a (x :*: y) = knownOutputDim (knownOutputDim a x) y

dimSat :: DimExpr a x b -> Sat KnownNat x
dimSat (ANat s) = s
dimSat (x :*: y) = dimSat x `satMul` dimSat y

normDim :: forall ws xs ys. DimExpr ws xs ys -> (ws * xs) :~: ys
normDim (ANat _) = Refl
normDim (a :*:b) = case normDim a of Refl -> case normDim b of Refl -> prodAssocS (Proxy @ws) (dimSat a) (dimSat b) Refl

data ShapeExpr (a :: Nat) (x :: Shape) (b :: Nat) where
  Single :: DimExpr a x b -> ShapeExpr a '[x] b
  AShape :: SShape x -> ShapeExpr a x (a * Product x)
  (:++:) :: ShapeExpr a x b -> ShapeExpr b y c -> ShapeExpr a (x++y) c

infixr 5 :++:
infixr 5 *:!
infixr 5 !:*

(!:*) :: DimExpr a x b -> ShapeExpr b xs c -> ShapeExpr a (x ': xs) c
x !:* xs = Single x :++: xs

(*:!) :: ShapeExpr a xs b -> DimExpr b x c -> ShapeExpr a (xs ++ '[x]) c
xs *:! x = xs :++: Single x

exprSShape :: forall a x b. ShapeExpr a x b -> SShape x
exprSShape (AShape s) = s
exprSShape (Single x) = case dimSat x of Sat -> typeSShape
exprSShape (x :++: y) = exprSShape x .+. exprSShape y

normShape :: forall ws xs ys. ShapeExpr ws xs ys -> (ws * Product xs) :~: ys
normShape (Single x) = normDim x
normShape (AShape _) = Refl
normShape (l :++: r) = case normShape l of
                         Refl ->  case normShape r of
                           Refl -> prodHomoS (exprSShape l) (exprSShape r) $
                                   prodAssocS (Proxy @ws) (productS (exprSShape l)) (productS (exprSShape r))
                                   Refl
        -- r :: normShape b y ys ----> (b * y) ~ ys   (1)
        -- l :: normShape ws x b ----> (ws * x) ~ b   (2)
        -- subst (2) in (1): ((ws * x) * y) ~ ys
        -- assoc: (ws * (x * y)) ~ ys

decideProductEq1 :: forall xs zs. ShapeExpr 1 xs zs -> Product xs :~: zs
decideProductEq1 a  = case normShape a of Refl -> Refl

type ShapeX = ShapeExpr 1

decideProductEq :: ShapeExpr 1 xs zs -> ShapeExpr 1 ys zs -> Product xs :~: Product ys
decideProductEq l r = case decideProductEq1 l of
                        Refl -> case decideProductEq1 r of
                          Refl -> Refl

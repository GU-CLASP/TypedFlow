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
{-# LANGUAGE CPP #-}
#if __GLASGOW_HASKELL__ >= 806
{-# LANGUAGE NoStarIsType #-}
#endif
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module TypedFlow.Types.Proofs where


import Prelude hiding (RealFrac(..))
import GHC.TypeLits
import Data.Proxy
import TypedFlow.Types hiding (T)
import Data.Type.Equality
import Unsafe.Coerce
import Data.Kind (Type)
class SingEq s where
  testEq :: forall a b. s a -> s b -> Maybe (a :~: b)

instance SingEq (Sat KnownNat) where
  testEq :: forall n m. Sat KnownNat n -> Sat KnownNat m -> Maybe (n :~: m)
  testEq Sat Sat = testNatEqual (Proxy @n) (Proxy @m)

testNatEqual :: KnownNat m => KnownNat n => Proxy m -> Proxy n -> Maybe (m :~: n)
testNatEqual m n = if natVal m == natVal n then Just (unsafeCoerce Refl) else Nothing

instance SingEq f => SingEq (NP f) where
  testEq Unit Unit = Just Refl
  testEq (x :* xs) (y :* ys) = case (testEq x y, testEq xs ys) of
    (Just Refl, Just Refl) -> Just Refl
    _ -> Nothing
  testEq _ _ = Nothing

instance SingEq SKind where
  testEq SBool SBool = Just Refl
  testEq SInt SInt = Just Refl
  testEq SFloat SFloat = Just Refl
  testEq _ _ = Nothing

instance SingEq SNBits where
  testEq SB32 SB32 = Just Refl
  testEq SB64 SB64 = Just Refl
  testEq _ _ = Nothing

instance SingEq STyp where
  testEq (STyp k b Refl) (STyp k' b' Refl) = case (testEq k k', testEq b b') of
    (Just Refl, Just Refl) -> Just Refl
    _ -> Nothing

-- | Use a reified equality relation
(#>) :: (a :~: b) -> ((a ~ b) => k) -> k
Refl #> k = k
infixr 0 #>

-- | Use a reified arbitrary predicate
(?>) :: Sat constraint a -> (constraint a => k) -> k
Sat ?> k = k
infixr 0 ?>

-- | Use a reified arbitrary constraint
(??>) :: Dict constraint -> (constraint => k) -> k
Dict ??> k = k
infixr 0 ??>

productS :: forall s. SShape s -> Sat KnownNat (Product s)
productS s = knownSShape s ?>
             knownProduct @s ?>
             Sat

plusComm :: forall x y. (x + y) :~: (y + x)
plusComm = unsafeCoerce Refl

plusCommS :: forall x y px py. px x -> py y -> (x + y) :~: (y + x)
plusCommS _ _ = plusComm @x @y

plusAssoc :: forall x y z. (x + y) + z :~: x + (y + z)
plusAssoc = unsafeCoerce Refl

plusAssocS :: forall x y z px py pz. px x -> py y -> pz z -> ((x + y) + z) :~: (x + (y + z))
plusAssocS _ _ _ = plusAssoc @x @y @z

prodAssoc :: forall x y z. (x * y) * z :~: x * (y * z)
prodAssoc = unsafeCoerce Refl

prodAssocS :: forall x y z px py pz. px x -> py y -> pz z -> ((x * y) * z) :~: (x * (y * z))
prodAssocS _ _ _ = prodAssoc @x @y @z

termCancelation :: forall a b. (a + b) - b :~: a
termCancelation = plusMinusAssoc @a @b @b #> cancelation @b #> Refl

plusMinusAssoc :: forall x y z. (x + y) - z :~: x + (y - z)
plusMinusAssoc = unsafeCoerce Refl

cancelation :: (a - a) :~: 0
cancelation = unsafeCoerce Refl

plusMono :: forall a b. (a <=? (a+b)) :~: 'True
plusMono = unsafeCoerce Refl

succPos :: (1 <=? 1+j) :~: 'True
  -- CmpNat 0 (1 + n) :~: 'LT
succPos = unsafeCoerce Refl

succPosProx2 :: forall n proxy a. proxy n a -> (0 :<: (1+n))
succPosProx2 _ = succPos @n

prodHomo ::  forall x y. Product (x ++ y) :~: Product x * Product y
prodHomo = unsafeCoerce Refl

prodHomoS ::  forall x y px py. px x -> py y -> ((Product (x ++ y) :~: (Product x * Product y)))
prodHomoS _ _ = prodHomo @x @y

knownProduct' :: forall s f. All KnownNat s => NP f s -> Sat KnownNat (Product s)
knownProduct' Unit = Sat
knownProduct' (_ :* n) = knownProduct' n ?> Sat

knownProduct :: forall s. KnownShape s => Sat KnownNat (Product s)
knownProduct = knownProduct' @s typeSList

knownSum' :: forall s f. All KnownNat s => NP f s -> Sat KnownNat (Sum s)
knownSum' Unit = Sat
knownSum' (_ :* n) = knownSum' n ?> Sat
knownSum :: forall s. KnownShape s => Sat KnownNat (Sum s)
knownSum = knownSum' @s typeSList

knownPlus :: forall m n. KnownNat m => KnownNat n => Sat KnownNat (m + n)
knownPlus = Sat

takeDrop :: forall s n. (PeanoNat n <= Length s) => (Take n s ++ Drop n s) :~: s
takeDrop = unsafeCoerce Refl

lengthHomo :: forall x y. Length (x ++ y) :~: Length x + Length y
lengthHomo = unsafeCoerce Refl

lengthHomoS :: forall x y proxyx proxyy. proxyx x -> proxyy y -> ((Length (x ++ y) :~: (Length x + Length y)))
lengthHomoS _ _ = lengthHomo @x @y

lengthInit :: forall s. (0 < Length s) => SList s -> ((Length (Init s) + 1) :~: Length s)
lengthInit x = lengthHomo @(Init s) @'[Last s] #> initLast x #> Refl

type a :<=: b = ((a <=? b):~: 'True)
type i :<: j = (i+1) :<=: j

incrPos :: forall x. 1 :<=: (x+1)
incrPos = unsafeCoerce Refl

incrCong :: forall x y. ((x+1) ~ (y+1)) => x :~: y
incrCong = unsafeCoerce Refl

initLast :: forall s. {-(0 < Length s) => FIXME -} SList s -> ((Init s ++ '[Last s]) :~: s)
initLast Unit = error "initLast': does not hold on empty lists"
initLast ((:*) _ Unit) = Refl
initLast ((:*) _ ((:*) y ys)) = initLast ((:*) y ys) #> Refl

initLast' :: forall s. {-(0 < Length s) => FIXME -} KnownShape s => ((Init s ++ '[Last s]) :~: s)
initLast' = initLast (typeSList @s)

appRUnit :: forall s. (s ++ '[]) :~: s
appRUnit = unsafeCoerce Refl

appAssoc ::  ((xs ++ ys) ++ zs) :~: (xs ++ (ys ++ zs))
appAssoc = unsafeCoerce Refl

appAssocS :: forall xs ys zs proxy1 proxy2 proxy3.
             proxy1 xs -> proxy2 ys -> proxy3 zs -> (((xs ++ ys) ++ zs) :~: (xs ++ (ys ++ zs)))
appAssocS _ _ _  = appAssoc @xs @ys @zs


knownLast' :: All KnownNat s => SList s -> (KnownNat (Last s) => k) -> k
knownLast' Unit _ = error "knownLast: does not hold on empty lists"
knownLast' ((:*) _ Unit) k = k
knownLast' ((:*) _ ((:*) y xs)) k = knownLast' ((:*) y xs) k

knownLast :: forall s k. KnownShape s => (KnownNat (Last s) => k) -> k
knownLast = knownLast' @s typeSList

knownInit' :: All KnownNat s => SList s -> Sat KnownShape (Init s)
knownInit' Unit = error "knownLast: does not hold on empty lists"
knownInit' ((:*) _ Unit) = Sat
knownInit' ((:*) _ ((:*) y xs)) = knownInit' ((:*) y xs) ?> Sat

knownInit :: forall s. KnownShape s => Sat KnownShape (Init s)
knownInit = knownInit' @s typeSList

knownTail' :: forall x s k. All KnownNat s => SList (x ': s) -> (KnownShape s => k) -> k
knownTail' ((:*) _ Unit) k = k
knownTail' ((:*) _ ((:*) y xs)) k = knownTail' ((:*) y xs) k

knownTail :: forall s x xs k. (s ~ (x ': xs), KnownShape s) => (KnownShape xs => k) -> k
knownTail = knownTail' @x @xs typeSList

knownAppendS :: forall s t pt. (All KnownNat s, KnownShape t) => SList s -> pt t -> Sat KnownShape (s ++ t)
knownAppendS Unit _t = Sat
knownAppendS ((:*) _ n) t = knownAppendS n t ?> Sat

knownAppend :: forall s t.  (KnownShape s, KnownShape t) => Sat KnownShape (s ++ t)
knownAppend = knownAppendS (typeSList @s) (Proxy @t)


-- knownFmap' :: forall f xs. SList xs -> SList (Ap (FMap f) xs)
-- knownFmap' Unit = Unit
-- knownFmap' ((:*) x n) = (:*) Proxy (knownFmap' @f n)

knownSList :: NP proxy xs -> Sat KnownLen xs
knownSList Unit = Sat
knownSList (_ :* n) = knownSList n ?> Sat

knownSShape :: SShape xs -> Sat KnownShape xs
knownSShape Unit = Sat
knownSShape ((:*) Sat s) = knownSShape s ?> Sat

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
normDim (a :*:b) = normDim a #>
                   normDim b #>
                   prodAssocS (Proxy @ws) (dimSat a) (dimSat b) #>
                   Refl

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
exprSShape (Single x) = dimSat x ?> typeSShape
exprSShape (x :++: y) = exprSShape x .+. exprSShape y

normShape :: forall ws xs ys. ShapeExpr ws xs ys -> (ws * Product xs) :~: ys
normShape (Single x) = normDim x
normShape (AShape _) = Refl
normShape (l :++: r) = normShape l #>
                       normShape r #>
                       prodHomoS (exprSShape l) (exprSShape r) #>
                       prodAssocS (Proxy @ws) (productS (exprSShape l)) (productS (exprSShape r)) #>
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


unsafePositive :: (1 <=? n) :~: 'True
unsafePositive = unsafeCoerce Refl

sucPred :: ((1 <=? n) ~ 'True) => (n - 1) + 1  :~: n
sucPred = unsafeCoerce Refl


natRec :: forall (n :: Nat) (p :: Nat -> Type). KnownNat n => p 0 -> (forall (m :: Nat). p m -> p (m+1)) -> p n
natRec z s = case natVal (Proxy @n) of
  0 -> unsafeCoerce z
  _ -> case unsafePositive @n of
         Refl -> case sucPred @n of
           Refl -> s @(n-1) (natRec @(n-1) @p z s)


data CountRes n where
  CountRes :: Integer -> V n Integer -> CountRes n

vcount :: forall n. KnownNat n => V n Integer
vcount =
  case natRec @n (CountRes (natVal (Proxy @n)-1) VUnit) (\(CountRes m xs) ->
                                                            plusCommS (Proxy @1) (F xs)
                                                            #> CountRes (m-1) (m :** xs)) of
  CountRes _ x -> x

data V n a where
  VUnit :: V 0 a
  (:**) :: a -> V n a -> V (1+n) a
infixr 5 :**

deriving instance (Functor (V n))

instance KnownNat n => Applicative (V n) where
  pure x = fmap (const x) (vcount @n)
  VUnit <*> VUnit = VUnit
  (f :** fs) <*> (a :** as) = succPosProx2 fs #> (f a :** (fs <*> unsafeCoerce as))

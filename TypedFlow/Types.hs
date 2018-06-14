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

module TypedFlow.Types where

import Text.PrettyPrint.Compact hiding (All,Last,Product,Sum)
import GHC.TypeLits
import Unsafe.Coerce
import Data.Proxy
import Control.Monad.State
import Data.Char (toLower)
import Data.Kind (Constraint,Type)
import Data.Type.Equality
import TypedFlow.Memo
import qualified Data.Map as M
import Data.Unique

data Sat (a :: k -> Constraint) (b::k) where
  Sat :: forall b a. a b => Sat a b

instance (Show (Sat a b)) where
  show _ = "Sat"

proxySat :: forall (b::k) (a :: k -> Constraint) proxy. a b => proxy b -> Sat a b
proxySat _ = Sat

natSat :: forall n. KnownNat n => Sat KnownNat n
natSat = Sat @Nat @KnownNat

type DOC = Doc ()

-- type i < j = CmpNat i j ~ 'LT
type i < j = (i+1) <= j
-- type i <= j = (i <=? j) ~ 'True

type family Product xs where
  Product '[] = 1
  Product (x ': xs) = x * Product xs

type family Sum xs where
  Sum '[] = 0
  Sum (x ': xs) = x + Sum xs


type family (++) xs ys where
   '[] ++  xs       = xs
   (x ': xs) ++ ys       = x ': (xs ++ ys)

type family Tail xs where
  Tail (x ': xs) = xs

type family Last xs where
  Last '[x] = x
  Last (x ': xs) = Last xs

type family Init xs where
  Init '[x] = '[]
  Init (x ': xs) = x ': Init xs


-- Some proofs.

-- initLast' :: forall s k. ((Init s ++ '[Last s]) ~ s => k) -> k

-- initLast' k = unsafeCoerce# k -- why not?

termCancelation' :: forall a b. (a + b) - b :~: a
termCancelation' = unsafeCoerce Refl

termCancelation  :: forall a b k. ((((a + b) - b) ~ a) => k) -> k
termCancelation k = case termCancelation' @a @b of Refl -> k

plusMono :: forall a b k. ((a <= (a+b)) => k) -> k
plusMono k = case plusMono' of Refl -> k
  where plusMono' :: (a <=? (a+b)) :~: 'True
        plusMono' = unsafeCoerce Refl

succPos' :: (1 <=? 1+j) :~: 'True
  -- CmpNat 0 (1 + n) :~: 'LT
succPos' = unsafeCoerce Refl

succPos :: forall n k. ((0 < (1+n)) => k) -> k
succPos k = case succPos' @n  of
  Refl -> k


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

prodHomo' ::  forall x y. Product (x ++ y) :~: Product x * Product y
prodHomo' = unsafeCoerce Refl

prodHomo ::  forall x y k. ((Product (x ++ y) ~ (Product x * Product y)) => k) -> k
prodHomo k = case prodHomo' @x @y of Refl -> k

prodHomoS ::  forall x y k px py. px x -> py y -> ((Product (x ++ y) ~ (Product x * Product y)) => k) -> k
prodHomoS _ _ k = case prodHomo' @x @y of Refl -> k

knownProduct' :: forall s k. All KnownNat s => SList s -> (KnownNat (Product s) => k) -> k
knownProduct' Unit k = k
knownProduct' ((:*) _ n) k = knownProduct' n k

knownProduct :: forall s k. KnownShape s => (KnownNat (Product s) => k) -> k
knownProduct = knownProduct' @s typeSList


takeDrop' :: forall s n. (PeanoNat n <= Length s) => SPeano n -> (Take n s ++ Drop n s) :~: s
takeDrop' _ = unsafeCoerce Refl

takeDrop :: forall s n k. (PeanoNat n <= Length s) => SPeano n -> ((Take n s ++ Drop n s) ~ s => k) -> k
takeDrop n k = case takeDrop' @s n of Refl -> k

lengthHomo' :: forall x y. Length (x ++ y) :~: Length x + Length y
lengthHomo' = unsafeCoerce Refl

lengthHomoS :: forall x y k proxyx proxyy. proxyx x -> proxyy y -> ((Length (x ++ y) ~ (Length x + Length y)) => k) -> k
lengthHomoS _ _ k = case lengthHomo' @x @y of Refl -> k

lengthInit' :: forall s k. (0 < Length s) => SList s -> ((Length (Init s) + 1) ~ Length s => k) -> k
lengthInit' x k = case lengthHomo' @(Init s) @'[Last s] of
  Refl -> initLast' x k

lengthInit :: forall s k. KnownLen s => (0 < Length s) => ((Length (Init s) + 1) ~ Length s => k) -> k
lengthInit = lengthInit' (typeSList @s)

incrPos' :: forall x. (1 <=? x+1) :~: 'True
incrPos' = unsafeCoerce Refl

incrPos :: forall x k. ((0 < (x + 1)) => k) -> k
incrPos k = case incrPos' @x of Refl -> k

incrCong' :: forall x y. ((x+1) ~ (y+1)) => x :~: y
incrCong' = unsafeCoerce Refl

incrCong :: forall x y k. ((x+1) ~ (y+1)) => ((x ~ y) => k) -> k
incrCong k = case incrCong' @x @y of Refl -> k


initLast' :: forall s k. {-(0 < Length s) => FIXME -} SList s -> ((Init s ++ '[Last s]) ~ s => k) -> k
initLast' Unit _ = error "initLast': does not hold on empty lists"
initLast' ((:*) _ Unit) k = k
initLast' ((:*) _ ((:*) y ys)) k = initLast' ((:*) y ys) k

initLast :: forall s k. KnownShape s => ((Init s ++ '[Last s]) ~ s => k) -> k
initLast = initLast' @s typeSList

appRUnit' :: forall (s::[t]). (s ++ '[]) :~: s
appRUnit' = unsafeCoerce Refl

appRUnit :: forall (s::[t]) k. (((s ++ '[]) ~ s) => k) -> k
appRUnit k = case appRUnit' @t @s of
  Refl -> k

appEmpty' :: (xs ++ '[]) :~: xs
appEmpty' = unsafeCoerce Refl

appEmpty :: forall xs k. (((xs ++ '[]) ~ xs) => k) -> k
appEmpty k = case appEmpty' @xs of Refl -> k

appAssoc' ::  ((xs ++ ys) ++ zs) :~: (xs ++ (ys ++ zs))
appAssoc' = unsafeCoerce Refl

appAssoc :: forall xs ys zs k. (((xs ++ ys) ++ zs) ~ (xs ++ (ys ++ zs)) => k) -> k
appAssoc k = case appAssoc' @xs @ys @zs of Refl -> k

appAssocS :: forall xs ys zs k proxy1 proxy2 proxy3.
             proxy1 xs -> proxy2 ys -> proxy3 zs -> (((xs ++ ys) ++ zs) ~ (xs ++ (ys ++ zs)) => k) -> k
appAssocS _ _ _  k = case appAssoc' @xs @ys @zs of Refl -> k


knownLast' :: All KnownNat s => SList s -> (KnownNat (Last s) => k) -> k
knownLast' Unit _ = error "knownLast: does not hold on empty lists"
knownLast' ((:*) _ Unit) k = k
knownLast' ((:*) _ ((:*) y xs)) k = knownLast' ((:*) y xs) k

knownLast :: forall s k. KnownShape s => (KnownNat (Last s) => k) -> k
knownLast = knownLast' @s typeSList

knownInit' :: All KnownNat s => SList s -> (KnownShape (Init s) => k) -> k
knownInit' Unit _ = error "knownLast: does not hold on empty lists"
knownInit' ((:*) _ Unit) k = k
knownInit' ((:*) _ ((:*) y xs)) k = knownInit' ((:*) y xs) k

knownInit :: forall s k. KnownShape s => (KnownShape (Init s) => k) -> k
knownInit = knownInit' @s typeSList

knownTail' :: forall x s k. All KnownNat s => SList (x ': s) -> (KnownShape s => k) -> k
knownTail' ((:*) _ Unit) k = k
knownTail' ((:*) _ ((:*) y xs)) k = knownTail' ((:*) y xs) k

knownTail :: forall s x xs k. (s ~ (x ': xs), KnownShape s) => (KnownShape xs => k) -> k
knownTail = knownTail' @x @xs typeSList

splitApp' :: forall ys xs k. SList xs -> ((Take (PeanoLength xs) (xs ++ ys) ~ xs,
                                              Drop (PeanoLength xs) (xs ++ ys) ~ ys) => k) -> k
splitApp' Unit k = k
splitApp' ((:*) _ n) k = splitApp' @ys n k

splitApp :: forall xs ys k. KnownLen xs => ((Take (PeanoLength xs) (xs ++ ys) ~ xs,
                                             Drop (PeanoLength xs) (xs ++ ys) ~ ys) => k) -> k
splitApp = splitApp' @ys (typeSList @xs)

knownAppend' :: forall t s k. (All KnownNat s, KnownShape t) => SList s -> (KnownShape (s ++ t) => k) -> k
knownAppend' Unit k = k
knownAppend' ((:*) _ n) k = knownAppend' @t n k


knownAppend :: forall s t k.  (KnownShape s, KnownShape t) => (KnownShape (s ++ t) => k) -> k
knownAppend = knownAppend' @t (typeSList @s)

-- knownCons :: proxy x -> SList xs -> (KnownLen (x ': xs) => k) -> k
-- knownCons _ Unit k = k
-- knownCons _ ((:*) x n) k = knownCons x n k

-- knownFmap' :: forall f xs. SList xs -> SList (Ap (FMap f) xs)
-- knownFmap' Unit = Unit
-- knownFmap' ((:*) x n) = (:*) Proxy (knownFmap' @f n)

knownSList :: NP proxy xs -> (KnownLen xs => k) -> k
knownSList Unit k = k
knownSList ((:*) _ n) k = knownSList n k

knownSShape :: SShape xs -> (KnownShape xs => k) -> k
knownSShape Unit k = k
knownSShape ((:*) Sat s) k = knownSShape s k

type family Length xs where
  Length '[] = 0
  Length (x ': xs) = 1 + Length xs

type family Reverse' xs ys where
  Reverse' '[] ys = ys
  Reverse' (x ': xs) ys = Reverse' xs (x ': ys )

type family Reverse xs where
  Reverse xs = Reverse' xs '[]

newtype V (n::Nat) a = V [a]
  deriving (Functor, Foldable, Traversable, Show)

lastV :: V (1+n) a -> a
lastV (V xs) = last xs

instance KnownNat n => Applicative (V n) where
  pure = V . replicate (fromIntegral (natVal (Proxy @n)))
  V fs <*> V xs = V (zipWith ($) fs xs)

-- From: https://www.cs.ox.ac.uk/projects/utgp/school/andres.pdf
data NP f (xs :: [k]) where
  Unit :: NP f '[]
  (:*) :: f x -> NP f xs -> NP f (x ': xs)
type SList' = NP


newtype I a = I a
newtype K a x = K a
type HList = NP I

pattern HSingle :: f a -> NP f '[a]
pattern HSingle x = x :* Unit

pattern VecSing :: Tensor s t -> HTV t '[s]
pattern VecSing t1 = F t1 :* Unit

pattern VecPair :: Tensor s t -> Tensor s' t -> HTV t '[s,s']
pattern VecPair t1 t2 = F t1 :* F t2 :* Unit

pattern VecTriple :: Tensor s t -> Tensor s' t -> Tensor s3 t -> HTV t '[s,s',s3]
pattern VecTriple t1 t2 t3 = F t1 :* F t2 :* F t3 :* Unit

type family All (c :: k -> Constraint) (xs :: [k]) :: Constraint where
  All c '[] = ()
  All c (x ': xs) = (c x, All c xs)

class Fun (c :: k -> Constraint)  where
  type Ap c (t :: k) :: l

class Cons (x :: k) (xs :: [k])
instance Fun (Cons x) where type Ap (Cons x) xs = x ': xs

class Snoc (x :: k) (xs :: [k])
instance Fun (Snoc x) where
  type Ap (Snoc x) '[] = '[x]
  type Ap (Snoc x) (y ': ys) = y ': Ap (Snoc x) ys

class FMap (c :: k -> Constraint) (xs :: [k]) where

instance Fun c => Fun (FMap c)  where
  type Ap (FMap c) '[] = '[]
  type Ap (FMap c) (x ': xs) = Ap c x ': Ap (FMap c) xs

-- type family All2 (c :: k -> l -> Constraint) (xs :: [k]) (ys :: [l]) :: Constraint where
--   All2 c '[] '[] = ()
--   All2 c (x ': xs) (y ': ys) = (c x y, All2 c xs ys)
--   All2 c '[] (y ': ys) = 'True ~ 'False
--   All2 c (y ': ys) '[] = 'True ~ 'False

-- | Flip at type level
newtype F g t s = F {fromF :: g s t}


-- | Tensor vector. (Elements in the indexing list are ignored.)
type TV s t = NP (K (Tensor s t))

-- | Heterogeneous tensor vector with the same kind of elements
type HTV t = NP (F T t)

type family Frst (x :: (a,b)) where Frst '(x,y) = x
type family Scnd (x :: (a,b)) where Scnd '(x,y) = y

class (KnownShape (Frst r), KnownTyp (Scnd r)) => KnownPair r where
instance (KnownShape x, KnownTyp y) => KnownPair '(x,y) where

newtype Uncurry g (s :: (a,b)) = Uncurry {fromUncurry :: g (Frst s) (Scnd s)}

-- | Tensor vector heterogenous in types and shapes.
type HHTV = NP (Uncurry T)

hhead :: NP f (x ': xs) -> f x
hhead (x :* _) = x

htail :: NP f (x ': xs) -> NP f xs
htail (_ :* xs) = xs

htmap :: forall f ss t u. (forall s. Tensor s t -> Tensor (Ap f s) u) -> HTV t ss -> HTV u (Ap (FMap f) ss)
htmap _ Unit = Unit
htmap f (F x :* xs) = F (f x) :* htmap @f f xs

-- htmap' :: forall f ss t u. All KnownShape ss => (forall s. KnownShape s => Tensor (Ap f s) t -> Tensor s u) -> SList ss -> HTV t (Ap (FMap f) ss) -> HTV u ss 
-- htmap' _ Unit Unit = Unit
-- htmap' f ((:*) _ n)(F x :* xs) = F (f x) :* htmap' @f f n xs

hmap :: (forall x. f x -> g x) -> NP f xs -> NP g xs
hmap _ Unit = Unit
hmap f (x :* xs) = f x :* hmap f xs

kmap :: (a -> b) -> NP (K a) xs -> NP (K b) xs
kmap _ Unit = Unit
kmap f (K x :* xs) = K (f x) :* kmap f xs


hendo :: NP Endo xs -> HList xs -> HList xs
hendo Unit Unit = Unit
hendo (Endo f :* fs) (I x :* xs) = (I (f x) :* hendo fs xs)

appSList, (.+.), happ :: NP f xs -> NP f ys -> NP f (xs ++ ys)
happ Unit xs = xs
happ (x :* xs) ys = x :* (happ xs ys)
appSList = happ
(.+.) = appSList

(*:) :: forall x xs f. NP f xs -> f x -> NP f (xs ++ '[x])
xs *: x = appSList xs ((:*) x Unit) 

data Both f g x = Both (f x) (g x)

hzip :: NP f xs -> NP g xs -> NP (Both f g) xs
hzip = hzipWith Both

hzipWith :: (forall x. f x -> g x -> h x) -> NP f xs -> NP g xs -> NP h xs
hzipWith _ Unit Unit = Unit
hzipWith f (x :* xs) (y :* ys) = f x y :* hzipWith f xs ys

hfor_ :: Monad m => NP f xs -> (forall x. f x -> m a) -> m ()
hfor_ Unit _  = return ()
hfor_ (x :* xs) f = f x >> hfor_ xs f

htoList :: NP (K a) xs -> [a]
htoList Unit = []
htoList (K x :* xs) = x : htoList xs

hsplit' :: SPeano n -> NP f xs -> (NP f (Take n xs), NP f (Drop n xs))
hsplit' SZero xs = (Unit,xs)
hsplit' (SSucc _n) Unit = (Unit,Unit)
hsplit' (SSucc n) (x :* xs) = case hsplit' n xs of
  (l,r) -> (x :* l,r)

hsplit :: forall xs ys f. KnownLen xs => NP f (xs++ys) -> (NP f xs, NP f ys)
hsplit xys = splitApp @xs @ys (hsplit' (shapePeano @xs) xys)

hsnoc :: NP f xs -> f x -> NP f (xs ++ '[x])
hsnoc xs x = happ xs (x :* Unit)

infixr 5 :*

data Peano = Zero | Succ Peano -- TODO: type Peano = '[()] (And then SPeano = SList') ?

axis0 :: Axis 'Zero (x ': xs)
axis0 = AxZero
axis1 :: Axis ('Succ 'Zero) (x0 ': (x1 ': xs))
axis1 = AxSucc axis0
axis2 :: Axis ('Succ ('Succ 'Zero)) (x0 ': (x1 ': (x2 ': xs)))
axis2 = AxSucc axis1
axis3 :: Axis ('Succ ('Succ ('Succ 'Zero))) (x0 ': (x1 ': (x2 ': (x3 ': xs))))
axis3 = AxSucc axis2


data Axis n xs where
  AxZero :: Axis 'Zero (x ': xs)
  AxSucc :: Axis n xs -> Axis ('Succ n) (x ': xs)

axisInt :: Axis n xs -> Integer
axisInt AxZero = 0
axisInt (AxSucc n) = 1 + axisInt n

sPeanoInt :: SPeano n -> Integer
sPeanoInt (SSucc n) = 1 + sPeanoInt n
sPeanoInt SZero = 0

type family PeanoNat (n::Peano) :: Nat where
  PeanoNat 'Zero = 0
  PeanoNat ('Succ n) = PeanoNat n + 1

data SPeano n where
  SZero :: SPeano 'Zero
  SSucc :: SPeano n -> SPeano ('Succ n)

type family Take n xs where
   Take 'Zero xs            =  '[]
   Take ('Succ n) '[] =  '[]
   Take ('Succ n) (x ': xs) =  x ': Take n xs

type family Drop n xs where
   Drop 'Zero xs            = xs
   Drop _ '[]       = '[]
   Drop ('Succ n) (x ': xs) = Drop n xs

type family At n xs where
  At 'Zero (x ': xs) = x
  At ('Succ n) (x ': xs) = At n xs

-- type family Drop n xs where
--    Drop 'Zero xs            = xs
--    Drop _ '[]       = '[]
--    Drop ('Succ n) (x ': xs) = Drop n xs

-- type family At n xs where
--   At 'Zero (x ': xs) = x
--   At ('Succ n) (x ': xs) = At n xs

data Kind = Float | Int | Bool deriving (Show,Eq,Ord)
data SKind (s::Kind) where
  SFloat :: SKind 'Float
  SInt :: SKind 'Int
  SBool :: SKind 'Bool

data NBits = B32 | B64 | B1 deriving (Show,Eq,Ord)

data SNBits s where
  SB32 :: SNBits 'B32
  SB64 :: SNBits 'B64
  SB1 :: SNBits 'B1

data Typ = Typ Kind NBits deriving (Eq,Ord)

kVal :: SKind t1 -> Kind
kVal SFloat = Float
kVal SInt = Int
kVal SBool = Bool

instance Eq (SKind t) where x == y = kVal x == kVal y
instance Ord (SKind t) where compare x y = compare (kVal x) (kVal y)

nbitsVal :: SNBits w -> NBits
nbitsVal SB1 = B1
nbitsVal SB64 = B64
nbitsVal SB32 = B32

instance Eq (SNBits t) where x == y = nbitsVal x == nbitsVal y
instance Ord (SNBits t) where compare x y = compare (nbitsVal x) (nbitsVal y)

sTypTyp :: STyp t1 -> Typ
sTypTyp (STyp k b) = Typ (kVal k) (nbitsVal b)

instance Eq (STyp t) where x == y = sTypTyp x == sTypTyp y
instance Ord (STyp t) where compare x y = compare (sTypTyp x) (sTypTyp y)

data STyp t where
  STyp :: SKind k -> SNBits b -> STyp ('Typ k b)


type Flt t = 'Typ 'Float t
type Float32 = 'Typ 'Float 'B32
type Int32 = 'Typ 'Int 'B32
type Int64 = 'Typ 'Int 'B64
type TFBool = 'Typ 'Bool 'B1
type Scalar t = T '[] t

instance Show Typ where
  show (Typ Bool _)= "tf.bool"
  show (Typ k l) = "tf." ++ map toLower (show k) ++ drop 1 (show l)

type Shape = [Nat]

class (KnownLen s, All KnownNat s) => KnownShape s where

instance KnownShape '[]
instance (KnownNat x, KnownShape xs) => KnownShape (x ': xs)

class KnownTyp t where
  typeSTyp :: STyp t

class KnownBits t where
  bitsVal :: SNBits t

instance KnownBits 'B1 where bitsVal = SB1
instance KnownBits 'B32 where bitsVal = SB32
instance KnownBits 'B64 where bitsVal = SB64

instance (KnownBits l, KnownKind k) => KnownTyp ('Typ k l) where
  typeSTyp = STyp (kindVal @k) (bitsVal @l)

typVal :: forall t. KnownTyp t => Typ
typVal = sTypTyp (typeSTyp @t)

knownBits :: SNBits t -> (KnownBits t => k) -> k
knownBits SB1 k = k
knownBits SB32 k = k
knownBits SB64 k = k

knownKind :: SKind t -> (KnownKind t => k) -> k
knownKind SFloat k = k
knownKind SInt k = k
knownKind SBool k = k

knownTyp :: STyp t -> (KnownTyp t => k) -> k
knownTyp (STyp k b) r = knownBits b $ knownKind k $ r

class Pretty t where
  pretty :: t -> DOC

instance Pretty Bool where pretty = bool
instance Pretty Float where pretty = float
instance Pretty Int where pretty = int
class (Pretty (HostType t)) => KnownKind t where
  kindVal :: SKind t
  type HostType t

instance KnownKind 'Bool where
  kindVal = SBool
  type HostType 'Bool = Bool
instance KnownKind 'Float where
  kindVal = SFloat
  type HostType 'Float = Float
instance KnownKind 'Int where
  kindVal = SInt
  type HostType 'Int = Int

type SList = NP Proxy

instance Ord (Sat KnownNat t) where
  compare x@Sat y@Sat = compare (natVal x) (natVal y)

instance Eq (Sat KnownNat t) where
   x@Sat == y@Sat = (natVal x) == (natVal y)

type SShape = NP (Sat KnownNat)

instance Ord (SShape s) where
  compare x y = compare (shapeToList' x) (shapeToList' y)

instance Eq (SShape s) where
  Unit == Unit = True
  ((:*) x xs) == ((:*) y ys) = x == y && xs == ys


instance Show (SShape s) where
  show x = show (shapeToList' x)


sListLength :: NP f s -> Integer
sListLength Unit = 0
sListLength ((:*) _ s) = 1+sListLength s


sListLenAsNat :: NP f s -> Sat KnownNat (Length s)
sListLenAsNat Unit = Sat
sListLenAsNat ((:*) _ s) = case sListLenAsNat s of
  Sat -> Sat

type family PeanoLength xs :: Peano where
  PeanoLength '[] = 'Zero
  PeanoLength (x ': xs) = 'Succ (PeanoLength xs)


withKnownNat :: forall k. Int -> (forall (n::Nat). KnownNat n => Proxy n -> k) -> k
withKnownNat 0 f = f (Proxy @0)
withKnownNat 1 f = f (Proxy @1)
withKnownNat n f = withKnownNat (n `div` 2) (if n `mod` 2 == 0 then f2x else f2x1)
  where f2x,f2x1 :: forall (n::Nat). KnownNat n => Proxy n -> k
        f2x  _ = f (Proxy @(n*2))
        f2x1 _ = f (Proxy @(n*2+1))

-- Probably a GHC bug:
-- withKnownNat'' :: forall k. Int -> (forall (n::Nat). KnownNat n => k) -> k
-- withKnownNat'' 0 f = f @0
-- withKnownNat'' n f = withKnownNat'' (n-1) fsucc
--   where fsucc :: forall (n::Nat). KnownNat n =>  k
--         fsucc = f @(n+1)

-- This also fails:
-- appProxy :: forall (n::Nat) k. KnownNat n => Proxy n -> (forall (m::Nat). KnownNat m => k) -> k
-- appProxy f _ = f @n

-- withKnownNat :: forall k. Int -> (forall (n::Nat). KnownNat n => k) -> k
-- withKnownNat n f = withKnownNat' n (\proxy -> appProxy proxy f)

class KnownNat (Length s) => KnownLen s where
  shapePeano :: SPeano (PeanoLength s)
  typeSList :: SList s

instance KnownLen '[] where
  shapePeano = SZero
  typeSList = Unit

instance KnownLen xs => KnownLen (x ': xs) where
  shapePeano = SSucc (shapePeano @xs)
  typeSList = (:*) Proxy (typeSList @xs)

listTypeLen :: forall xs. KnownLen xs => Integer
listTypeLen = sListLength (typeSList @xs)

typeSListProxy :: KnownLen xs => proxy xs -> SList xs
typeSListProxy _ = typeSList

sListProxy :: SList' f xs -> Proxy xs
sListProxy _ = Proxy

knownNatVal :: forall x. Sat KnownNat x -> Integer
knownNatVal Sat = natVal (Proxy @x)

shapeToList' :: SShape s -> [Integer]
shapeToList' Unit = []
shapeToList' ((:*) x xs) = knownNatVal x : shapeToList' xs

shapeToList'' :: All KnownNat s => NP proxy s -> [Integer]
shapeToList'' Unit = []
shapeToList'' ((:*) x xs) = natVal x : shapeToList'' xs

shapeToList :: ∀(s::Shape). KnownShape s => [Integer]
shapeToList = shapeToList'' (typeSList @ s)

typeSShape :: forall s. KnownShape s => SShape s
typeSShape = sListSShape (typeSList @s)

proxySShape :: forall s. KnownShape s => Proxy s -> SShape s
proxySShape _ = typeSShape

sListSShape :: forall s. All KnownNat s => SList s -> SShape s
sListSShape Unit = Unit
sListSShape ((:*) n s) = (:*) (proxySat n) (sListSShape s)

type None = 514229 --  fibonnaci prime.
-- type None = 0 - 1 -- GHC does not like negative Nats.
-- Using a maybe type would be a RPITA.


--------------------------------
-- Generation Effects


data ParamInfo = ParamInfo {paramName :: String
                           ,paramShape :: [Integer]
                           ,paramDType :: Typ
                           ,paramVar   :: forall s t. (KnownShape s, KnownTyp t) => Tensor s t}
data GState = GState {nextVar :: Integer, -- ^ next free variable
                      genText :: DOC,
                      genParams :: [ParamInfo], -- ^ optimizable parameters
                      genRegularizers :: [Scalar Float32], -- ^ accumulated regularizers
                      genTrainingPlaceholder :: Scalar TFBool, -- ^ flag which is true when training
                      genPureTable :: SSNMap2 Shape Typ T DOC,
                      -- ^ Table mapping pointers to their
                      -- interpretations, so that sharing in the data
                      -- structures can be exploited when generating
                      genAssignTable :: M.Map String DOC,
                      -- ^ Table mapping expressions to variables, so
                      -- that lost sharing can be recovered
                      genPeeks :: [(String,UntypedExpression)]}
newtype Gen x = Gen {fromGen :: State GState x} deriving (Monad, MonadState GState, Functor, Applicative)

--------------------------
-- Tensors

type UntypedExpression = DOC

instance Show DOC where
  show = renderWith (Options 92 (const id))

-- | An indexing tensor in the format expected by GatherND
type IndexTensor indexShape containerShape w = T (indexShape ++ '[Length containerShape]) ('Typ 'Int w)

data Distribution (s :: Shape) (t :: Typ) where
  TruncatedNormalD :: Float -> Distribution s ('Typ 'Float w)
  UniformD :: Float -> Float -> Distribution s t
  OrthogonalD  :: Distribution '[m,n] ('Typ 'Float w)

data T (s :: Shape) (t :: Typ) where
  T :: UntypedExpression -> T s t
  Noise :: Integer -> -- this is the unique noise identifier, preventing two different noises to ever be re-shared.
           SShape s0 -> SShape s1 ->
           Distribution s1 t ->
           T (s0 ++ s1) t
  BinOp :: (KnownTyp t, KnownTyp u) => BinOp -> SShape s0 -> SShape s1 -> SShape s2 -> SShape s3 -> T (s0 ++ s1) t -> T (s0 ++ s2) u -> T (s0 ++ s3) v
  UnOp :: KnownTyp t => UnOp -> SShape s0 -> SShape s1 -> SShape s2 -> T (s0 ++ s1) t -> T (s0 ++ s2) u
  Unbroadcast :: Sat KnownNat n -> Unique -> T (n ': s) t -> T s t
  DirectBroadcast :: SShape s0 -> NP proxy' s1 -> SShape s2 -> NP proxy' s3 -> T (s0 ++ s2) t -> T (s0 ++ (s1 ++ (s2 ++ s3))) t
  ReshapeFrom :: Product s ~ Product s0 => SShape s0 -> T s0 t -> T s t
  Transpose :: SShape s0 -> Permutation s0 s -> T s0 t -> T s t
  Stack :: SShape s0 -> Sat KnownNat m -> SShape s1 -> V m (T (s0 ++ s1) t) -> T (s0 ++ (m ': s1)) t
  Gather :: KnownBits w => SShape indexShape -> SShape s0 -> Sat KnownNat m -> SShape s1
    -> T (s0 ++ (m ': s1)) t -> T indexShape ('Typ 'Int w) -> T (s0 ++ indexShape ++ s1) t
  GatherND :: KnownBits w => SShape containerShape -> SShape elementShape -> SShape indexShape
    -> T (containerShape ++ elementShape) t -> IndexTensor indexShape containerShape w -> T (indexShape ++ elementShape) t

  MatMul :: forall s m n o t. SShape s -> Sat KnownNat n -> Sat KnownNat  o -> Sat KnownNat m -> T (s ++ '[n,o]) t -> T (s ++ [o,m]) t -> T (s ++ [n,m]) t
  Where :: T s TFBool  -> T s t -> T s t -> T s t
  If :: Scalar TFBool -> T s t -> T s t -> T s t
  Convolution :: Sat KnownNat bs -> Sat KnownNat inChannels -> Sat KnownNat outChannels -> SShape filterSpatialShape -> SShape s
            -> T (bs ': s ++ '[inChannels]) t -- input tensor (batched)
            -> T (filterSpatialShape ++ '[inChannels,outChannels]) t -- filters
            -> T (bs ': s ++ '[outChannels]) t
  Pool :: Length outSpatial ~ Length window =>
          Sat KnownNat bs -> SShape window -> PoolingType -> Sat KnownNat numChannels -> SShape outSpatial
            -> T (bs ': ZipWithMulShapes window outSpatial ++ '[numChannels]) t
            -> T (bs ': outSpatial ++ '[numChannels]) t

instance Show Unique where
  show _ = "<Unique>"

-- deriving instance (Show (T s t))

type family ZipWithMulShapes (xs::Shape) (xy::Shape) :: Shape
type instance ZipWithMulShapes (x ': xs) (y ': ys) = x*y ': ZipWithMulShapes xs ys
type instance ZipWithMulShapes '[] _ = '[]
type instance ZipWithMulShapes _ '[] = '[]

satMul :: forall n m. Sat KnownNat n -> Sat KnownNat m -> Sat KnownNat (n*m)
satMul Sat Sat = Sat

satAdd :: forall n m. Sat KnownNat n -> Sat KnownNat m -> Sat KnownNat (n+m)
satAdd Sat Sat = Sat

zipWithMulSShapes :: SShape xs -> SShape ys -> SShape (ZipWithMulShapes xs ys)
zipWithMulSShapes Unit _ = Unit
zipWithMulSShapes _ Unit = Unit
zipWithMulSShapes ((:*) x xs) ((:*) y ys) = (:*) (satMul x y) (zipWithMulSShapes xs ys)

data PoolingType = MaxPool | AvgPool deriving Show

type Tensor shape = T shape

data UnOp  = Simple1Op String [DOC] | SliceOp Integer Integer | Axis1Op String [(String,DOC)] Integer | IndexOp {indexOpAxis :: Integer, indexOpIndex :: Integer}
             deriving Show
data BinOp = Simple2Op String (Maybe (String,String)) | Axis2Op String Integer deriving Show

data Permutation (s :: [k]) (t :: [k]) where
  PermId :: Permutation s t
  PermSkip :: Permutation s t -> Permutation (n ': s) (n ': t)
  PermSwap :: Permutation (n ': m ': s) (m ': n ': s)
  PermTrans :: Permutation s t -> Permutation t u -> Permutation s u

deriving instance Show (Permutation s t)

class KnownTensors p where
  -- | traverse all the tensors contained in p.
  travTensor :: Monad m => (forall s t. (KnownTyp t, KnownShape s) => String -> T s t -> m (T s t)) -> String -> p -> m p 

instance (KnownTyp t, KnownShape shape) => KnownTensors (T shape t) where
  travTensor f = f

instance (KnownTyp t, All KnownShape ys) => KnownTensors (HTV t ys) where
  travTensor :: forall m. Monad m => (forall s t'. (KnownTyp t', KnownShape s) => String -> T s t' -> m (T s t')) -> String -> (HTV t ys) -> m (HTV t ys)
  travTensor f s = ttr 0
    where ttr :: forall xs. All KnownShape xs => Int -> HTV t xs -> m (HTV t xs)
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
  defaultInitializer :: Gen p

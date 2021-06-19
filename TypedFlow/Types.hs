{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE CPP #-}
#if __GLASGOW_HASKELL__ >= 806
{-# LANGUAGE NoStarIsType #-}
#endif
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
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ApplicativeDo #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module TypedFlow.Types where

import GHC.TypeLits 
import Data.Proxy
import Control.Monad.State
import Control.Monad.RWS (RWS(..), local, ask, tell)
import Data.Kind (Constraint,Type)
import qualified Data.Int as Hask
import Data.Type.Equality
import Data.Monoid hiding (Sum,Product,Last,All,Ap)

newtype (∘) f (g :: k -> k2) (a::k) where
  Comp :: forall f g a. f (g a) -> (f ∘ g) a

type Sat = (∘) Dict
type Sat' f x = f x

data Dict :: Constraint -> Type where
  Dict :: a => Dict a

pattern Sat :: forall k (g :: k -> Constraint) (a :: k). () => g a => (∘) Dict g a -- the second context is the PROVIDED constraint!
pattern Sat = Comp Dict

instance (Show (Sat a b)) where
  show _ = "Sat"

proxySat :: forall k (b::k) (a :: k -> Constraint) proxy. a b => proxy b -> Sat a b
proxySat _ = Sat

natSat :: forall n. KnownNat n => Sat KnownNat n
natSat = Sat

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



type family Length xs where
  Length '[] = 0
  Length (x ': xs) = 1 + Length xs

type family Reverse' xs ys where
  Reverse' '[] ys = ys
  Reverse' (x ': xs) ys = Reverse' xs (x ': ys )

type family Reverse xs where
  Reverse xs = Reverse' xs '[]

-- From: https://www.cs.ox.ac.uk/projects/utgp/school/andres.pdf
data NP f (xs :: [k]) where
  Unit :: NP f '[]
  (:*) :: f x -> NP f xs -> NP f (x ': xs)

deriving instance (forall x. Show (f x)) => Show (NP f xs)
type SList' = NP

(.+.) = appSList
infixr 5 .+.
infixr 5 *:
infixr 5 :*
(*:) :: forall x xs f. NP f xs -> f x -> NP f (xs ++ '[x])
xs *: x = appSList xs (x :* Unit)

hlookup :: Axis n xs -> NP f xs -> f (At n xs)
hlookup AxZero  (x :* _) = x
hlookup (AxSucc n) (_ :* xs) = hlookup n xs

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

knownAll :: forall constraint s k. NP (Sat constraint) s -> (All constraint s => KnownLen s => k) -> k
knownAll Unit k = k
knownAll (Sat :* xs) k = knownAll xs $ k

allKnown' :: forall constraint s proxy. All constraint s => NP proxy s -> NP (Sat constraint) s
allKnown' Unit = Unit
allKnown' (_ :* xs) = Sat :* allKnown' xs

allKnown :: forall k s. KnownLen s => All k s => NP (Sat k) s
allKnown = allKnown' typeSList

class Fun (c :: k -> Constraint)  where -- FIXME: use type, not constraint?
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

mapFMap :: forall g f xs. (forall x. f x -> f (Ap g x)) -> NP f xs -> NP f (Ap (FMap g) xs)
mapFMap _ Unit = Unit
mapFMap f (x :* xs) = f x :* mapFMap @g @f f xs

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

class Scnd' (x::(a,b))
instance Fun (Scnd') where type Ap Scnd' '(a,b) = b

class Frst' (x::(a,b))
instance Fun (Frst') where type Ap Frst' '(a,b) = a


type family Frst (x :: (a,b)) where Frst '(x,y) = x
type family Scnd (x :: (a,b)) where Scnd '(x,y) = y

type family Frst3 (x :: (a,b,c)) where Frst3 '(x,y,z) = x
type family Scnd3 (x :: (a,b,c)) where Scnd3 '(x,y,z) = y
type family Thrd3 (x :: (a,b,c)) where Thrd3 '(x,y,z) = z

class (KnownShape (Scnd3 r), KnownTyp (Thrd3 r), KnownSymbol (Frst3 r)) => KnownPlaceholder r
instance (KnownShape y, KnownTyp z, KnownSymbol x) => KnownPlaceholder '(x,y,z)
class (KnownShape (Frst r), KnownTyp (Scnd r)) => KnownPair r
instance (KnownShape x, KnownTyp y) => KnownPair '(x,y)

newtype Uncurry g (s :: (a,b)) = Uncurry {fromUncurry :: g (Frst s) (Scnd s)}

-- | Tensor vector heterogenous in types and shapes.
type HHTV = NP (Uncurry T)

type Placeholders = NP Placeholder

newtype Placeholder (s :: (Symbol,Shape,Typ)) = PHT (T (Scnd3 s) (Thrd3 s))

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

-- | Map a natural transformation
hmap :: (forall x. f x -> g x) -> NP f xs -> NP g xs
hmap _ Unit = Unit
hmap f (x :* xs) = f x :* hmap f xs

hTraverse :: Applicative m => (forall x. f x -> m (g x)) -> NP f xs -> m (NP g xs)
hTraverse _ Unit = pure Unit
hTraverse f (x :* xs) = do
  x' <- f x
  xs' <- hTraverse f xs
  return (x' :* xs')

-- | Variant of hmap with a constraint
hmapK :: forall k f g xs. All k xs => (forall x. k x => f x -> g x) -> NP f xs -> NP g xs
hmapK _ Unit = Unit
hmapK f (x :* xs) = f x :* hmapK @k f xs

-- | If NP is in fact a vector, we have a "usual" map.
kmap :: (a -> b) -> NP (K a) xs -> NP (K b) xs
kmap _ Unit = Unit
kmap f (K x :* xs) = K (f x) :* kmap f xs

-- | If NP is in fact a tuple, we can apply a tuple of endomorphisms. (special case of <*>)
hendo :: NP Endo xs -> HList xs -> HList xs
hendo Unit Unit = Unit
hendo (Endo f :* fs) (I x :* xs) = (I (f x) :* hendo fs xs)

appSList, (.+.), happ :: NP f xs -> NP f ys -> NP f (xs ++ ys)
happ Unit xs = xs
happ (x :* xs) ys = x :* (happ xs ys)
appSList = happ

data Both f g x = Both {frst :: f x, scnd :: g x}

bothFromPair :: (f x, g x) -> Both f g x
bothFromPair (x,y) = (Both x y)

bothToPair :: Both f g x -> (f x, g x)
bothToPair (Both x y)  = (x,y)


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

splitApp' :: forall ys xs k. SList xs -> ((Take (PeanoLength xs) (xs ++ ys) ~ xs,
                                              Drop (PeanoLength xs) (xs ++ ys) ~ ys) => k) -> k
splitApp' Unit k = k
splitApp' ((:*) _ n) k = splitApp' @ys n k

splitApp :: forall xs ys k. KnownLen xs => ((Take (PeanoLength xs) (xs ++ ys) ~ xs,
                                             Drop (PeanoLength xs) (xs ++ ys) ~ ys) => k) -> k
splitApp = splitApp' @ys (typeSList @xs)

hsnoc :: NP f xs -> f x -> NP f (xs ++ '[x])
hsnoc xs x = happ xs (x :* Unit)


data Peano = Zero | Succ Peano -- TODO: type Peano = '[()] (And then SPeano = NP) ?

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

data Typ = Typ Kind NBits deriving (Eq,Ord)
type family TypKind (t :: Typ) where TypKind ('Typ k b)  = k
type family TypBits (t :: Typ) where TypBits ('Typ k b)  = b

type KnownNumeric t = (NumericKind (TypKind t), KnownBits (TypBits t), t ~ 'Typ (TypKind t) (TypBits t))
type KnownFloating t = (TypKind t ~ 'Float, KnownBits (TypBits t), t ~ 'Typ 'Float (TypBits t))


class KnownKind t => NumericKind t where
instance NumericKind 'Float
instance NumericKind 'Int

kVal :: SKind t1 -> Kind
kVal SFloat = Float
kVal SInt = Int
kVal SBool = Bool

instance Eq (SKind t) where x == y = kVal x == kVal y
instance Ord (SKind t) where compare x y = compare (kVal x) (kVal y)

nbitsVal :: SNBits w -> NBits
nbitsVal SB64 = B64
nbitsVal SB32 = B32

instance Eq (SNBits t) where x == y = nbitsVal x == nbitsVal y
instance Ord (SNBits t) where compare x y = compare (nbitsVal x) (nbitsVal y)

sTypTyp :: STyp t1 -> Typ
sTypTyp (STyp k b Refl) = Typ (kVal k) (nbitsVal b)

instance Eq (STyp t) where x == y = sTypTyp x == sTypTyp y
instance Ord (STyp t) where compare x y = compare (sTypTyp x) (sTypTyp y)

data STyp t where
  STyp :: SKind (TypKind t) -> SNBits (TypBits t) -> (t :~: 'Typ (TypKind t) (TypBits t)) -> STyp t

type Flt t = 'Typ 'Float t
type Float32 = 'Typ 'Float 'B32
type Int32 = 'Typ 'Int 'B32
type Int64 = 'Typ 'Int 'B64
type TFBool = 'Typ 'Bool 'B32
type Scalar t = T '[] t

type Shape = [Nat]

class (KnownLen s, All KnownNat s) => KnownShape s where

instance KnownShape '[]
instance (KnownNat x, KnownShape xs) => KnownShape (x ': xs)

type KnownTyp t = (KnownBits (TypBits t), KnownKind (TypKind t), t ~ 'Typ (TypKind t) (TypBits t))

typeSTyp :: forall t. KnownTyp t => STyp t
typeSTyp = STyp (kindVal @(TypKind t)) (bitsVal @(TypBits t)) Refl

type family HaskType t where
  HaskType Float32 = Float
  HaskType ('Typ 'Float 'B64) = Double
  HaskType ('Typ 'Int 'B64) = Hask.Int64
  HaskType ('Typ 'Int 'B32) = Hask.Int32
  HaskType ('Typ 'Bool w) = Bool

class KnownBits t where
  bitsVal :: SNBits t

instance KnownBits 'B32 where bitsVal = SB32
instance KnownBits 'B64 where bitsVal = SB64

typVal :: forall t. KnownTyp t => Typ
typVal = Typ (kVal k) (nbitsVal b)
  where k = kindVal @(TypKind t)
        b = bitsVal @(TypBits t)

knownBits :: SNBits t -> (KnownBits t => Fractional (HaskType ('Typ 'Float t)) => Floating (HaskType ('Typ 'Float t)) => k) -> k
knownBits SB32 k = k
knownBits SB64 k = k

knownKind :: SKind t -> (KnownKind t => k) -> k
knownKind SFloat k = k
knownKind SInt k = k
knownKind SBool k = k

knownTyp :: STyp t -> (KnownTyp t => k) -> k
knownTyp (STyp k b Refl) r = knownKind k $ knownBits b r

knownFloating :: forall w k. KnownBits w => (Fractional (HaskType ('Typ 'Float w)) => Floating (HaskType ('Typ 'Float w)) => k) -> k
knownFloating = knownBits (bitsVal @w) 

knownNum :: forall t k. KnownNumeric t => (KnownTyp t => Num (HaskType t) => k) -> k
knownNum k = case kindVal @(TypKind t) of
  SFloat -> case bitsVal @(TypBits t) of
    SB32 -> k
    SB64 -> k
  SBool -> error "KnownNumeric bug"
  SInt -> case bitsVal @(TypBits t) of
    SB32 -> k
    SB64 -> k

class KnownKind t where kindVal :: SKind t
instance KnownKind 'Bool where kindVal = SBool
instance KnownKind 'Float where kindVal = SFloat
instance KnownKind 'Int where kindVal = SInt

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


instance {-# OVERLAPPING #-} Show (SShape s) where
  show x = show (shapeToList' x)


sListLength :: NP f s -> Integer
sListLength Unit = 0
sListLength ((:*) _ s) = 1+sListLength s

sListLen :: NP f s -> Int
sListLen = fromIntegral . sListLength

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

sListProxy :: NP f xs -> Proxy xs
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
sListSShape = allKnown'



type None = 514229 --  fibonnaci prime.
-- type None = 0 - 1 -- GHC does not like negative Nats.
-- Using a maybe type would be a RPITA.


--------------------------------
-- Generation Effects (TODO: move to other module)

data VarInfo = forall s t. VarInfo {varTrainable :: Bool,
                                    varName :: String,
                                    varRef :: Ref s t,
                                    varInitial :: Maybe (T s t)} 

data GState = GState {nextVar :: Integer, -- ^ next free variable
                      genRegularizers :: [Scalar Float32] -- ^ accumulated regularizers
                     }
initialGstate :: GState
initialGstate = (GState {nextVar = 0
                        ,genRegularizers=[]
                        })



  
data Gen a where
  GPId :: Gen Integer
  GPVariable :: forall (shape :: Shape) t. (KnownTyp t,KnownShape shape) => Bool -> String -> Maybe (T shape t) -> Gen (Ref shape t) 
  GPModify :: (KnownShape s,KnownTyp t) => Ref s t -> T s t -> Gen (T s t)
  GPReturn :: a -> Gen a
  GPState :: (GState -> (a,GState)) -> Gen a
  GPApp :: (Gen (a -> b)) -> Gen a -> Gen b
  GPBind :: Gen a -> (a -> Gen b) -> Gen b


genGets :: (GState -> a) -> Gen a
genGets f = GPState  (\s -> (f s, s))

instance Applicative Gen where
  (<*>) = GPApp
  pure = GPReturn

instance Monad Gen where
  (>>=) = GPBind

instance Functor Gen where
  fmap f = (pure f <*>)


--------------------------
-- Tensors

-- | An indexing tensor in the format expected by GatherND
type IndexTensor indexShape containerShape w = T (indexShape ++ '[Length containerShape]) ('Typ 'Int w)

-- | Description of a random distribution
data Distribution (s :: Shape) (t :: Typ) where
  -- | Each element is from a truncated normal distribution with given standard dev.
  TruncatedNormalD :: Float ->  Distribution s ('Typ 'Float w)
  -- | Each element is from a uniform distribution with given bounds (low, high)
  UniformD :: Float -> Float -> Distribution s ('Typ 'Float w)
  OrthogonalD  :: Distribution '[m,n] ('Typ 'Float w)

data Ref s t = Ref Int (SShape s) (STyp t)

data NilOp s t where
  Magic :: String -> NilOp s t
  Variable :: Ref s t -> NilOp s t
  Constant :: HaskType t -> NilOp '[] t
  Range :: KnownBits w => Sat KnownNat n -> NilOp '[n] ('Typ 'Int w)

data Catable s1 s2 t n = Catable (Sat KnownNat n) (T (s1 ++ (n ': s2)) t)
  -- deriving Show


type Unique = Integer

data T (s :: Shape) (t :: Typ) where
  MapT :: KnownTyp t => Sat KnownNat n -> SShape s -> (T s t -> T r u) ->  T (n ': s) t -> T (n ': r) u
  ZipT :: (KnownTyp t, KnownTyp u) => Sat KnownNat n -> SShape s -> SShape r -> (T s t -> T r u -> T q v) ->  T (n ': s) t -> T (n ': r) u -> T (n ': q) v
  Zip3T :: (KnownTyp t, KnownTyp u, KnownTyp v) => Sat KnownNat n -> SShape s -> SShape r -> SShape q -> (T s t -> T r u -> T q v -> T p w) ->  T (n ': s) t -> T (n ': r) u -> T (n ': q) v -> T (n ': p) w
  T :: NilOp s t -> T s t
  Noise :: Integer -> -- this is the unique noise identifier, preventing two different noises to ever be re-shared.
           SShape s0 -> SShape s1 ->
           Distribution s1 t ->
           T (s0 ++ s1) t
  BinOp :: (KnownTyp t,KnownTyp u) => BinOp s1 t s2 u s3 v -> SShape s0 -> SShape s1 -> STyp t -> SShape s2 -> STyp u -> T (s0 ++ s1) t -> T (s0 ++ s2) u -> T (s0 ++ s3) v
  UnOp :: KnownTyp t => UnOp s1 t s2 u -> SShape s0 -> T (s0 ++ s1) t -> T (s0 ++ s2) u
  Unbroadcast :: Sat KnownNat n -> Unique -> T (n ': s) t -> T s t
  DirectBroadcast :: SShape s0 -> NP proxy' s1 -> SShape s2 -> NP proxy' s3 -> T (s0 ++ s2) t -> T (s0 ++ (s1 ++ (s2 ++ s3))) t
  ReshapeFrom :: Product s ~ Product s0 => SShape s0 -> T s0 t -> T s t
  Transpose :: SShape s0 -> Permutation s0 s -> T s0 t -> T s t
  Concat :: SShape s0 -> SShape s1 -> NP (Catable s0 s1 t) ns -> T (s0 ++ (Sum ns ': s1)) t
  Gather :: KnownTyp ('Typ 'Int w) => SShape indexShape -> SShape s0 -> Sat KnownNat m -> SShape s1
    -> T (s0 ++ (m ': s1)) t -> T indexShape ('Typ 'Int w) -> T (s0 ++ indexShape ++ s1) t
  GatherND :: KnownTyp ('Typ 'Int w) => SShape containerShape -> SShape elementShape -> SShape indexShape
    -> T (containerShape ++ elementShape) t -> IndexTensor indexShape containerShape w -> T (indexShape ++ elementShape) t

  MatMul :: forall s m n o t. KnownNumeric t => SShape s -> Sat KnownNat n -> Sat KnownNat  o -> Sat KnownNat m -> T (s ++ '[n,o]) t -> T (s ++ [o,m]) t -> T (s ++ [n,m]) t
  Where :: T s TFBool  -> T s t -> T s t -> T s t
  If :: Scalar TFBool -> T s t -> T s t -> T s t
  Convolution :: KnownFloating t => Sat KnownNat bs -> Sat KnownNat inChannels -> Sat KnownNat outChannels -> SShape filterSpatialShape -> SShape s
            -> T (bs ': s ++ '[inChannels]) t -- input tensor (batched)
            -> T (filterSpatialShape ++ '[inChannels,outChannels]) t -- filters
            -> T (bs ': s ++ '[outChannels]) t
  Pool :: Length outSpatial ~ Length window =>
          Sat KnownNat bs -> SShape window -> PoolingType -> Sat KnownNat numChannels -> SShape outSpatial
            -> T (bs ': ZipWithMulShapes window outSpatial ++ '[numChannels]) t
            -> T (bs ': outSpatial ++ '[numChannels]) t
  Softmax :: Sat KnownNat bs -> Sat KnownNat n -> T '[bs,n] (Flt w) -> T '[bs,n] (Flt w)
    -- yes, softmax is shape-fixed: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/softmax

-- instance Show Unique where
--   show _ = "<Unique>"

-- deriving instance (Show (T s t))

type family ZipWithMulShapes (xs::Shape) (xy::Shape) :: Shape
type instance ZipWithMulShapes (x ': xs) (y ': ys) = x*y ': ZipWithMulShapes xs ys
type instance ZipWithMulShapes '[] _ = '[]
type instance ZipWithMulShapes _ '[] = '[]

satMul :: forall n m. Sat KnownNat n -> Sat KnownNat m -> Sat KnownNat (n*m)
satMul Sat Sat = Sat

satProd :: SShape s -> Sat KnownNat (Product s)
satProd Unit = natSat @1
satProd (x :* xs) = satMul x (satProd xs)

satAdd :: forall n m. Sat KnownNat n -> Sat KnownNat m -> Sat KnownNat (n+m)
satAdd Sat Sat = Sat

zipWithMulSShapes :: SShape xs -> SShape ys -> SShape (ZipWithMulShapes xs ys)
zipWithMulSShapes Unit _ = Unit
zipWithMulSShapes _ Unit = Unit
zipWithMulSShapes ((:*) x xs) ((:*) y ys) = (:*) (satMul x y) (zipWithMulSShapes xs ys)

data PoolingType = MaxPool | AvgPool deriving Show

type Tensor shape = T shape

data ReduceOp = Mean | Max | Min | Sum
data Axis1Op s1 t s2 u where
  ArgMax :: KnownNumeric t => Sat KnownNat n -> SShape s -> Axis1Op (n ': s) t s ('Typ 'Int b)
  OneHot :: KnownNumeric t => Sat KnownNat n -> SShape s -> Axis1Op s ('Typ 'Int b) (n ': s) t
  ReduceOp :: KnownNumeric t => Sat KnownNat n -> SShape s -> ReduceOp -> Axis1Op (n ': s) t s t

data Float1Op
  = ClipByValue Float Float
  | Tanh
  | Sin
  | Exp
  | Sigmoid
  | HardSigmoid
  | Relu
  | Floor
  | Round
  | Cos
  | Log
  | Asin
  | Acos
  | Sinh
  | Cosh
  | Asinh
  | Acosh
  | Atan
  | Atanh
  | Sqrt
  deriving Show
data Num1Op = Square | Negate | Abs | Sign
  deriving Show
data UnOp (s1 :: Shape) (t :: Typ) (s2 :: Shape) (u :: Typ) where
  Diag :: Sat KnownNat n -> UnOp '[n] t '[n,n] t
  StopGradient :: UnOp '[] t '[] t
  Cast :: UnOp '[] t '[] u
  Num1Op :: KnownNumeric t => Num1Op -> UnOp '[] t '[] t
  Float1Op :: Float1Op -> UnOp '[] (Flt w) '[] (Flt w)
  SliceOp :: forall m n s t proxy. proxy m -> Sat KnownNat n -> SShape s -> Integer -> Integer -> UnOp (n ': s) t (m ': s) t
  Axis1Op :: Axis1Op s1 t s2 u -> UnOp s1 t s2 u
             -- deriving Show

data Simple2Op t u where
  Divide :: Simple2Op (Flt w) (Flt w)
  Equal :: KnownTyp t => Simple2Op t TFBool
  Subtract :: KnownNumeric t => Simple2Op t t
  Multiply :: KnownNumeric t => Simple2Op t t
  Add :: KnownNumeric t => Simple2Op t t
  Minimum :: KnownNumeric t => Simple2Op t t
  Maximum :: KnownNumeric t => Simple2Op t t
  FloorMod :: KnownNumeric t => Simple2Op t t
  LessThan :: KnownNumeric t => Simple2Op t TFBool

-- deriving instance Show (Simple2Op t u)

data BinOp s1 t1 s2 t2 s3 t3 where
  Simple2Op :: Simple2Op t u -> BinOp '[] t '[] t '[] u
  SigmoidCrossEntropyWithLogits :: KnownFloating t => BinOp '[] t '[] t '[] t
  SoftmaxCrossEntropyWithLogits :: KnownFloating t => BinOp '[n] t '[n] t '[] t
  SparseSoftmaxCrossEntropyWithLogits :: BinOp '[] Int32 '[n] (Flt w) '[] (Flt w)

-- deriving instance Show (BinOp a b c d e f)

data Permutation (s :: [k]) (t :: [k]) where
  PermId :: Permutation s t
  PermSkip :: Permutation s t -> Permutation (n ': s) (n ': t)
  PermSwap :: Permutation (n ': m ': s) (m ': n ': s)
  PermTrans :: Permutation s t -> Permutation t u -> Permutation s u

deriving instance Show (Permutation s t)

class KnownTensors p where -- TODO: delete
  -- | traverse all the tensors contained in p.
  travTensor :: Applicative m => (forall s t. (KnownTyp t, KnownShape s) => String -> (T s t) -> m (T s t)) -> String -> p -> m p

instance (KnownTyp t, KnownShape shape) => KnownTensors (T shape t) where
  travTensor f = f

instance (All KnownPair ys) => KnownTensors (HHTV ys) where
  travTensor :: forall m. Applicative m => (forall s t'. (KnownTyp t', KnownShape s) => String -> T s t' -> m (T s t')) -> String -> HHTV ys -> m (HHTV ys)
  travTensor f s = ttr 0
    where ttr :: forall xs. All KnownPair xs => Int -> HHTV xs -> m (HHTV xs)
          ttr _ Unit = pure Unit
          ttr n (Uncurry x :* xs) = do
            x' <- f (s <> "_" <> show n) x
            xs' <- ttr (n + 1) xs
            return (Uncurry x' :* xs')

instance (KnownTyp t, All KnownShape ys) => KnownTensors (HTV t ys) where
  travTensor :: forall m. Applicative m => (forall s t'. (KnownTyp t', KnownShape s) => String -> T s t' -> m (T s t')) -> String -> (HTV t ys) -> m (HTV t ys)
  travTensor f s = ttr 0
    where ttr :: forall xs. All KnownShape xs => Int -> HTV t xs -> m (HTV t xs)
          ttr _ Unit = pure Unit
          ttr n (F x :* xs) = do
            x' <- f (s <> "_" <> show n) x
            xs' <- ttr (n + 1) xs
            return (F x' :* xs')

instance (KnownTensors p, KnownTensors q) => KnownTensors (p,q) where
  travTensor f s (x,y) = (,) <$> travTensor f (s<>"_fst") x <*> travTensor f (s<>"_snd") y

instance (KnownTensors p1, KnownTensors p2, KnownTensors p3) => KnownTensors (p1,p2,p3) where
  travTensor f s (x,y,z) = (,,) <$> travTensor f (s<>"_1") x <*> travTensor f (s<>"_2") y <*> travTensor f (s<>"_3") z

instance (KnownTensors p1, KnownTensors p2, KnownTensors p3, KnownTensors p4) => KnownTensors (p1,p2,p3,p4) where
  travTensor f s (x,y,z,w) = (,,,) <$> travTensor f (s<>"_1") x <*> travTensor f (s<>"_2") y <*> travTensor f (s<>"_3") z <*> travTensor f (s<>"_4") w

class KnownTensors p => ParamWithDefault p where
  defaultInitializer :: Gen p



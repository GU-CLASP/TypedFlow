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
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module TypedFlow.Types where

import Text.PrettyPrint.Compact hiding (All,Last,Product,Sum)
import GHC.TypeLits
import Data.Proxy
import Control.Monad.State
import Data.Kind (Constraint)
import TypedFlow.Memo
import qualified Data.Map as M
import Data.Unique
import qualified Data.Int as Hask
import Data.Type.Equality

data Sat (a :: k -> Constraint) (b::k) where
  Sat :: forall b a. a b => Sat a b

instance (Show (Sat a b)) where
  show _ = "Sat"

proxySat :: forall (b::k) (a :: k -> Constraint) proxy. a b => proxy b -> Sat a b
proxySat _ = Sat

natSat :: forall n. KnownNat n => Sat KnownNat n
natSat = Sat

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

(.+.) = appSList
infixr 5 .+.
infixr 5 *:
infixr 5 :*
(*:) :: forall x xs f. NP f xs -> f x -> NP f (xs ++ '[x])
xs *: x = appSList xs (x :* Unit)

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

knownAll :: forall constraint s k. SList' (Sat constraint) s -> (All constraint s => KnownLen s => k) -> k
knownAll Unit k = k
knownAll (Sat :* xs) k = knownAll xs $ k

allKnown :: forall constraint s proxy. All constraint s => SList' proxy s -> SList' (Sat constraint) s
allKnown Unit = Unit
allKnown (_ :* xs) = Sat :* allKnown xs

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

mapFMap :: forall g f xs. (forall x. f x -> f (Ap g x)) -> SList' f xs -> SList' f (Ap (FMap g) xs)
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

data Both f g x = Both (f x) (g x)

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

data Typ = Typ Kind NBits deriving (Eq,Ord)
type family TypKind (t :: Typ) where TypKind ('Typ k b)  = k
type family TypBits (t :: Typ) where TypBits ('Typ k b)  = b

type TFNumeric t = (NumericKind (TypKind t), KnownBits (TypBits t), t ~ 'Typ (TypKind t) (TypBits t))

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


knownBits :: SNBits t -> (KnownBits t => k) -> k
knownBits SB32 k = k
knownBits SB64 k = k

knownKind :: SKind t -> (KnownKind t => k) -> k
knownKind SFloat k = k
knownKind SInt k = k
knownKind SBool k = k

numericKnown :: forall t k. TFNumeric t => (KnownTyp t => k) -> k
numericKnown k = case kindVal @(TypKind t) of
  SFloat -> k
  SBool -> k
  SInt -> k

knownTyp :: STyp t -> (KnownTyp t => k) -> k
knownTyp (STyp k b Refl) r = knownKind k $ knownBits b r

knownFractional :: forall w k. KnownBits w => (Fractional (HaskType ('Typ 'Float w)) => k) -> k
knownFractional k = case bitsVal @w of
    SB32 -> k
    SB64 -> k

prettyKnown :: forall t. KnownTyp t => HaskType t -> DOC
prettyKnown = case kindVal @(TypKind t) of
  SInt -> case bitsVal @(TypBits t) of
    SB32 -> int . fromIntegral
    SB64 -> int . fromIntegral
  SBool -> bool
  SFloat -> case bitsVal @(TypBits t) of
    SB32 -> float
    SB64 -> double

class Pretty t where
  pretty :: t -> DOC

instance Pretty Bool where pretty = bool
instance Pretty Float where pretty = float
instance Pretty Double where pretty = double
instance Pretty Hask.Int64 where pretty = int . fromIntegral
instance Pretty Hask.Int32 where pretty = int . fromIntegral

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


data ParamInfo = forall s t. (KnownShape s, KnownTyp t) => 
  ParamInfo {paramName :: String
            ,paramShape :: [Integer]
            ,paramDType :: Typ
            ,paramVar   :: Tensor s t}
data GState = GState {nextVar :: Integer, -- ^ next free variable
                      genText :: DOC,
                      genParams :: [ParamInfo], -- ^ optimizable parameters
                      genPeeks :: [ParamInfo], -- ^ variables available after running the model (outputs)
                      genRegularizers :: [Scalar Float32], -- ^ accumulated regularizers
                      genTrainingPlaceholder :: Scalar TFBool, -- ^ flag which is true when training
                      genPureTable :: SSNMap2 Shape Typ T DOC,
                      -- ^ Table mapping pointers to their
                      -- interpretations, so that sharing in the data
                      -- structures can be exploited when generating
                      genAssignTable :: M.Map String DOC
                      -- ^ Table mapping expressions to variables, so
                      -- that lost sharing can be recovered
                      -- genPeeks :: [(String,UntypedExpression)]
                     }
data Gen a where
  GPVariable :: forall (shape :: Shape) t. (KnownTyp t,KnownShape shape) => Bool -> String -> T shape t -> Gen (T shape t) 
  GPPlaceholder :: forall s t. SShape s -> STyp t -> String -> Gen (T s t)
  GPModify :: (KnownShape s,KnownTyp t) => T s t -> T s t -> Gen (T s t)
  GPReturn :: a -> Gen a
  GPState :: (GState -> (a,GState)) -> Gen a
  GPBind :: Gen a -> (a -> Gen b) -> Gen b

instance MonadState GState Gen where
  state = GPState

instance Monad Gen where
  (>>=) = GPBind
  return = GPReturn

instance Applicative Gen where
  (<*>) = ap
  pure = return

instance Functor Gen where
  fmap f = (pure f <*>)


-- | Name of a placeholder of a given shape and type.
data HolderName (st :: (Shape,Typ)) = HolderName String


newVar :: Gen String
newVar = do
  n <- newId
  return ("var" <> show n)

-- newId :: Gen Integer
newId :: MonadState GState m => m Integer
newId = do
  n <- gets nextVar
  modify $ \GState{..} -> GState {nextVar=nextVar+1,..}
  return n

--------------------------
-- Tensors

type UntypedExpression = DOC

instance Show DOC where
  show = renderWith (Options 92 (const id))

-- | An indexing tensor in the format expected by GatherND
type IndexTensor indexShape containerShape w = T (indexShape ++ '[Length containerShape]) ('Typ 'Int w)

-- | Description of a random distribution
data Distribution (s :: Shape) (t :: Typ) where
  -- | Each element is from a truncated normal distribution with given standard dev.
  TruncatedNormalD :: Float ->  Distribution s ('Typ 'Float w)
  -- | Each element is from a uniform distribution with given bounds (low, high)
  UniformD :: Float -> Float -> Distribution s ('Typ 'Float w)
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
  Gather :: KnownTyp ('Typ 'Int w) => SShape indexShape -> SShape s0 -> Sat KnownNat m -> SShape s1
    -> T (s0 ++ (m ': s1)) t -> T indexShape ('Typ 'Int w) -> T (s0 ++ indexShape ++ s1) t
  GatherND :: KnownTyp ('Typ 'Int w) => SShape containerShape -> SShape elementShape -> SShape indexShape
    -> T (containerShape ++ elementShape) t -> IndexTensor indexShape containerShape w -> T (indexShape ++ elementShape) t

  MatMul :: forall s m n o t. TFNumeric t => SShape s -> Sat KnownNat n -> Sat KnownNat  o -> Sat KnownNat m -> T (s ++ '[n,o]) t -> T (s ++ [o,m]) t -> T (s ++ [n,m]) t
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

instance (All KnownPair ys) => KnownTensors (HHTV ys) where
  travTensor :: forall m. Monad m => (forall s t'. (KnownTyp t', KnownShape s) => String -> T s t' -> m (T s t')) -> String -> (HHTV ys) -> m (HHTV ys)
  travTensor f s = ttr 0
    where ttr :: forall xs. All KnownPair xs => Int -> HHTV xs -> m (HHTV xs)
          ttr _ Unit = return Unit
          ttr n (Uncurry x :* xs) = do
            x' <- f (s <> "_" <> show n) x
            xs' <- ttr (n + 1) xs
            return (Uncurry x' :* xs')

instance (KnownTyp t, All KnownShape ys) => KnownTensors (HTV t ys) where
  travTensor :: forall m. Monad m => (forall s t'. (KnownTyp t', KnownShape s) => String -> T s t' -> m (T s t')) -> String -> (HTV t ys) -> m (HTV t ys)
  travTensor f s = ttr 0
    where ttr :: forall xs. All KnownShape xs => Int -> HTV t xs -> m (HTV t xs)
          ttr _ Unit = return Unit
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

{-# LANGUAGE InstanceSigs #-}
{-|
Module      : TypedFlow.Abstract
Description : Abstract Tensor representations
Copyright   : (c) Jean-Philippe Bernardy, 2018
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental

This module provides operations on the abstract representation of
tensor operations. It is not normally imported directly by users.
-}

{-# LANGUAGE ApplicativeDo #-}
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
{-# LANGUAGE CPP #-}
#if __GLASGOW_HASKELL__ >= 806
{-# LANGUAGE NoStarIsType #-}
#endif

module TypedFlow.Broadcast (
  -- * broadcasting
  doBroadcast,mapPlaceHolders, ConsSh, unopInputShape,
  -- * helpers which are also useful elsewhere
  -- ** reshapes
  reshape, reshapeAuto, reshapeFrom, reshapeTo, inflate2, flatten2,
  permToFun, 
  -- ** transpositions
  transpose01, transposeN, transposeN', transposeN01,
  -- ** others
  concatT', range,
  ) where

import Control.Monad.State
-- import Data.Kind (Type,)
import Data.Proxy
import Data.Type.Equality
import GHC.TypeLits
import Prelude hiding (RealFrac(..))
import System.IO.Unsafe
import TypedFlow.Memo2 hiding (Comp)
import TypedFlow.Types (T(..), type (∘)(..))
import TypedFlow.Types hiding (T)
import TypedFlow.Types.Proofs


data GS = GS { gsUnique :: Unique }


type G = StateT GS IO 

runG :: Unique -> G x -> x
runG u m = fst (unsafePerformIO  (runStateT m GS { gsUnique = u }))

doBroadcast :: All KnownPlaceholder ps => Placeholders ps -> Placeholders ps
doBroadcast phs = runG 0 $ do
  F3m' bc <- mkBroadcastFn
  let broadcast :: forall n s t. BroadcastFn n s t
      broadcast = unwrapBCFn bc
  F2m' gBC' <- mkGenerateBC broadcast
  let generateBC :: forall s t. GenBCFn s t
      generateBC = unwrapGBCFn gBC'
      generateBCMany :: forall ps. All KnownPlaceholder ps => Placeholders ps -> G (Placeholders ps)
      generateBCMany = \case
        Unit -> return Unit
        (PHT x :* xs) -> do
          x' <- generateBC x
          xs' <- generateBCMany xs
          return (PHT x' :* xs')
  generateBCMany phs


getUnique :: G Unique
getUnique = do
  u <- gets ((1+) . gsUnique)
  modify $ \GS {} -> GS {gsUnique = u,..}
  return u

generateBC' :: (forall n s t proxy. KnownTyp t => KnownShape s => KnownNat n => Unique -> Bool -> proxy n -> T s t -> G (T (n : s) t))
            -> (forall s' t'. KnownTyp t' => SShape s' -> T s' t' -> G (T s' t'))
            -> forall s t. KnownTyp t
            => SShape s
            -> T s t
            -> G (T s t)
generateBC' broadcast rec (n@Sat :* sR) (Zip3T _ _s1 _s2 _s3 f x y z) = knownSShape sR ?> do
  u <- getUnique
  -- ATTN: it is critical not to do recursive calls to x,y,z here. Doing so would create new nodes, loosing sharing, and creating problems down the line.
  a' <- rec sR (f (Unbroadcast n u x) (Unbroadcast n u y) (Unbroadcast n u z))
  broadcast u False n a'
generateBC' broadcast rec (n@Sat :* sR) (ZipT _ _s1 _s2 f x y) = knownSShape sR ?> do
  u <- getUnique
  a' <- rec sR (f (Unbroadcast n u x) (Unbroadcast n u y))
  broadcast u False n a'
generateBC' broadcast rec (n@Sat :* sR) (MapT _ _s' f x) = knownSShape sR ?> do
  u <- getUnique
  a' <- rec sR (f (Unbroadcast n u x))
  broadcast u False n a'
generateBC' broadcast rec (n@Sat :* sR) (BroadcastT maybeUnique varyNoise _ _s' a) = knownSShape sR ?> do
  u <- case maybeUnique of
          Nothing -> getUnique
          Just u' -> return u'
  a' <- rec sR a
  broadcast u varyNoise n a'
generateBC' _ _ _ (n@T {}) = return n
generateBC' _ _ _ (n@Noise {}) = return n
generateBC' _ rec _ (BinOp op s0 s1 t1 s2 t2 x y) = knownTyp t1 $ knownTyp t2 $ BinOp op s0 s1 t1 s2 t2 <$> (rec (s0 .+. s1) x) <*> (rec (s0 .+. s2) y)
generateBC' _ rec _ (UnOp op s0 x) = UnOp op s0 <$> rec (s0 .+. unopInputShape op) x
generateBC' _ rec sR (Unbroadcast p u' x) = Unbroadcast p u' <$> rec (p :* sR) x
generateBC' _ rec _ (DirectBroadcast s0 s1 s2 s3 x) = DirectBroadcast s0 s1 s2 s3 <$> (rec (s0 .+. s2) x)
generateBC' _ rec _ (ReshapeFrom s0 x) = reshapeFrom s0 <$> rec s0 x
generateBC' _ rec _ (Transpose s0 t x) = Transpose s0 t <$> (rec s0 x)
generateBC' _ rec _ (Concat s0 s1 xs) = Concat s0 s1 <$> hTraverse (\(Catable m x) -> Catable m <$> (rec (s0 .+. m :* s1) x)) xs
generateBC' _ rec _ (Gather is s0 m s1 x ix) = Gather is s0 m s1 <$> (rec (s0 .+. m :* s1) x) <*> rec (s0 .+. is) ix
generateBC' _ rec _ (GatherND cs es is x ix) = GatherND cs es is <$> (rec (cs .+. es) x) <*> (rec (is *: sListLenAsNat cs) ix)
generateBC' _ rec _ (MatMul s0 a b c x y) = MatMul s0 a b c <$> (rec (s0 .+. a :* b :* Unit) x) <*> (rec (s0 .+. b :* c :* Unit) y)
generateBC' _ rec sR (Where cond x y) = Where <$> rec sR cond <*> rec sR x <*> rec sR y
generateBC' _ rec sR (If cond x y) = If <$> rec Unit cond <*> rec sR x <*> rec sR y
generateBC' _ rec _ (Convolution bs@Sat inChans outChans filterShape s0 x filters) = Convolution bs inChans outChans filterShape s0 <$> (rec (bs :* (s0 *: inChans)) x) <*> (rec (filterShape .+. inChans :* outChans :* Unit) filters)
generateBC' _ rec _ (Pool bs@Sat window pt numChans outSpatial x) = Pool bs window pt numChans outSpatial <$> rec (bs :* (zipWithMulSShapes window outSpatial *: numChans)) x
generateBC' _ rec _ (Softmax bs n x) = Softmax bs n <$> (rec (bs :* n :* Unit) x)
generateBC' _ _ _ _ = error "generateBC': unhandled case"



(<&&>) :: Applicative f => f Bool -> f Bool -> f Bool
x <&&> y = (&&) <$> x <*> y

-- | True if the argument does not contain an expression which should be broadcast.
protoFinished :: Unique -> Bool -> (forall s' t'. Unique -> Bool -> T s' t' -> G Bool) -> T s t -> G Bool
protoFinished u varyNoise rec0 =
  let rec :: forall s t. T s t -> G Bool
      rec = rec0 u varyNoise
  in \case
    BroadcastT _ _ _ _s a -> rec a
    MapT _ s f x -> rec x <&&> rec (f (T (Variable (Ref 0 s typeSTyp))))
    ZipT _ s0 s1 f x y -> rec x <&&> rec y <&&> rec (f (T (Variable (Ref 0 s0 typeSTyp))) (T (Variable (Ref 0 s1 typeSTyp))))  
    Zip3T _ s0 s1 s2 f x y z -> rec x <&&> rec y <&&> rec z <&&> rec (f (T (Variable (Ref 0 s0 typeSTyp))) (T (Variable (Ref 0 s1 typeSTyp))) (T (Variable (Ref 0 s2 typeSTyp))))  
    Softmax _ _ x -> rec x
    DirectBroadcast _ _ _ _ x -> rec x
    GatherND _ _ _ x y -> rec x <&&> rec y
    Noise _ _ _ _ -> return (not varyNoise)
    Where cond x y -> rec cond <&&> rec x <&&> rec y
    If cond x y ->  rec cond <&&> rec x <&&> rec y
    T _ -> return True
    Unbroadcast _p u' _x -> return (u /= u')
    UnOp _op _ x -> rec x
    MatMul _ _ _ _ x y -> rec x <&&> rec y
    BinOp _op _ _ _ _ _ x y -> rec x <&&> rec y
    Gather _is _s0 _m _s1 x ix -> rec x <&&> rec ix
    Transpose _ _t x -> rec x
    ReshapeFrom _s x -> rec x
    Concat _s0  _s1 xs -> (and . htoList) <$> hTraverse (\(Catable _ x) -> K <$> rec x) xs
    Convolution _bs _inChans _outChans _filterShape _s x filters -> rec x <&&> rec filters
    Pool _ _ _ _ _ x  -> rec x
    -- _ -> error "protoFinished: unhandled case"

data K02 t x y = K02 {fromK02 :: t}


mkFinished :: G (F2m G (Sig02 Bool (Sig02 Unique T)) (K02 Bool) ) -- forall s' t'. Unique -> Bool -> T s' t' -> G (F2m _)
mkFinished = memo2 (ordMap @Bool `containing02` (ordMap @Unique `containing02` snMap2 @T)) $
             \rec (Ex02 u (Ex02 v x)) -> K02 <$> protoFinished v u (unwrapFin rec) x

unwrapFin :: ((Sig02 Bool (Sig02 Unique T)) s t -> G (K02 Bool s t)) -> Unique -> Bool -> T s t -> G Bool
unwrapFin f u v x = fromK02 <$> f (Ex02 v (Ex02 u x))

data KT s t where
  KT ::  STyp t -> SShape s -> KT s t

type GenBCFn s t = (KnownTyp t, KnownShape s) => T s t -> G (T s t)


unwrapGBCFn :: forall s t. (T s t -> KT s t -> G (T s t)) -> GenBCFn  s t
unwrapGBCFn f x' = f x' (KT typeSTyp typeSShape)

-- isBroadcastT :: T s t -> Bool
-- isBroadcastT (BroadcastT {}) = True
-- isBroadcastT _ = False

mkGenerateBC :: (forall n s t. BroadcastFn n s t) -> G (F2m' G T KT T)
mkGenerateBC broadcast = memo2' (snMap2 @T) $
                         \rec x (KT t s) -> knownTyp t $ do
                           r <- generateBC' broadcast (\sh' x' -> rec x' (KT typeSTyp sh')) s x
                           -- when (isBroadcastT r) $ liftIO $ putStrLn "YIKES!"
                           return r

newtype BC'd (n :: Nat) (s :: Shape) (t :: Typ) = BC'd {fromBC'd :: (T (n : s) t)}

data KTn n s t where
  KTn ::  STyp t -> SShape s -> KTn n s t

type BroadcastFn n s t = forall proxy. (KnownNat n, KnownShape s, KnownTyp t) => Unique -> Bool -> proxy n -> T s t -> G (T (n : s) t)

unwrapBCFn :: ((Sig03 Unique (Sig03 Bool (Sig12 (Sat KnownNat) T))) n s t -> KTn n s t -> G (BC'd n s t)) -> BroadcastFn n s t
unwrapBCFn f u v _n x' = fromBC'd <$> f (Ex03 u (Ex03 v (Ex12 natSat x'))) (KTn typeSTyp typeSShape)

mkBroadcastFn :: G (F3m' G (Sig03 Unique (Sig03 Bool (Sig12 (Sat KnownNat) T))) KTn BC'd)
mkBroadcastFn = do
  F2m fin <- mkFinished
  memo3' (ordMap @Unique `containing03` (ordMap @Bool `containing03` (verifMap1 @(Sat KnownNat) `containing12` snMap2 @T))) $
    \rec (Ex03 u (Ex03 v (Ex12 n x))) (KTn st sh) ->
      BC'd <$> protoBroadcast u v n
                  (\sh' x' -> fromBC'd <$> rec (Ex03 u (Ex03 v (Ex12 n x'))) (KTn typeSTyp sh'))
                  (unwrapFin fin u v) st sh x


class ConsSh (x :: Nat) (p :: (Symbol,Shape,Typ))
instance Fun (ConsSh x) where type Ap (ConsSh x) p = '(Frst3 p,x ': Scnd3 p,Thrd3 p)

-- -- | Turns a tensor of indices in a container into a tensor of indices
-- -- in a container of higher rank. The added indexed dimension
-- -- corresponds to the first dimension of the index.
-- broadcastIndex :: forall n containerShape indexShape w.
--   KnownBits w => Sat KnownNat n ->
--   SShape containerShape ->
--   SShape indexShape ->
--   IndexTensor (n ': indexShape) containerShape w ->
--   IndexTensor (n ': indexShape) (n ': containerShape) w
-- broadcastIndex n cs = broadcastIndex' n (sListLenAsNat cs)

broadcastIndex' :: forall n containerRank indexShape w.
  KnownBits w => Sat KnownNat n ->
  Sat KnownNat containerRank ->
  SShape indexShape ->
  T (n ': indexShape ++ '[containerRank])  ('Typ 'Int w) ->
  T (n ': indexShape ++ '[1 + containerRank]) ('Typ 'Int w)
broadcastIndex' n@(Comp Dict) cr is ix = concatT' ((:*) n is) (natSat @1) cr Unit nIndex ix
  where nIndex :: T (n ': indexShape ++ '[1]) ('Typ 'Int w)
        nIndex = DirectBroadcast Unit Unit ((:*) n Unit) (is .+. (:*) (natSat @1) Unit) range

-- directBroadcast0 :: forall n s t. KnownShape s => KnownNat n => T s t -> T (n:s) t
-- directBroadcast0 = appRUnit @s #> DirectBroadcast Unit ((:*) (natSat @n) Unit) (typeSShape @s) Unit

-- broadcastIndexMany :: forall n containerShape indexShape w.
--   KnownBits w =>
--   Sat KnownNat n ->
--   SShape containerShape ->
--   SShape indexShape ->
--   IndexTensor indexShape '[n] w ->
--   IndexTensor (containerShape ++ indexShape) (containerShape ++ '[n]) w
-- broadcastIndexMany _ Unit _ x = x
-- broadcastIndexMany n ((:*) m@Sat cs) is x =
--   knownSShape (cs .+. (*:) is (sListLenAsNat (cs *: n))) ?>
--   -- (m : cs ++ is ++  '[(Length (m : cs ++ [n]))])
--   (broadcastIndex m ((*:) cs n) (cs .+. is) $
--   -- (m : (cs ++ is ++  '[Length (cs ++ [n])]))
--   (appAssocS cs is ((:*) (sListLenAsNat (cs *: n)) Unit) #>
--   -- (m : cs ++ is ++ '[Length (cs ++ [n])])
--   directBroadcast0 $
--   -- (cs ++ is ++  '[Length (cs ++ [n])])
--   broadcastIndexMany n cs is x))
--   -- is

--  Product (filterSpatialShape ++ '[inChannels, outChannels * n])
-- Product ((filterSpatialShape ++ '[inChannels, outChannels]) ++ '[n])

axisOpInputShape :: Axis1Op s1 t s2 u -> SShape s1
axisOpInputShape o = case o of
  ArgMax n -> HSingle n
  OneHot _n -> Unit
  ReduceOp n _ -> HSingle n
  ReverseT n -> HSingle n
  SliceOp _ n _ _ -> HSingle n
  AccessOp n _ -> HSingle n

unopInputShape :: UnOp s t s' t' -> SShape s
unopInputShape (Diag n) = n :* Unit
unopInputShape Cast = Unit
unopInputShape (Axis1Op s o) = axisOpInputShape o .+. s
unopInputShape StopGradient = Unit
unopInputShape (Num1Op _) = Unit
unopInputShape (Float1Op _) = Unit
unopInputShape (ExpM n) = n :* n :* Unit
unopInputShape (ZeroTriangle n _ _) = n :* n :* Unit
unopInputShape Conjugate = Unit
unopInputShape RealPart = Unit

protoBroadcast :: forall n s t.
  Unique -- unique identifier marking the variable tensor which will be marking inputs (not to broadcast).
  -> Bool -- how to expand the noise? (If True use different noise for all indices)
  -> Sat KnownNat n -- added dimension's size
  -> (forall s' t'. KnownTyp t' => SShape s' -> T s' t' -> G (T (n ': s') t')) -- recursive case
  -> (forall s' t'. T s' t' -> G Bool) -- test if we're done
  -> STyp t -- representation of the type
  -> SShape s -- representation of the shape
  -> T s t -- tensor (expression) to broadcast
  -> G (T (n ': s) t) -- return broadcated expression (on 1st position)
protoBroadcast u varyNoise n@(Comp Dict) rec finished ty s tensor = do
  isFinished <- finished tensor
  case isFinished of
    True -> simpleBC
    False -> knownTyp ty $ case tensor of
      BroadcastT {} -> error "BroadcastT case remaining, this should have been dealt with by generateBC"
      MapT {} -> error "MapT case remaining, this should have been dealt with by generateBC"
      ZipT {} -> error "ZipT case remaining, this should have been dealt with by generateBC"
      Zip3T {} -> error "Zip3T case remaining, this should have been dealt with by generateBC"
      Softmax bs@Sat m@Sat x -> prodAssocS n bs m #> do
        x' <- rec (typeSShape) x
        return (reshapeAuto (Softmax (satMul n bs) m (reshapeAuto x')))
      DirectBroadcast s0 s1 s2 s3 x -> do
        x' <- (rec (s0 .+. s2) x)
        return (DirectBroadcast (n :* s0) s1 s2 s3 x')
      GatherND cs es is x ix -> do
        xFinished <- finished x
        case xFinished of
          True -> GatherND cs es (n :* is) x <$> (rec (is *: sListLenAsNat cs) ix)
          False -> do
              ix' <- rec (is *: sListLenAsNat cs) ix
              x' <- (rec (cs .+. es) x)
              return (GatherND (n :* cs) es (n :* is) x' (broadcastIndex' n (sListLenAsNat cs) is ix'))
      Noise v s0 s1 x -> if varyNoise then return (Noise v (n :* s0) s1 x) else simpleBC
      -- When varying noise, then we extend the shape of the noise (so
      -- more stuff is sampled), otherwise we copy the noise using simple
      -- broadcasting
      Pool bs@Sat window pt numChans outSpatial x ->
        (knownSShape (zipWithMulSShapes window outSpatial *: numChans) ?>
         (prodAssocS n bs (productS (zipWithMulSShapes window outSpatial *: numChans)) #>
         (prodAssocS n bs (productS (outSpatial *: numChans)) #> do
             x' <- (rec typeSShape x)
             return $ (reshapeFrom (satMul n bs :* outSpatial *: numChans) $
                       Pool (satMul n bs) window pt numChans outSpatial (reshapeAuto x')))))
      Where cond x y -> Where <$> (rec s cond) <*> (rec s x) <*> (rec s y)
      If cond x y -> do
        condFinished <- finished cond
        case condFinished of
          True -> If cond <$> (rec s x) <*> (rec s y)
          False ->  error "broadcast on 'if' condition not implemented"
      T _ -> error "panic: broadcast constant should be finished!"
      Unbroadcast p@Sat u' x
        | u == u' -> return $ case testEq p n of
            Nothing -> UnOp (error "panic.unbroadcast.unit") Unit x
            Just Refl -> x
        | otherwise -> knownSShape s ?> do
            x' <- (rec (p :* s) x)
            return (Unbroadcast p u' (transpose01 x'))
          -- An uncomplete broadcast (in another dimension).
      MatMul s0 a@Sat b@Sat c@Sat x y -> do
        yFinished <- finished y
        case (s0,yFinished) of
           (Unit,True) -> do
             -- this optimisation is absolutely critical to implement dense
             -- layers efficiently (at least with TF 1.3). (about 10x performance increase)
             x' <- (rec (a :* b :* Unit) x)
             return $ inflate2 (MatMul Unit (satMul n a) b c (flatten2 x') y)
           _ -> MatMul (n :* s0) a b c  <$> (rec (s0 .+. a :* b :* Unit) x) <*> (rec (s0 .+. b :* c :* Unit) y)
      BinOp op s0 s1 t1 s2 t2 x y -> knownTyp t1 $ knownTyp t2 $ do
        BinOp op (n :* s0) s1 t1 s2 t2 <$> (rec (s0 .+. s1) x) <*> (rec (s0 .+. s2) y)
      UnOp op s0 x -> UnOp op (n :* s0) <$> (rec (s0 .+. unopInputShape op) x)
      Gather is s0 m s1 x ix -> do
        xFinished <- finished x
        case (s0,xFinished) of -- this optimisation is important to get efficient embeddings (???)
          (Unit,True) -> Gather (n :* is) Unit m s1 x <$> (rec is ix)
          _ -> Gather is (n :* s0) m s1 <$> (rec (s0 .+. m :* s1) x) <*> (rec (s0 .+. is) ix)
      Transpose s0 t x -> Transpose (n :* s0) (PermSkip t) <$> (rec s0 x)
      ReshapeFrom s0 x -> reshapeFrom (n :* s0) <$> (rec s0 x)
      Concat s0 s1 xs -> do
        Concat (n :* s0) s1 <$> hTraverse (\(Catable m x) -> Catable m <$> (rec (s0 .+. m :* s1) x)) xs
      Convolution bs@(Sat) inChans outChans filterShape s0 x filters -> do
        filtersFinished <- finished filters
        xFinished <- finished x
        case (filtersFinished,xFinished) of
          (True,_) ->
            prodAssocS n bs (productS (s0 *: inChans))  #>
            prodAssocS n bs (productS (s0 *: outChans)) #>
            knownSShape (s0 *: inChans)                 ?> do
              x' <- (rec typeSShape x)
              return $ reshapeFrom (satMul n bs :* s0 *: outChans) 
                      (Convolution (satMul n bs) inChans outChans filterShape s0 (reshapeAuto x') filters)
          (_,True) ->
            knownSShape (filterShape .+. inChans :* outChans :* Unit) ?>
            knownSShape (bs :* s0 .+. outChans :* Unit) ?> do
              filters' <- rec typeSShape filters
              return $ transposeN' $
                reshapeProven (ANat bs !:* AShape s0 *:! (ANat outChans :*: ANat n))
                              ((ANat bs !:* AShape s0 *:! ANat outChans) *:! ANat n) $
                Convolution bs inChans (outChans `satMul` n) filterShape s0 x $
                reshapeProven ((AShape filterShape :++: (ANat inChans !:* Single (ANat outChans))) *:! ANat n)
                              (AShape filterShape :++: ANat inChans !:* Single (ANat outChans :*: ANat n)) $
                transposeN $ filters'

          _ -> error "broadcast on both convolution filter and data not implemented"
      _ -> error "protoBroadcast: unhandled case" 
  where simpleBC :: G (T (n ': s) t)
        simpleBC = appRUnit @s #>
                   return (DirectBroadcast Unit (n :* Unit) s Unit tensor)

inversePerm :: Permutation a b -> Permutation b a
inversePerm PermId = PermId
inversePerm (PermSkip x) = PermSkip (inversePerm x)
inversePerm PermSwap = PermSwap
inversePerm (PermTrans x y) = PermTrans (inversePerm y) (inversePerm x)

permToFun :: Permutation s t -> Integer -> Integer
permToFun = \case
  PermId -> \x -> x
  PermTrans a b -> permToFun b . permToFun a
  PermSwap -> \case
    0 -> 1
    1 -> 0
    x -> x
  PermSkip p -> \case
    0 -> 0
    x -> permToFun p (x-1) + 1

reshapeAuto :: forall s s0 t. KnownShape s0 => Product s ~ Product s0 => T s0 t -> T s t
reshapeAuto = reshapeFrom typeSShape

reshapeProven :: forall s s0 t n. ShapeX s0 n -> ShapeX s n -> T s0 t -> T s t
reshapeProven s1 s2 = case decideProductEq s1 s2 of
                        Refl -> knownSShape (exprSShape s1) ?> reshapeAuto

reshapeTo :: forall s s0 t proxy. KnownShape s0=> Product s ~ Product s0 => proxy s -> T s0 t -> T s t
reshapeTo _ = reshapeAuto

reshapeFrom :: forall s s0 t. Product s ~ Product s0 => SShape s0 -> T s0 t -> T s t
reshapeFrom _ (ReshapeFrom s1 x) = ReshapeFrom s1 x -- avoid reshaping over and over
reshapeFrom s0 x = ReshapeFrom s0 x


type BatchedPlaceholders n ps = Placeholders (BPH n ps)
type BPH n ps = (Ap (FMap (ConsSh n)) ps)

-- | Batch the model (adding one dimension).
mapPlaceHolders :: forall batchSize shapesAndTypes resShapesAndTypes.
    (KnownNat batchSize, KnownLen shapesAndTypes, KnownLen resShapesAndTypes, All KnownPlaceholder shapesAndTypes, All KnownPlaceholder resShapesAndTypes)
  => Unique
  -> Bool
  -> (Placeholders shapesAndTypes -> Placeholders resShapesAndTypes)
  -> BatchedPlaceholders batchSize shapesAndTypes -> (BatchedPlaceholders batchSize resShapesAndTypes)
mapPlaceHolders u varyNoise f xs = broadcastPlacehoders @batchSize typeSList (f (unbroadcastPlacehoders @batchSize typeSList xs))  where
    unbroadcastPlacehoders :: forall n r. KnownNat n => SList r -> BatchedPlaceholders n r -> Placeholders r
    unbroadcastPlacehoders Unit Unit = Unit
    unbroadcastPlacehoders (_ :* ss) (PHT x :* xs') = PHT (Unbroadcast batchSize u x) :* unbroadcastPlacehoders @n ss xs'
      where batchSize = natSat @n

    broadcastPlacehoders :: forall n r. All KnownPlaceholder r => KnownNat n => SList r -> Placeholders r -> (BatchedPlaceholders n r)
    broadcastPlacehoders Unit Unit = Unit
    broadcastPlacehoders (_ :* ss) (PHT x :* xs) =
      let x' = BroadcastT (Just u) varyNoise (natSat @n) typeSShape x
          xs' = broadcastPlacehoders @n ss xs
      in (PHT x' :* xs') 

----------------------------------------------------------------
-- Here start helper functions

permN :: SList s -> Permutation (n ': s) (s ++ '[n])
permN Unit = PermId
permN ((:*) _n s) = PermSwap `PermTrans` PermSkip (permN s)

permN01 :: SList s -> Proxy m -> Proxy n -> Permutation (s ++ [m,n]) (s ++ [n,m])
permN01 Unit _ _ = PermSwap
permN01 ((:*) _n s) m n = PermSkip (permN01 s m n)


-- | Transposition. See the type for the permutation of dimensions.
transposeN :: ∀ s n t. KnownNat n => KnownShape s => T (n ': s) t -> T (s ++ '[n]) t
transposeN  = doTranspose typeSShape (permN (typeSList @s))

-- | Transposition. See the type for the permutation of dimensions.
transposeN' :: ∀ s n t. KnownNat n => KnownShape s => T (s ++ '[n]) t -> T (n ': s) t
transposeN' = doTranspose (typeSShape @s *: (natSat @n)) (inversePerm (permN (typeSList @s)))


-- | Transposition. See the type for the permutation of dimensions.
transposeN01 :: ∀ s m n t. KnownNat n => KnownNat m => KnownShape s => T (s ++ [m,n]) t -> T (s ++ [n,m]) t
transposeN01 = doTranspose (typeSShape @s .+. typeSShape @'[m,n]) (permN01 (typeSList @s) (Proxy @m) (Proxy @n))


-- | Transposition. See the type for the permutation of dimensions.
transpose01 :: ∀ s m n t. KnownNat n => KnownNat m => KnownShape s => T (m ': n ': s) t -> T (n ': m ': s) t
transpose01 = doTranspose typeSShape PermSwap

doTranspose :: SShape s0 -> Permutation s0 s -> T s0 t -> T s t
doTranspose _  p (Transpose sh' q x) = doTranspose sh' (PermTrans q p) x
doTranspose sh p x = Transpose sh p x


-- | Concatenate tensors with explicit shapes.
concatT' :: ∀ s0 d1 d2 s1 t. KnownTyp t =>
    SShape s0 -> Sat KnownNat d1 -> Sat KnownNat d2 -> SShape s1 -> T (s0 ++ (d1 ': s1)) t -> T (s0 ++ (d2 ': s1)) t -> T (s0 ++ ((d1+d2) ': s1)) t
concatT' s0 d1@(Comp Dict) d2@(Comp Dict) s1 x y = Concat s0 s1 (Catable d1 x :* Catable d2 y :* Unit)


-- | Reshape a tensor so that the first dimension is expanded into two.
inflate2 :: ∀ m n s t. KnownTyp t => (KnownNat m, KnownNat n, KnownShape s) => Tensor (m*n ': s) t -> Tensor (m ': n ': s) t
inflate2 = prodAssoc @m @n @(Product s) #> reshape


-- | Reshape a tensor so that the first two dimensions are collapsed
flatten2 :: ∀ m n s t. KnownTyp t => (KnownNat m, KnownNat n, KnownShape s) => Tensor (m ': n ': s) t -> Tensor (m*n ': s) t
flatten2 = prodAssoc @m @n @(Product s) #> reshape


reshape :: ∀ s2 s1 t. KnownShape s1 => KnownShape s2 => Product s1 ~ Product s2 => Tensor s1 t -> Tensor s2 t
reshape = reshapeAuto

-- | range[i] = i
range :: forall n w. KnownNat n => KnownBits w => T '[n] ('Typ 'Int w)
range = T (Range (natSat @n))

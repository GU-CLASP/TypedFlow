{-|
Module      : TypedFlow.Layers.RNN.Base
Description : RNN cells, layers and combinators.
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental
-}

{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE TypeInType #-}
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
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE PatternSynonyms #-}

module TypedFlow.Layers.RNN.Base (
  -- * Types
  RnnCell, RnnLayer,
  -- * Monad-like interface
  Component(..), bindC, returnC, liftC,
  -- * Combinators
  stackRnnCells, (.-.),
  stackRnnLayers, (.--.),
  bothRnnCells, (.|.),
  bothRnnLayers,(.++.),
  withBypass, withFeedback,
  onStates,
  timeDistribute, timeDistribute',
  -- * RNN unfolding functions
  rnn,
  rnnBackward,
  rnnWithCull,
  -- rnnBackwardsWithCull,
  )

where
import Prelude hiding (tanh,Num(..),Floating(..),floor)
import GHC.TypeLits
import TypedFlow.TF
import TypedFlow.Types
-- import Data.Type.Equality
-- import Data.Kind (Type,Constraint)

newtype Component t states a = C {runC :: HTV (Flt t) states -> Gen (HTV (Flt t) states , a)}

instance Functor (Component t states) where
  fmap = mapC

mapC :: (a -> b) -> Component t s a -> Component t s b
mapC f c = C $ \s ->  do
  (s',x) <- runC c s
  return (s', f x)

returnC :: a -> Component t '[] a
returnC x = C $ \Unit -> return (Unit,x)

bindC :: forall t s0 s1 a b. KnownLen s1
  => Component t s0 a -> (a -> Component t s1 b) -> Component t (s1++s0) b
bindC f g = C $ \(hsplit @s1 -> (s1,s0)) -> do
  (s0',x) <- runC f s0
  (s1',y) <- runC (g x) s1
  return (happ s1' s0',y)

liftC :: Gen a -> Component t '[] a
liftC f = C $ \Unit -> do
  x <-  f
  return (Unit,x)

-- | A cell in an rnn. @state@ is the state propagated through time.
type RnnCell t states input output = input -> Component t states output

-- | A layer in an rnn. @n@ is the length of the time sequence. @state@ is the state propagated through time.
type RnnLayer n b state input output = RnnCell b state (V n input) (V n output) 

----------------------
-- Lifting functions

-- | Convert a pure function (feed-forward layer) to an RNN cell by
-- ignoring the RNN state.
timeDistribute :: (a -> b) -> RnnCell t '[] a b
timeDistribute pureLayer = timeDistribute' (return . pureLayer)

-- | Convert a stateless generator into an RNN cell by ignoring the
-- RNN state.
timeDistribute' :: (a -> Gen b) -> RnnCell t '[] a b
timeDistribute' stateLess a = liftC (stateLess a)

--------------------------------------
-- Combinators


-- | Compose two rnn layers. This is useful for example to combine
-- forward and backward layers.
(.--.),stackRnnLayers :: forall s1 s2 a b c n bits. KnownLen s2
  => RnnLayer n bits s1 a b -> RnnLayer n bits s2 b c -> RnnLayer n bits (s2 ++ s1) a c
stackRnnLayers = stackRnnCells

infixr .--.
(.--.) = stackRnnLayers

-- | Compose two rnn layers in parallel.
bothRnnLayers,(.++.)  :: forall s1 s2 a b c n bs bits t. KnownLen s1 => KnownLen s2 => KnownNat n => KnownLen bs
  => RnnLayer n bits s1 a (T (b ': bs) t) -> RnnLayer n bits s2 a (T (c ': bs) t) -> RnnLayer n bits (s2 ++ s1) a (T ((b+c) ': bs) t)
bothRnnLayers f g x =
  f x `bindC` \y ->
  g x `bindC` \z ->
  returnC (concat0 <$> y <*> z)

infixr .++.
(.++.) = bothRnnLayers

-- | Apply a function on the cell state(s) before running the cell itself.
onStates ::  (HTV (Flt t) xs -> HTV (Flt t) xs) -> RnnCell t xs a b -> RnnCell t xs a b
onStates f cell x = C $ \h -> do
  runC (cell x) (f h)

-- | Stack two RNN cells (LHS is run first)
stackRnnCells, (.-.) :: forall s0 s1 a b c t. KnownLen s1
  => RnnCell t s0 a b -> RnnCell t s1 b c -> RnnCell t (s1 ++ s0) a c
stackRnnCells l1 l2 x = l1 x `bindC` l2
(.-.) = stackRnnCells


-- | Compose two rnn cells in parallel.
bothRnnCells, (.|.) :: forall s0 s1 a b c t bs bits. KnownLen s0 => KnownLen s1
  => RnnCell t s0 a (T '[b,bs] (Flt bits))
  -> RnnCell t s1 a (T '[c,bs] (Flt bits))
  -> RnnCell t (s1 ++ s0) a (T '[b+c,bs] (Flt bits))
bothRnnCells l1 l2 x  = 
  l1 x `bindC` \y -> 
  l2 x `bindC` \z -> 
  returnC (concat0 y z)

(.|.) = bothRnnCells


-- | Run the cell, and forward the input to the output, by concatenation with the output of the cell.
withBypass :: forall x y bs t b s0. KnownLen s0
  => RnnCell b s0 (T '[x,bs] t) (T '[y,bs] t) -> RnnCell b s0 (T '[x,bs] t) (T '[x+y,bs] t)
withBypass cell x = appEmpty @s0 $
  cell x `bindC` \y ->
  returnC (concat0 x y)

-- | Run the cell, and feeds its output as input to the next time-step
withFeedback :: forall outputSize inputSize bs w ss.
  RnnCell w ss                       (T '[inputSize+outputSize,bs] (Flt w)) (T '[outputSize,bs] (Flt w)) ->
  RnnCell w ('[outputSize,bs] ': ss) (T '[inputSize           ,bs] (Flt w)) (T '[outputSize,bs] (Flt w))
withFeedback cell x = C $ \(F prevoutputnVector :* s) -> do
  (s',y) <- runC (cell (concat0 x prevoutputnVector)) s
  return (F y :* s',y)

---------------------------------------------------------
-- RNN unfolding

-- | Build a RNN by repeating a cell @n@ times.
rnn :: ∀ n state input output b.
       (KnownNat n) =>
       RnnCell b state input output -> RnnLayer n b state input output
rnn c x = C $ \s -> chainForward (\(t,y) -> runC (c y) t) (s,x)

-- | Build a RNN by repeating a cell @n@ times. However the state is
-- propagated in the right-to-left direction (decreasing indices in
-- the time dimension of the input and output tensors)
rnnBackward :: ∀ n state input output b.
       (KnownNat n) =>
       RnnCell b state input output -> RnnLayer n b state input output
rnnBackward c x = C $ \s -> chainBackward (\(t,y) -> runC (c y) t) (s,x)

-- | RNN helper
chainForward :: ∀ state a b n. ((state , a) -> Gen (state , b)) → (state , V n a) -> Gen (state , V n b)
chainForward _ (s0 , V []) = return (s0 , V [])
chainForward f (s0 , V (x:xs)) = do
  (s1,x') <- f (s0 , x)
  (sFin,V xs') <- chainForward f (s1 , V xs)
  return (sFin,V (x':xs'))

-- | RNN helper
chainBackward :: ∀ state a b n. ((state , a) -> Gen (state , b)) → (state , V n a) -> Gen (state , V n b)
chainBackward _ (s0 , V []) = return (s0 , V [])
chainBackward f (s0 , V (x:xs)) = do
  (s1,V xs') <- chainBackward f (s0,V xs)
  (sFin, x') <- f (s1,x)
  return (sFin,V (x':xs'))


-- | RNN helper
chainForwardWithState :: ∀ state a b n. ((state , a) -> Gen (state , b)) → (state , V n a) -> Gen (V n b, V n state)
chainForwardWithState _ (_s0 , V []) = return (V [], V [])
chainForwardWithState f (s0 , V (x:xs)) = do
  (s1,x') <- f (s0 , x)
  (V xs',V ss) <- chainForwardWithState f (s1 , V xs)
  return (V (x':xs'), V (s1:ss) )

-- -- | RNN helper
-- chainBackwardWithState ::
--   ∀ state a b n. ((state , a) -> Gen (state , b)) → (state , V n a) -> Gen (state , V n b, V n state)
-- chainBackwardWithState _ (s0 , V []) = return (s0 , V [], V [])
-- chainBackwardWithState f (s0 , V (x:xs)) = do
--   (s1,V xs',V ss') <- chainBackwardWithState f (s0,V xs)
--   (sFin, x') <- f (s1,x)
--   return (sFin,V (x':xs'),V (sFin:ss'))

-- | RNN helper
transposeV :: forall n xs t. All KnownLen xs =>
               SList xs -> V n (HTV (Flt t) xs) -> HTV (Flt t) (Ap (FMap (Cons n)) xs)
transposeV LZ _ = Unit
transposeV (LS _ n) xxs  = F ys' :* yys'
  where (ys,yys) = help @(Tail xs) xxs
        ys' = stack0 ys
        yys' = transposeV n yys

        help :: forall ys x tt. V n (HTV tt (x ': ys)) -> (V n (T x tt) , V n (HTV tt ys))
        help (V xs) = (V (map (fromF . hhead) xs),V (map htail xs))

-- | @(gatherFinalStates dynLen states)[i] = states[dynLen[i]]@
gatherFinalStates :: KnownLen x => KnownNat n => LastEqual bs x => T '[bs] Int32 -> T (n ': x) t -> T x t
gatherFinalStates dynLen states = nth0 0 (reverseSequences dynLen states)

-- a more efficient algorithm (perhaps:)
-- gatherFinalStates' :: forall x n bs t. KnownLen x => KnownNat n => LastEqual bs x => T '[bs] Int32 -> T (x ++ '[n,bs]) t -> T x (x ++ '[bs])
-- gatherFinalStates' (T dynLen)t = gather (flattenN2 @x @n @bs t) indexInFlat
--  where indexInFlat = (dynLen - 1) + tf.range(0, bs) * n

gathers :: forall n bs xs t. All (LastEqual bs) xs => All KnownLen xs => KnownNat n =>
            SList xs -> T '[bs] Int32 -> HTV (Flt t) (Ap (FMap (Cons n)) xs) -> HTV (Flt t) xs
gathers LZ _ Unit = Unit
gathers (LS _ n) ixs (F x :* xs) = F (gatherFinalStates ixs x) :* gathers @n n ixs xs

-- | @rnnWithCull dynLen@ constructs an RNN as normal, but returns the
-- state after step @dynLen@ only.
rnnWithCull :: forall n bs x y ls b.
  KnownLen ls => KnownNat n => All KnownLen ls => All (LastEqual bs) ls =>
  T '[bs] Int32 -> RnnCell b ls x y -> RnnLayer n b ls x y
rnnWithCull dynLen cell xs = C $ \s0 -> do
  (us,ss) <- chainForwardWithState (uncurry (flip (runC . cell))) (s0,xs)
  let sss = transposeV @n (shapeSList @ls) ss
  return (gathers @n (shapeSList @ls) dynLen sss,us)

-- -- | Like @rnnWithCull@, but states are threaded backwards.
-- rnnBackwardsWithCull :: forall n bs x y ls b.
--   KnownLen ls => KnownNat n => All KnownLen ls => All (LastEqual bs) ls =>
--   T '[bs] Int32 -> RnnCell b ls x y -> RnnLayer n b ls x y
-- rnnBackwardsWithCull dynLen cell (s0, t) = do
--  (us,ss) <- chainBackwardWithState cell (s0,xs)
  -- let sss = transposeV @n (shapeSList @ls) ss
  -- return (gathers @n (shapeSList @ls) (n - dynLen) sss,us)

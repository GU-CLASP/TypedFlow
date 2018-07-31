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
  -- * Cell Combinators
  RnnCell,
  simpleRnn,
  runCell, mkCell,
  stackRnnCells, (.-.),
  bothRnnCells, (.|.),
  withBypass, withFeedback,
  onStates,
  -- * Rnn Combinators
  Rnn,
  runRnn,
  stackRnns, (.--.),
  bothRnns,(.++.),
  -- * RNN unfolding functions
  timeDistribute,
  iterateCell,
  iterateCellBackward,
  iterateWithCull,
  -- * Monad-like interface for cell construction
  Component(..), bindC, returnC,
  -- rnnBackwardsWithCull,
  )

where
import Prelude hiding (tanh,Num(..),Floating(..),floor)
import GHC.TypeLits
import TypedFlow.TF
import TypedFlow.Types
import TypedFlow.Types.Proofs
-- import Data.Type.Equality
-- import Data.Kind (Type,Constraint)

-- | The RNN Component generalized monad. This can be used to build
-- RNNs cells which do not follow the simple and usual "stacking"
-- pattern. This is not a simple monad, because the indexing over
-- states is non-uniform; see 'BindC'.
newtype Component t states a = C {runC :: HTV (Flt t) states -> (HTV (Flt t) states , a)}

instance Functor (Component t states) where
  fmap = mapC


mapC :: (a -> b) -> Component t s a -> Component t s b
mapC f c = C $ \s ->
  let (s',x) = runC c s
  in (s', f x)

-- | Unit of the Component monad.
returnC :: a -> Component t '[] a
returnC x = C $ \Unit -> (Unit,x)

-- | Bind operation for Components. States are accumulated.
bindC :: forall t s0 s1 a b. KnownLen s1
  => Component t s0 a -> (a -> Component t s1 b) -> Component t (s1++s0) b
bindC f g = C $ \(hsplit @s1 -> (s1,s0)) -> 
  let (s0',x) = runC f s0
      (s1',y) = runC (g x) s1
  in (happ s1' s0',y)

-- | A cell (one time-step) in an rnn. @state@ is the state propagated through time.
type RnnCell t states input output = input -> Component t states output

-- | An rnn. @n@ is the length of the time sequence. @state@ is the state propagated through time.
type Rnn n b state input output = RnnCell b state (V n input) (V n output) 

-- | Run a cell
runCell :: RnnCell t states input output -> (HTV (Flt t) states,input) -> (HTV (Flt t) states, output)
runCell cell = uncurry (flip (runC . cell))

-- | Run an RNN, using a tensor as input. @n@ is the length of the time sequence. 
runRnn :: (KnownNat n,KnownShape s0, KnownShape s1, KnownTyp t1)
       => Rnn n t2 states (T s1 t1) (T s0 t0)
       -> (HTV (Flt t2) states, Tensor (n ': s1) t1)
       -> (HTV (Flt t2) states, Tensor (n ': s0) t0)
runRnn l (s,x) =
  let x' = unstack0 x
      (s',y) = runCell l (s,x')
  in (s',stack0 y)

-- | Run an RNN composed of a single RNN cell.
simpleRnn :: KnownTyp t1 => KnownShape s1 => KnownShape s0 => KnownNat n
          => RnnCell t2 states (T s1 t1) (T s0 t0)
          -> (HTV (Flt t2) states, Tensor (n : s1) t1)
          -> (HTV (Flt t2) states, Tensor (n : s0) t0)
simpleRnn = runRnn . iterateCell

-- | Construct a cell from an arbitrary stateful function
mkCell :: ((HTV (Flt t) states,input) -> (HTV (Flt t) states, output)) -> RnnCell t states input output
mkCell cell = C . flip (curry cell)

----------------------
-- Lifting functions

-- | Convert a pure function (feed-forward layer) to an RNN cell by
-- ignoring the RNN state.
timeDistribute :: (a -> b) -> RnnCell t '[] a b
timeDistribute stateLess a = returnC (stateLess a)

--------------------------------------
-- Combinators

-- | Compose two rnn layers. This is useful for example to combine
-- forward and backward layers.
(.--.),stackRnns :: forall s1 s2 a b c n bits. KnownLen s2
  => Rnn n bits s1 a b -> Rnn n bits s2 b c -> Rnn n bits (s2 ++ s1) a c
stackRnns = stackRnnCells

infixr .--.
(.--.) = stackRnns

-- | Compose two rnn layers in parallel.
bothRnns,(.++.)  :: forall s1 s2 a b c n bits t.
  KnownTyp t => KnownLen s1 => KnownLen s2 => KnownNat n
  => KnownNat b => KnownNat c
  => Rnn n bits s1 a (T '[b] t) -> Rnn n bits s2 a (T '[c] t) -> Rnn n bits (s2 ++ s1) a (T ('[b+c]) t)
bothRnns f g x =
  f x `bindC` \y ->
  g x `bindC` \z ->
  returnC (concat0 <$> y <*> z)

infixr .++.
(.++.) = bothRnns

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
bothRnnCells, (.|.) :: forall s0 s1 a b c t bits. KnownLen s0 => KnownLen s1
  => KnownBits bits
  => KnownNat b => KnownNat c
  => RnnCell t s0 a (T '[b] (Flt bits))
  -> RnnCell t s1 a (T '[c] (Flt bits))
  -> RnnCell t (s1 ++ s0) a (T '[b+c] (Flt bits))
bothRnnCells l1 l2 x  =
  l1 x `bindC` \y ->
  l2 x `bindC` \z ->
  returnC (concat0 y z)

(.|.) = bothRnnCells


-- | Run the cell, and forward the input to the output, by
-- concatenation with the output of the cell. This bypass is sometimes
-- called a 'highway' in the literature.
withBypass :: forall x y t b s0. KnownNat x => KnownNat y => KnownLen s0
  => KnownTyp t
  => RnnCell b s0 (T '[x] t) (T '[y] t) -> RnnCell b s0 (T '[x] t) (T '[x+y] t)
withBypass cell x = appRUnit @s0 $
  cell x `bindC` \y ->
  returnC (concat0 x y)

-- | Run the cell, and feeds its output as input to the next time-step
withFeedback :: forall outputSize inputSize w ss.
  KnownBits w => KnownNat outputSize => KnownNat inputSize =>
  RnnCell w ss                    (T '[inputSize+outputSize] (Flt w)) (T '[outputSize] (Flt w)) ->
  RnnCell w ('[outputSize] ': ss) (T '[inputSize           ] (Flt w)) (T '[outputSize] (Flt w))
withFeedback cell x = C $ \(F prevoutputnVector :* s) -> 
  let (s',y) = runC (cell (concat0 x prevoutputnVector)) s
  in  (F y :* s',y)

---------------------------------------------------------
-- RNN unfolding

-- | Build a RNN by repeating a cell @n@ times.
iterateCell :: ∀ n state input output b.
       (KnownNat n) =>
       RnnCell b state input output -> Rnn n b state input output
iterateCell c x = C $ \s -> chainForward (\(t,y) -> runC (c y) t) (s,x)

-- | Build a RNN by repeating a cell @n@ times. However the state is
-- propagated in the right-to-left direction (decreasing indices in
-- the time dimension of the input and output tensors)
iterateCellBackward :: ∀ n state input output b.
       (KnownNat n) =>
       RnnCell b state input output -> Rnn n b state input output
iterateCellBackward c x = C $ \s -> chainBackward (\(t,y) -> runC (c y) t) (s,x)

-- | RNN helper
chainForward :: ∀ state a b n. ((state , a) -> (state , b)) → (state , V n a) -> (state , V n b)
chainForward _ (s0 , V []) = (s0 , V [])
chainForward f (s0 , V (x:xs)) = 
  let (s1,x') = f (s0 , x)
      (sFin,V xs') = chainForward f (s1 , V xs)
  in  (sFin,V (x':xs'))

-- | RNN helper
chainBackward :: ∀ state a b n. ((state , a) -> (state , b)) → (state , V n a) -> (state , V n b)
chainBackward _ (s0 , V []) = (s0 , V [])
chainBackward f (s0 , V (x:xs)) =
  let (s1,V xs') = chainBackward f (s0,V xs)
      (sFin, x') = f (s1,x)
  in (sFin,V (x':xs'))


-- | RNN helper
chainForwardWithState :: ∀ state a b n. ((state , a) -> (state , b)) → (state , V n a) -> (V n b, V n state)
chainForwardWithState _ (_s0 , V []) = (V [], V [])
chainForwardWithState f (s0 , V (x:xs)) =
  let (s1,x') = f (s0 , x)
      (V xs',V ss) = chainForwardWithState f (s1 , V xs)
  in (V (x':xs'), V (s1:ss) )

-- -- | RNN helper
-- chainBackwardWithState ::
--   ∀ state a b n. ((state , a) -> (state , b)) → (state , V n a) -> (state , V n b, V n state)
-- chainBackwardWithState _ (s0 , V []) = return (s0 , V [], V [])
-- chainBackwardWithState f (s0 , V (x:xs)) = do
--   (s1,V xs',V ss') <- chainBackwardWithState f (s0,V xs)
--   (sFin, x') <- f (s1,x)
--   return (sFin,V (x':xs'),V (sFin:ss'))

-- | RNN helper
transposeV :: forall n xs t. All KnownShape xs => KnownNat n =>
               SList xs -> V n (HTV (Flt t) xs) -> HTV (Flt t) (Ap (FMap (Cons n)) xs)
transposeV Unit _ = Unit
transposeV (_ :* n) xxs  = F ys' :* yys'
  where (ys,yys) = help @(Tail xs) xxs
        ys' = stack0 ys
        yys' = transposeV n yys

        help :: forall ys x tt. V n (HTV tt (x ': ys)) -> (V n (T x tt) , V n (HTV tt ys))
        help (V xs) = (V (map (fromF . hhead) xs),V (map htail xs))

-- | @(gatherFinalStates dynLen states)[i] = states[dynLen[i]-1]@
gatherFinalStates :: KnownShape x => KnownNat n => T '[] Int32 -> T (n ': x) t -> T x t
gatherFinalStates dynLen states = gather states (dynLen ⊝ constant 1)

gathers :: forall n xs t. All KnownShape xs => KnownNat n =>
            SList xs -> T '[] Int32 -> HTV (Flt t) (Ap (FMap (Cons n)) xs) -> HTV (Flt t) xs
gathers Unit _ Unit = Unit
gathers (_ :* n) ixs (F x :* xs) = F (gatherFinalStates ixs x) :* gathers @n n ixs xs

-- | @rnnWithCull dynLen@ constructs an RNN as normal, but returns the
-- state after step @dynLen@ only.
iterateWithCull :: forall n x y ls b.
  KnownLen ls => KnownNat n => All KnownShape ls =>
  T '[] Int32 -- ^ dynamic length
  -> RnnCell b ls x y -> Rnn n b ls x y
iterateWithCull dynLen cell xs = C $ \s0 ->
  let (us,ss) = chainForwardWithState (uncurry (flip (runC . cell))) (s0,xs)
      sss = transposeV @n (typeSList @ls) ss
  in (gathers @n (typeSList @ls) dynLen sss,us)

-- -- | Like @rnnWithCull@, but states are threaded backwards.
-- rnnBackwardsWithCull :: forall n bs x y ls b.
--   KnownLen ls => KnownNat n => All KnownLen ls => All (LastEqual bs) ls =>
--   T '[bs] Int32 -> RnnCell b ls x y -> RNN n b ls x y
-- rnnBackwardsWithCull dynLen cell (s0, t) = do
--  (us,ss) <- chainBackwardWithState cell (s0,xs)
  -- let sss = transposeV @n (shapeSList @ls) ss
  -- return (gathers @n (shapeSList @ls) (n - dynLen) sss,us)

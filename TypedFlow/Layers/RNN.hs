{-|
Module      : TypedFlow.Layers.RNN
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

module TypedFlow.Layers.RNN (
  -- * Types
  RnnCell, RnnLayer,
  -- * Combinators
  stackRnnCells, (.-.),
  stackRnnLayers, (.--.),
  bothRnnLayers,(.++.),
  withBypass,
  onStates,
  timeDistribute, timeDistribute',
  -- * RNN Cells
  cellInitializerBit,
  LSTMP(..),
  lstm,
  GRUP(..),
  gru,
  -- * RNN unfolding functions
  rnn,
  rnnBackward,
  rnnBackwardsWithCull,
  rnnWithCull,
  -- * Attention mechanisms
  -- ** Scoring functions
  AttentionScoring,
  multiplicativeScoring,
  AdditiveScoringP(..), additiveScoring,
  -- ** Attention functions
  AttentionFunction,
  uniformAttn,
  luongAttention,
  -- ** Attention combinators
  attentiveWithFeedback
  )

where

import Prelude hiding (tanh,Num(..),Floating(..),floor)
import GHC.TypeLits
-- import Text.PrettyPrint.Compact (float)
import TypedFlow.TF
import TypedFlow.Types
import TypedFlow.Layers.Core (DenseP(..),(#))
-- import Data.Type.Equality
-- import Data.Kind (Type,Constraint)
import Data.Monoid ((<>))


-- | A cell in an rnn. @state@ is the state propagated through time.
type RnnCell t states input output = (HTV (Flt t) states , input) -> Gen (HTV (Flt t) states , output)

-- | A layer in an rnn. @n@ is the length of the time sequence. @state@ is the state propagated through time.
type RnnLayer b n state input t output u = HTV (Flt b) state -> Tensor (n ': input) t -> Gen (HTV (Flt b) state , Tensor (n ': output) u)

--------------------------------------
-- Combinators


-- | Compose two rnn layers. This is useful for example to combine
-- forward and backward layers.
(.--.),stackRnnLayers :: forall s1 s2 a t b u c v n bits. KnownLen s1 =>
                  RnnLayer bits n s1 a t b u -> RnnLayer bits n s2 b u c v -> RnnLayer bits n (s1 ++ s2) a t c v
stackRnnLayers f g (hsplit @s1 -> (s0,s1)) x = do
  (s0',y) <- f s0 x
  (s1',z) <- g s1 y
  return (happ s0' s1',z)

infixr .--.
(.--.) = stackRnnLayers


-- | Compose two rnn layers in parallel.
bothRnnLayers,(.++.)  :: forall s1 s2 a t b u c n bs bits. KnownLen s1 =>
                  RnnLayer bits n s1 a t '[b,bs] u -> RnnLayer bits n s2 a t '[c,bs] u -> RnnLayer bits n (s1 ++ s2) a t '[b+c,bs] u
bothRnnLayers f g (hsplit @s1 -> (s0,s1)) x = do
  (s0',y) <- f s0 x
  (s1',z) <- g s1 x
  return (happ s0' s1',concat1 y z)


infixr .++.
(.++.) = bothRnnLayers

-- | Apply a function on the cell state(s) before running the cell itself.
onStates ::  (HTV (Flt t) xs -> HTV (Flt t) xs) -> RnnCell t xs a b -> RnnCell t xs a b
onStates f cell (h,x) = do
  cell (f h, x)

-- | Stack two RNN cells (LHS is run first)
stackRnnCells, (.-.) :: forall s0 s1 a b c t. KnownLen s0 => RnnCell t s0 a b -> RnnCell t s1 b c -> RnnCell t (s0 ++ s1) a c
stackRnnCells l1 l2 (hsplit @s0 -> (s0,s1),x) = do
  (s0',y) <- l1 (s0,x)
  (s1',z) <- l2 (s1,y)
  return ((happ s0' s1'),z)

(.-.) = stackRnnCells

-- | Run the cell, and forward the input to the output, by concatenation with the output of the cell.
withBypass :: RnnCell b s0 (T '[x,bs] t) (T '[y,bs] t) -> RnnCell b s0 (T '[x,bs] t) (T '[x+y,bs] t)
withBypass cell (s,x) = do
  (s',y) <- cell (s,x)
  return (s',concat0 x y)

--------------------------------------
-- Cells

-- | Convert a pure function (feed-forward layer) to an RNN cell by
-- ignoring the RNN state.
timeDistribute :: (a -> b) -> RnnCell t '[] a b
timeDistribute pureLayer = timeDistribute' (return . pureLayer)

-- | Convert a stateless generator into an RNN cell by ignoring the
-- RNN state.
timeDistribute' :: (a -> Gen b) -> RnnCell t '[] a b
timeDistribute' stateLess (Unit,a) = do
  b <- stateLess a
  return (Unit,b)

-- | Standard RNN gate initializer. (The recurrent kernel is
-- orthogonal to avoid divergence; the input kernel is glorot)
cellInitializerBit :: ∀ n x t. (KnownNat n, KnownNat x, KnownBits t) => DenseP t (n + x) n
cellInitializerBit = DenseP (concat0 recurrentInitializer kernelInitializer) biasInitializer
  where
        recurrentInitializer :: Tensor '[n, n] ('Typ 'Float t)
        recurrentInitializer = randomOrthogonal
        kernelInitializer :: Tensor '[x, n] ('Typ 'Float t)
        kernelInitializer = glorotUniform
        biasInitializer = zeros

-- | Parameter for an LSTM
data LSTMP t n x = LSTMP (DenseP t (n+x) n) (DenseP t (n+x) n) (DenseP t (n+x) n) (DenseP t (n+x) n)

instance (KnownNat n, KnownNat x, KnownBits t) => KnownTensors (LSTMP t n x) where
  travTensor f s (LSTMP x y z w) = LSTMP <$> travTensor f (s<>"_f") x <*> travTensor f (s<>"_i") y <*> travTensor f (s<>"_c") z <*> travTensor f (s<>"_o") w
instance (KnownNat n, KnownNat x, KnownBits t) => ParamWithDefault (LSTMP t n x) where
  defaultInitializer = LSTMP forgetInit cellInitializerBit cellInitializerBit cellInitializerBit
    where forgetInit = DenseP (denseWeights cellInitializerBit) ones

-- | Standard LSTM
lstm :: ∀ n x bs t. LSTMP t n x ->
        RnnCell t '[ '[n,bs], '[n,bs]] (Tensor '[x,bs] (Flt t)) (Tensor '[n,bs] (Flt t))
lstm (LSTMP wf wi wc wo) (VecPair ht1 ct1, input) = do
  hx <- assign (concat0 ht1 input)
  let f = sigmoid (wf # hx)
      i = sigmoid (wi # hx)
      cTilda = tanh (wc # hx)
      o = sigmoid (wo # hx)
  c <- assign ((f ⊙ ct1) + (i ⊙ cTilda))
  h <- assign (o ⊙ tanh c)
  return (VecPair h c, h)

-- -- | LSTM for an attention model. The result of attention is combined using + to generate output (bad!)
-- attentiveLstmPlus :: forall x n bs t. KnownNat bs =>
--   AttentionFunction t bs n n ->
--   LSTMP t n x ->
--   RnnCell t '[ '[n,bs], '[n,bs]] (Tensor '[x,bs] (Flt t)) (Tensor '[n,bs] (Flt t))
-- attentiveLstmPlus att w x = do
--   (VecPair ht ct, _ht) <- lstm w x
--   a <- att ht
--   let ht' = ht ⊕ a -- alternatively add a dense layer to combine
--   return (VecPair ht' ct, a)

-- | Parameter for a GRU
data GRUP t n x = GRUP (T [n+x,n] ('Typ 'Float t)) (T [n+x,n] ('Typ 'Float t)) (T [n+x,n] ('Typ 'Float t))

instance (KnownNat n, KnownNat x, KnownBits t) => KnownTensors (GRUP t n x) where
  travTensor f s (GRUP x y z) = GRUP <$> travTensor f (s<>"_z") x <*> travTensor f (s<>"_r") y <*> travTensor f (s<>"_w") z
instance (KnownNat n, KnownNat x, KnownBits t) => ParamWithDefault (GRUP t n x) where
  defaultInitializer = GRUP (denseWeights cellInitializerBit) (denseWeights cellInitializerBit) (denseWeights cellInitializerBit)


-- | Standard GRU cell
gru :: ∀ n x bs t. (KnownNat bs, KnownNat n, KnownBits t) => GRUP t n x ->
        RnnCell t '[ '[n,bs] ] (Tensor '[x,bs] (Flt t)) (Tensor '[n,bs] (Flt t))
gru (GRUP wz wr w) (VecSing ht1, xt) = do
  hx <- assign (concat0 ht1 xt)
  let zt = sigmoid (wz ∙ hx)
      rt = sigmoid (wr ∙ hx)
      hTilda = tanh (w ∙ (concat0 (rt ⊙ ht1) xt))
  ht <- assign ((ones ⊝ zt) ⊙ ht1 + zt ⊙ hTilda)
  return (VecSing ht, ht)

----------------------------------------------
-- "Attention" layers


-- | An attention scoring function. This function should produce a
-- score (between 0 and 1) for each of the @nValues@ entries of size
-- @valueSize@.
type AttentionScoring t batchSize keySize valueSize nValues = 
  Tensor '[keySize,batchSize] ('Typ 'Float t) -> Tensor '[nValues,valueSize,batchSize] ('Typ 'Float t) -> Tensor '[nValues,batchSize] ('Typ 'Float t)

-- | A function which attends to an external input. Typically a
-- function of this type is a closure which has the attended input in
-- its environment.
type AttentionFunction t batchSize keySize valueSize =
  T '[keySize,batchSize] (Flt t) -> Gen (T '[valueSize,batchSize] (Flt t))

{- NICER, SLOW

type AttentionScoring t batchSize keySize valueSize =
  Tensor '[keySize,batchSize] ('Typ 'Float t) -> Tensor '[valueSize,batchSize] ('Typ 'Float t) -> Tensor '[batchSize] ('Typ 'Float t)


-- | @attnExample1 θ h st@ combines each element of the vector h with
-- s, and applies a dense layer with parameters θ. The "winning"
-- element of h (using softmax) is returned.
uniformAttn :: ∀ valueSize m keySize batchSize t. KnownNat m => KnownBits t =>
               T '[batchSize] Int32 ->
               AttentionScoring t batchSize keySize valueSize ->
               T '[m,valueSize,batchSize] (Flt t) -> AttentionFunction t batchSize keySize valueSize
uniformAttn lengths score hs_ ht = do
  xx <- mapT (score ht) hs_
  let   αt :: T '[m,batchSize] (Flt t)
        αt = softmax0 (mask ⊙ xx)
        ct :: T '[valueSize,batchSize] (Flt t)
        ct = squeeze0 (matmul hs_ (expandDim0 αt))
        mask = cast (sequenceMask @m lengths) -- mask according to length
  return ct



-- | A multiplicative scoring function. See 
-- https://github.com/tensorflow/nmt#background-on-the-attention-mechanism
-- commit 75aa22dfb159f10a1a5b4557777d9ff547c1975a
multiplicativeScoring :: forall valueSize keySize batchSize t.
  T [keySize,valueSize] ('Typ 'Float t) ->  AttentionScoring t batchSize keySize valueSize
multiplicativeScoring w dt h = h · ir
  where ir :: T '[valueSize,batchSize] ('Typ 'Float t)
        ir = w ∙ dt


additiveScoring :: AdditiveScoringP sz keySize valueSize t -> AttentionScoring t batchSize valueSize keySize
additiveScoring (AdditiveScoringP v w1 w2) dt h = squeeze0 (v ∙ tanh ((w1 ∙ h) ⊕ (w2 ∙ dt)))

-}

-- | @attnExample1 θ h st@ combines each element of the vector h with
-- s, and applies a dense layer with parameters θ. The "winning"
-- element of h (using softmax) is returned.
uniformAttn :: ∀ valueSize m keySize batchSize t. KnownNat m => KnownBits t
            => AttentionScoring t batchSize keySize valueSize m -- ^ scoring function
            -> T '[batchSize] Int32 -- ^ lengths of the inputs
            -> T '[m,valueSize,batchSize] (Flt t) -- ^ inputs
            -> AttentionFunction t batchSize keySize valueSize
uniformAttn score lengths hs_ ht = do
  let   αt :: T '[m,batchSize] (Flt t)
        xx = score ht hs_
        αt = softmax0 (mask ⊙ xx)
        ct :: T '[valueSize,batchSize] (Flt t)
        ct = squeeze0 (matmul hs_ (expandDim0 αt))
        mask = cast (sequenceMask @m lengths) -- mask according to length
  return ct

-- | Add some attention to an RnnCell, and feed the attention vector to
-- the next iteration in the rnn. (This follows the diagram at
-- https://github.com/tensorflow/nmt#background-on-the-attention-mechanism
-- commit 75aa22dfb159f10a1a5b4557777d9ff547c1975a).
attentiveWithFeedback ::forall attSize cellSize inputSize bs w ss.
  AttentionFunction w bs cellSize attSize ->
  RnnCell w ss                      (T '[inputSize+attSize,bs] (Flt w)) (T '[cellSize,bs] (Flt w)) ->
  RnnCell w ('[attSize,bs] ': ss)   (T '[inputSize        ,bs] (Flt w)) (T '[attSize,bs] (Flt w))
attentiveWithFeedback attn cell ((F prevAttnVector :* s),x) = do
  (s',y) <- cell (s,concat0 x prevAttnVector)
  focus <- attn y
  return ((F focus :* s'),focus)

-- -- | LSTM for an attention model. The result of attention is fed to the next step.
-- attentiveLstm :: forall attSize n x bs t. KnownNat bs =>
--   AttentionFunction t bs n attSize ->
--   LSTMP t n (x+attSize) ->
--   RnnCell t '[ '[attSize,bs], '[n,bs], '[n,bs] ] (Tensor '[x,bs] (Flt t)) (Tensor '[attSize,bs] (Flt t))
-- attentiveLstm att w = attentiveWithFeedback att (lstm w)


-- | Luong attention function (following
-- https://github.com/tensorflow/nmt#background-on-the-attention-mechanism
-- commit 75aa22dfb159f10a1a5b4557777d9ff547c1975a).
-- Essentially a dense layer with tanh activation, on top of uniform attention.
luongAttention :: ∀ attnSize d m e batchSize w. KnownNat m => KnownBits w
  => Tensor '[d+e,attnSize] (Flt w)     -- ^ weights for the dense layer
  -> AttentionScoring w batchSize e d m -- ^ scoring function
  -> Tensor '[batchSize] Int32          -- ^ length of the input
  -> T '[m,d,batchSize] (Flt w)         -- ^ inputs
  -> AttentionFunction w batchSize e attnSize
luongAttention w scoring lens hs_ ht = do
  ct <- uniformAttn scoring lens hs_ ht
  return (tanh (w ∙ (concat0 ct ht)))

-- | Multiplicative scoring function
multiplicativeScoring :: forall valueSize keySize batchSize nValues t.
  KnownNat batchSize => T [keySize,valueSize] ('Typ 'Float t) ->  AttentionScoring t batchSize keySize valueSize nValues 
multiplicativeScoring w dt hs = squeeze1 (matmul (expandDim1 ir) hs)
  where ir :: T '[valueSize,batchSize] ('Typ 'Float t)
        ir = w ∙ dt


data AdditiveScoringP sz keySize valueSize t = AdditiveScoringP
  (Tensor '[sz, 1]         ('Typ 'Float t))
  (Tensor '[keySize, sz]   ('Typ 'Float t))
  (Tensor '[valueSize, sz] ('Typ 'Float t))

instance (KnownNat n, KnownNat k, KnownNat v, KnownBits t) => KnownTensors (AdditiveScoringP k v n t) where
  travTensor f s (AdditiveScoringP x y z) = AdditiveScoringP <$> travTensor f (s<>"_v") x <*> travTensor f (s<>"_w1") y <*> travTensor f (s<>"_w2") z
instance (KnownNat n, KnownNat k, KnownNat v, KnownBits t) => ParamWithDefault (AdditiveScoringP k v n t) where
  defaultInitializer = AdditiveScoringP glorotUniform glorotUniform glorotUniform

-- | An additive scoring function. See https://arxiv.org/pdf/1412.7449.pdf
additiveScoring :: forall sz keySize valueSize t nValues batchSize. KnownNat sz => KnownNat keySize => (KnownNat nValues, KnownNat batchSize) =>
  AdditiveScoringP sz keySize valueSize t -> AttentionScoring t batchSize valueSize keySize nValues
additiveScoring (AdditiveScoringP v w1 w2) dt h = transpose r''
  where w1h :: Tensor '[sz,batchSize, nValues] ('Typ 'Float t)
        w1h = transposeN01 @'[sz] (reshape @'[sz,nValues, batchSize] w1h')
        w1h' = matmul (reshape @'[keySize, nValues*batchSize] (transpose01 h)) (transpose01 w1)
        w2dt = w2 ∙ dt
        z' = reshape @'[sz,batchSize*nValues] (tanh (w1h + w2dt))
        r'' = reshape @[batchSize,nValues] (matmul z' (transpose v))

---------------------------------------------------------
-- RNN unfolding


-- | Build a RNN by repeating a cell @n@ times.
rnn :: ∀ n state input output t u b.
       (KnownNat n, KnownShape input, KnownShape output) =>
       RnnCell b state (T input t) (T output u) -> RnnLayer b n state input t output u
rnn cell s0 t = do
  xs <- unstack t
  (sFin,us) <- chainForward cell (s0,xs)
  return (sFin,stack us)
-- There will be lots of stacking and unstacking at each layer for no
-- reason; we should change the in/out from tensors to vectors of
-- tensors.

-- | Build a RNN by repeating a cell @n@ times. However the state is
-- propagated in the right-to-left direction (decreasing indices in
-- the time dimension of the input and output tensors)
rnnBackward :: ∀ n state input output t u b.
       (KnownNat n, KnownShape input, KnownShape output) =>
       RnnCell b state (T input t) (T output u) -> RnnLayer b n state input t output u

rnnBackward cell s0 t = do
  xs <- unstack t
  (sFin,us) <- chainBackward cell (s0,xs)
  return (sFin,stack us)



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
        ys' = stack ys
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
rnnWithCull :: forall n bs x y t u ls b.
  KnownLen ls => KnownNat n => KnownLen x  => KnownLen y => All KnownLen ls =>
  All (LastEqual bs) ls =>
  T '[bs] Int32 -> RnnCell b ls (T x t) (T y u) -> RnnLayer b n ls x t y u
rnnWithCull dynLen cell s0 t = do
  xs <- unstack t
  (us,ss) <- chainForwardWithState cell (s0,xs)
  let sss = transposeV @n (shapeSList @ls) ss
  return (gathers @n (shapeSList @ls) dynLen sss,stack us)

-- | Like @rnnWithCull@, but states are threaded backwards.
rnnBackwardsWithCull :: forall n bs x y t u ls b.
  KnownLen ls => KnownNat n => KnownLen x  => KnownLen y => All KnownLen ls =>
  All (LastEqual bs) ls => LastEqual bs x => LastEqual bs y =>
  T '[bs] Int32 -> RnnCell b ls (T x t) (T y u) -> RnnLayer b n ls x t y u
rnnBackwardsWithCull dynLen cell s0 t = do
  (sFin,hs) <- rnnWithCull dynLen cell s0 (reverseSequences dynLen t)
  hs' <- assign (reverseSequences dynLen hs)
  return (sFin, hs')


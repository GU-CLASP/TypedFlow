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

module TypedFlow.Layers where

import Prelude hiding (tanh,Num(..),Floating(..))
import qualified Prelude ()
import GHC.TypeLits
import Text.PrettyPrint.Compact (float)
import TypedFlow.TF
import TypedFlow.Types
-- import Data.Kind (Type)

---------------------
-- Linear functions


-- A linear function form a to b is a matrix and a bias.
type (a ⊸ b) = (Tensor '[a,b] Float32, Tensor '[b] Float32)

-- | Apply a linear function
(#) :: (a ⊸ b) -> T '[a,batchSize] Float32 -> Tensor '[b,batchSize] Float32
(weightMatrix, bias) # v = weightMatrix ∙ v + bias

-----------------------
-- Feed-forward layers

-- | embedding layer

type EmbbeddingP numObjects embeddingSize t = Tensor '[numObjects, embeddingSize] ('Typ 'Float t)

embeddingInitializer :: (KnownNat numObjects, KnownBits b, KnownNat embeddingSize) => EmbbeddingP numObjects embeddingSize b
embeddingInitializer = randomUniform (-1) 1

embedding :: ∀ embeddingSize numObjects batchSize t.
             EmbbeddingP numObjects embeddingSize t -> Tensor '[batchSize] Int32 -> Tensor '[embeddingSize,batchSize] ('Typ 'Float t)
embedding param input = gather @ '[embeddingSize] (transpose param) input


denseInitialiser :: (KnownNat n, KnownNat m) => (n ⊸ m)
denseInitialiser = (glorotUniform,truncatedNormal 0.1)

dense :: ∀m n batchSize. (n ⊸ m) -> Tensor '[n, batchSize] Float32 -> (Tensor '[m, batchSize] Float32)
dense lf t = (lf # t)

data KeepProb = KeepProb Float

dropout :: KeepProb -> Tensor s t -> Tensor s t
dropout (KeepProb p) (T x) = T (funcall "tf.nn.dropout" [x, float p])

------------------------
-- Convolutional layers
convInitialiser :: (KnownShape s1, KnownShape s2) =>
                   (T s1 ('Typ 'Float w), T s2 ('Typ 'Float w))
convInitialiser = (truncatedNormal 0.1, constant 0.1)

conv :: forall outChannels filterSpatialShape inChannels s t.
                  ((1 + Length filterSpatialShape) ~ Length s,
                   KnownLen filterSpatialShape,
                   KnownShape s) => -- the last dim of s is the batch size
                  (T ('[outChannels,inChannels] ++ filterSpatialShape) t, T ('[outChannels] ++ Init s) t) ->
                  T ('[inChannels] ++ s) t -> (T ('[outChannels] ++ s) t)
conv (filters,bias) input = (initLast @s (add @'[Last s] c  bias))
 where c = (convolution input filters)


maxPool2D :: forall stridex (stridey::Nat) batch height width channels.
             (KnownNat stridex, KnownNat stridey) =>
             T '[channels,width*stridex,height*stridex,batch] Float32 -> T '[channels,width,height,batch] Float32
maxPool2D (T value) = T (funcall "tf.nn.max_pool" [value
                                                  ,showShape @'[1,stridex,stridey,1]
                                                  ,showShape @'[1,stridex,stridey,1]
                                                  ,named "padding" (str "SAME") ])

-------------------------------
-- RNN layers and combinators

type RnnCell states input output = (HList states , input) -> Gen (HList states , output)


-- | Any pure function (feed-forward layer) can be transformed into a
-- cell by ignoring the RNN state.

timeDistribute :: (a -> b) -> RnnCell '[] a b
timeDistribute pureLayer (Unit,a) = return (Unit, pureLayer a)

cellInitializerBit :: ∀ n x. (KnownNat n, KnownNat x) => (n + x) ⊸ n
cellInitializerBit = (concat0 recurrentInitializer kernelInitializer,biasInitializer)
  where
        recurrentInitializer :: Tensor '[n, n] Float32
        recurrentInitializer = randomOrthogonal
        kernelInitializer :: Tensor '[x, n] Float32
        kernelInitializer = glorotUniform
        biasInitializer = zeros

type LSTMP n x = (((n + x) ⊸ n),
                  ((n + x) ⊸ n),
                  ((n + x) ⊸ n),
                  ((n + x) ⊸ n))

lstmInitializer :: (KnownNat n, KnownNat x) => LSTMP n x
lstmInitializer = (cellInitializerBit, cellInitializerBit, cellInitializerBit,cellInitializerBit)


lstm :: ∀ n x bs. (KnownNat bs) => LSTMP n x ->
        RnnCell '[(T '[n,bs] Float32, T '[n,bs] Float32)] (Tensor '[x,bs] Float32) (Tensor '[n,bs] Float32)
lstm (wf,wi,wc,wo) ((I (ht1, ct1) :* Unit) , input) = do
  hx <- assign (concat0 ht1 input)
  let f = sigmoid (wf # hx)
      i = sigmoid (wi # hx)
      cTilda = tanh (wc # hx)
      o = sigmoid (wo # hx)
  c <- assign ((f ⊙ ct1) + (i ⊙ cTilda))
  h <- assign (o + tanh c)
  return (I (c,h) :* Unit , h)

-- type GRUP n x =
  
-- gru :: ∀ n x bs. (KnownNat bs) => GRUP n x ->
--         RnnCell (T '[n,bs] Float32) (Tensor '[x,bs] Float32) (Tensor '[n,bs] Float32)


-- recurrentDropout 

-- -- | Stack two RNN cells
-- stackRnnCells :: RnnCell s0 a b -> RnnCell s1 b c -> RnnCell (s0,s1) a c
-- stackRnnCells l1 l2 ((s0,s1),x) = do
--   (s0',y) <- l1 (s0,x)
--   (s1',z) <- l2 (s1,y)
--   return ((s0',s1'),z)

-- idLayer :: RnnLayer n (HList '[]) a t a t
-- idLayer st x = return (st,x)

-- (.|.) :: forall s0 a b s1 c. RnnCell s0 a b -> RnnCell s1 b c -> RnnCell (s0, s1) a c
-- (.|.) = stackRnnCells

-- | @addAttention attn l@ adds the attention function @attn@ to the
-- rnn cell @l@.  Note that @attn@ can depend in particular on a
-- constant external value @h@ which is the complete input to pay
-- attention to.  The type parameter @x@ is the size of the portion of
-- @h@ that the cell @l@ will observe.
addAttention :: KnownShape s =>
                (HList states -> Gen (T (x ': s) t)) ->
                RnnCell states (T ((a+x) ': s) t) (T (b ': s) t) ->
                RnnCell states (T ( a    ': s) t) (T (b ': s) t)
addAttention attn l (s,a) = do
  focus <- attn s
  l (s,concat0 a focus)

-- | @attnExample1 θ h st@ combines each element of the vector h with
-- s, and applies a dense layer with parameters θ. The "winning"
-- element of h (using softmax) is returned.
uniformAttn :: ∀ d m e batchSize. (KnownNat m) =>
               AttentionScoring batchSize e d ->
               T '[m,d,batchSize] Float32 -> T '[e,batchSize] Float32 -> Gen (T '[d,batchSize] Float32)
uniformAttn score hs_ ht = do
  xx <- mapT (score ht) hs_
  let   αt :: T '[m,batchSize] Float32
        αt = softmax0 xx
        ct :: T '[d,batchSize] Float32
        ct = squeeze0 (matmul hs_ (expandDim0 αt))
  return ct


-- | Add some attention, but feed back the attention vector back to
-- the next iteration in the rnn. (This follows the diagram at
-- https://github.com/tensorflow/nmt#background-on-the-attention-mechanism
-- commit 75aa22dfb159f10a1a5b4557777d9ff547c1975a).  The main
-- difference with 'addAttention' above is that the attn function is
-- that the final result depends on the attention vector rather than the output of the underlying cell.
addAttentionWithFeedback ::KnownShape s => 
                ((T (b ': s) t) -> Gen (T (x ': s) t)) ->
                RnnCell state                    (T ((a+x) ': s) t) (T (b ': s) t) ->
                RnnCell (T (x ': s) t ': state)   (T ( a    ': s) t) (T (x ': s) t)
addAttentionWithFeedback attn cell ((I prevAttnVector :* s),a) = do
  (s',y) <- cell (s,concat0 a prevAttnVector)
  focus <- attn y
  return ((I focus :* s'),focus)

-- | Luong attention model (following
-- https://github.com/tensorflow/nmt#background-on-the-attention-mechanism
-- commit 75aa22dfb159f10a1a5b4557777d9ff547c1975a)
luongAttention :: ∀ x d m e batchSize. (KnownNat m, KnownNat batchSize) =>
               AttentionScoring batchSize e d ->
               Tensor '[d+e,x] Float32 ->
               T '[m,d,batchSize] Float32 -> T '[e,batchSize] Float32 -> Gen (T '[x,batchSize] Float32)
luongAttention score w hs_ ht = do
  ct <- uniformAttn score hs_ ht
  let at = tanh (w ∙ concat0 ct ht)
  return at

type AttentionScoring batchSize e d = Tensor '[e,batchSize] Float32 -> Tensor '[d,batchSize] Float32 -> Tensor '[batchSize] Float32

luongMultiplicativeScoring :: forall e d batchSize. T [e,d] Float32 ->  AttentionScoring batchSize e d
luongMultiplicativeScoring w ht hs = hs · ir
  where ir :: T '[d,batchSize] Float32
        ir = w ∙ ht

-- luongWrapper score w1 w2 θ hs = addAttentionWithFeedback (luongAttention (luongMultiplicativeScoring w1) w2 hs) (lstm θ)
-- attnExample' θ1 θ2 h  = addAttention (attnExample1 θ1 h . snd) (lstm θ2)


-- | A layer in an rnn. @n@ is the length of the time sequence. @state@ is state propagated through time.
type RnnLayer n state input t output u = HList state -> Tensor (n ': input) t -> Gen (HList state , Tensor (n ': output) u)

-- | Build a RNN by repeating a cell @n@ times.
rnn :: ∀ n state input output t u.
       (KnownNat n, KnownShape input, KnownShape output) =>
       RnnCell state (T input t) (T output u) -> RnnLayer n state input t output u
rnn cell s0 t = do
  xs <- unstack t
  (sFin,us) <- chainForward cell (s0,xs)
  return (sFin,stack us)

-- | Build a RNN by repeating a cell @n@ times. However the state is
-- propagated in the right-to-left direction (decreasing indices in
-- the time dimension of the input and output tensors)
rnnBackwards :: ∀ n state input output t u.
       (KnownNat n, KnownShape input, KnownShape output) =>
       RnnCell state (T input t) (T output u) -> RnnLayer n state input t output u

rnnBackwards cell s0 t = do
  xs <- unstack t
  (sFin,us) <- chainBackward cell (s0,xs)
  return (sFin,stack us)

-- | Compose two rnn layers. This is useful for example to combine
-- forward and backward layers.
(.--.),stackRnnLayers :: forall s1 s2 a t b u c v n. KnownLen s1 =>
                  RnnLayer n s1 a t b u -> RnnLayer n s2 b u c v -> RnnLayer n (s1 ++ s2) a t c v
stackRnnLayers f g (hsplit @s1 -> (s0,s1)) x = do
  (s0',y) <- f s0 x
  (s1',z) <- g s1 y
  return (happ s0' s1',z)


infixr .--.
(.--.) = stackRnnLayers

-- | RNN helper
chainForward :: ∀ state a b n. ((state , a) -> Gen (state , b)) → (state , V n a) -> Gen (state , V n b)
chainForward _ (s0 , V []) = return (s0 , V [])
chainForward f (s0 , V (x:xs)) = do
  (s1,x') <- f (s0 , x)
  (sFin,V xs') <- chainForward f (s1 , V xs)
  return (sFin,V (x':xs'))
-- TODO: attempt to do the above with
-- tf.foldl

chainBackward :: ∀ state a b n. ((state , a) -> Gen (state , b)) → (state , V n a) -> Gen (state , V n b)
chainBackward _ (s0 , V []) = return (s0 , V [])
chainBackward f (s0 , V (x:xs)) = do
  (s1,V xs') <- chainBackward f (s0,V xs)
  (sFin, x') <- f (s1,x)
  return (sFin,V (x':xs'))


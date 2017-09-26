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

module TypedFlow.Layers where

import Prelude hiding (tanh,Num(..),Floating(..),floor)
import qualified Prelude
import GHC.TypeLits
-- import Text.PrettyPrint.Compact (float)
import TypedFlow.TF
import TypedFlow.Types
import Control.Monad.State (gets)
-- import Data.Type.Equality
-- import Data.Kind (Type,Constraint)
import Data.Monoid ((<>))
---------------------
-- Linear functions


-- A linear function form a to b is a matrix and a bias.
type (a ⊸ b) = DenseP Float32 a b

data DenseP t a b = DenseP {denseWeights :: Tensor '[a,b] t
                           ,denseBiases  :: Tensor '[b] t}

-----------------------
-- Feed-forward layers

-- | Parameters for the embedding layers
newtype EmbbeddingP numObjects embeddingSize t = EmbbeddingP (Tensor '[numObjects, embeddingSize] ('Typ 'Float t))

instance (KnownNat numObjects, KnownBits b, KnownNat embeddingSize) => KnownTensors (EmbbeddingP numObjects embeddingSize b) where
  travTensor f s (EmbbeddingP p) = EmbbeddingP <$> travTensor f s p

instance (KnownNat numObjects, KnownBits b, KnownNat embeddingSize) => ParamWithDefault (EmbbeddingP numObjects embeddingSize b) where
  defaultInitializer = EmbbeddingP (randomUniform (-0.05) 0.05)

-- | embedding layer
embedding :: ∀ embeddingSize numObjects batchSize t.
             EmbbeddingP numObjects embeddingSize t -> Tensor '[batchSize] Int32 -> Tensor '[embeddingSize,batchSize] ('Typ 'Float t)
embedding (EmbbeddingP param) input = gather @ '[embeddingSize] (transpose param) input

instance (KnownNat a, KnownNat b, KnownTyp t) => KnownTensors (DenseP t a b) where
  travTensor f s (DenseP x y) = DenseP <$> travTensor f (s<>"_w") x <*> travTensor f (s<>"_bias") y

instance (KnownNat n, KnownNat m) => ParamWithDefault (n ⊸ m) where
  defaultInitializer = DenseP glorotUniform (truncatedNormal 0.1)

-- | Apply a linear function
(#) :: (a ⊸ b) -> T '[a,batchSize] Float32 -> Tensor '[b,batchSize] Float32
(DenseP weightMatrix bias) # v = (weightMatrix ∙ v) + bias

-- | Dense layer
dense :: ∀m n batchSize. (n ⊸ m) -> Tensor '[n, batchSize] Float32 -> (Tensor '[m, batchSize] Float32)
dense lf t = lf # t

-- | A drop probability. (This type is used to make sure one does not
-- confuse keep probability and drop probability)
data DropProb = DropProb Float

-- | Generate a dropout function. The mask applied by the returned
-- function will be constant for any given call to mkDropout. This
-- behavior allows to use the same mask in the several steps of an
-- RNN.
mkDropout :: forall s t. KnownShape s => KnownBits t => DropProb -> Gen (Tensor s ('Typ 'Float t) -> Tensor s ('Typ 'Float t))
mkDropout (DropProb dropProb) = do
  let keepProb = 1.0 Prelude.- dropProb
  isTraining <- gets genTrainingPlaceholder
  mask <- assign (if_ isTraining
                   (floor (randomUniform keepProb (1 Prelude.+ keepProb)) ⊘ constant keepProb)
                   ones)
  return (mask ⊙)

newtype EndoTensor t s = EndoTensor (Tensor s t -> Tensor s t)

-- | Generate a dropout function for an heterogeneous tensor vector.
mkDropouts :: KnownBits t => KnownLen shapes => All KnownShape shapes => DropProb -> Gen (HTV ('Typ 'Float t) shapes -> HTV ('Typ 'Float t) shapes)
mkDropouts d = appEndoTensor <$> mkDropouts' shapeSList where
   mkDropouts' :: forall shapes t. KnownBits t => All KnownShape shapes =>
                  SList shapes -> Gen (NP (EndoTensor ('Typ 'Float t)) shapes)
   mkDropouts' LZ = return Unit
   mkDropouts' (LS _ rest) = do
     x <- mkDropout d
     xs <- mkDropouts' rest
     return (EndoTensor x :* xs)

   appEndoTensor :: NP (EndoTensor t) s -> HTV t s -> HTV t s
   appEndoTensor Unit Unit = Unit
   appEndoTensor (EndoTensor f :* fs) (F x :* xs) = F (f x) :* appEndoTensor fs xs


------------------------
-- Convolutional layers

data ConvP t outChannels inChannels filterSpatialShape
  = ConvP (T ('[outChannels,inChannels] ++ filterSpatialShape)  ('Typ 'Float t)) (T '[outChannels] ('Typ 'Float t))

instance (KnownNat outChannels,KnownNat inChannels, KnownShape filterSpatialShape, KnownBits t) =>
  ParamWithDefault (ConvP t outChannels inChannels filterSpatialShape) where
  defaultInitializer = ConvP (truncatedNormal 0.1) (constant 0.1)

instance (KnownNat outChannels,KnownNat inChannels, KnownShape filterSpatialShape, KnownBits t) =>
  KnownTensors (ConvP t outChannels inChannels filterSpatialShape) where
  travTensor f s (ConvP x y) = ConvP <$> travTensor f (s<>"_filters") x <*> travTensor f (s <> "_biases") y

-- | Size-preserving convolution layer
conv :: forall outChannels filterSpatialShape inChannels s t.
                  ((1 + Length filterSpatialShape) ~ Length s,
                   Length filterSpatialShape <= 3,
                   KnownLen filterSpatialShape) => -- the last dim of s is the batch size
                  ConvP t outChannels inChannels filterSpatialShape ->
                  T ('[inChannels] ++ s) ('Typ 'Float t) -> (T ('[outChannels] ++ s) ('Typ 'Float t))
conv (ConvP filters bias) input = convolution input filters + bias


-- | 2 by 2 maxpool layer.
maxPool2D :: forall stridex (stridey::Nat) batch height width channels.
             (KnownNat stridex, KnownNat stridey) =>
             T '[channels,width*stridex,height*stridex,batch] Float32 -> T '[channels,width,height,batch] Float32
maxPool2D (T value) = T (funcall "tf.nn.max_pool" [value
                                                  ,showShape @'[1,stridex,stridey,1]
                                                  ,showShape @'[1,stridex,stridey,1]
                                                  ,named "padding" (str "SAME") ])

-------------------------------
-- RNN layers and combinators



-- | Convert a pure function (feed-forward layer) to an RNN cell by
-- ignoring the RNN state.
timeDistribute :: (a -> b) -> RnnCell '[] a b
timeDistribute pureLayer = timeDistribute' (return . pureLayer)

-- | Convert a stateless generator into an RNN cell by ignoring the
-- RNN state.
timeDistribute' :: (a -> Gen b) -> RnnCell '[] a b
timeDistribute' stateLess (Unit,a) = do
  b <- stateLess a
  return (Unit,b)

-- | Standard RNN gate initializer. (The recurrent kernel is
-- orthogonal to avoid divergence; the input kernel is glorot)
cellInitializerBit :: ∀ n x. (KnownNat n, KnownNat x) => DenseP Float32 (n + x) n
cellInitializerBit = DenseP (concat0 recurrentInitializer kernelInitializer) biasInitializer
  where
        recurrentInitializer :: Tensor '[n, n] Float32
        recurrentInitializer = randomOrthogonal
        kernelInitializer :: Tensor '[x, n] Float32
        kernelInitializer = glorotUniform
        biasInitializer = zeros

-- | Parameter for an LSTM
data LSTMP n x = LSTMP ((n + x) ⊸ n) ((n + x) ⊸ n) ((n + x) ⊸ n) ((n + x) ⊸ n)

instance (KnownNat n, KnownNat x) => KnownTensors (LSTMP n x) where
  travTensor f s (LSTMP x y z w) = LSTMP <$> travTensor f (s<>"_f") x <*> travTensor f (s<>"_i") y <*> travTensor f (s<>"_c") z <*> travTensor f (s<>"_o") w
instance (KnownNat n, KnownNat x) => ParamWithDefault (LSTMP n x) where
  defaultInitializer = LSTMP forgetInit cellInitializerBit cellInitializerBit cellInitializerBit
    where forgetInit = DenseP (denseWeights cellInitializerBit) ones

-- | Standard LSTM
lstm :: ∀ n x bs. (KnownNat bs) => LSTMP n x ->
        RnnCell '[ '[n,bs], '[n,bs]] (Tensor '[x,bs] Float32) (Tensor '[n,bs] Float32)
lstm (LSTMP wf wi wc wo) (VecPair ht1 ct1, input) = do
  hx <- assign (concat0 ht1 input)
  let f = sigmoid (wf # hx)
      i = sigmoid (wi # hx)
      cTilda = tanh (wc # hx)
      o = sigmoid (wo # hx)
  c <- assign ((f ⊙ ct1) + (i ⊙ cTilda))
  h <- assign (o ⊙ tanh c)
  return (VecPair h c, h)

-- | LSTM for an attention model. Takes an attention function.
attentiveLstm :: forall x n bs. KnownNat bs =>
  AttentionFunction bs n n ->
  LSTMP n x ->
  RnnCell '[ '[n,bs], '[n,bs]] (Tensor '[x,bs] Float32) (Tensor '[n,bs] Float32)
attentiveLstm att w x = do
  (VecPair ht ct, _ht) <- lstm w x
  a <- att ht
  let ht' = ht ⊕ a -- alternatively add a dense layer to combine
  return (VecPair ht' ct, a)

-- | Parameter for a GRU
data GRUP n x = GRUP ((n + x) ⊸ n)  ((n + x) ⊸ n)  ((n + x) ⊸ n)

instance (KnownNat n, KnownNat x) => KnownTensors (GRUP n x) where
  travTensor f s (GRUP x y z) = GRUP <$> travTensor f (s<>"_z") x <*> travTensor f (s<>"_r") y <*> travTensor f (s<>"_w") z
instance (KnownNat n, KnownNat x) => ParamWithDefault (GRUP n x) where
  defaultInitializer = GRUP cellInitializerBit cellInitializerBit cellInitializerBit


-- | Standard GRU cell
gru :: ∀ n x bs. (KnownNat bs, KnownNat n) => GRUP n x ->
        RnnCell '[ '[n,bs] ] (Tensor '[x,bs] Float32) (Tensor '[n,bs] Float32)
gru (GRUP wz wr w) (VecSing ht1, xt) = do
  hx <- assign (concat0 ht1 xt)
  let zt = sigmoid (wz # hx)
      rt = sigmoid (wr # hx)
      hTilda = tanh (w # (concat0 (rt ⊙ ht1) xt))
  ht <- assign ((ones ⊝ zt) ⊙ ht1 + zt ⊙ hTilda)
  return (VecSing ht, ht)

-- | Apply a function on the cell state(s) before running the cell itself.
onStates ::  (HTV Float32 xs -> HTV Float32 xs) -> RnnCell xs a b -> RnnCell xs a b
onStates f cell (h,x) = do
  cell (f h, x)

-- | Stack two RNN cells (LHS is run first)
stackRnnCells, (.-.) :: forall s0 s1 a b c. KnownLen s0 => RnnCell s0 a b -> RnnCell s1 b c -> RnnCell (s0 ++ s1) a c
stackRnnCells l1 l2 (hsplit @s0 -> (s0,s1),x) = do
  (s0',y) <- l1 (s0,x)
  (s1',z) <- l2 (s1,y)
  return ((happ s0' s1'),z)

(.-.) = stackRnnCells

-- | Run the cell, and forward the input to the output, by concatenation with the output of the cell.
withBypass :: KnownNat bs => RnnCell s0 (T '[x,bs] t) (T '[y,bs] t) -> RnnCell s0 (T '[x,bs] t) (T '[x+y,bs] t)
withBypass cell (s,x) = do
  (s',y) <- cell (s,x)
  return (s',concat0 x y)

-- | An attention scoring function. We assume that each portion of the
-- input has size @e@. @d@ is typically the size of the current state
-- of the output RNN.
type AttentionScoring t batchSize keySize valueSize =
  Tensor '[keySize,batchSize] ('Typ 'Float t) -> Tensor '[valueSize,batchSize] ('Typ 'Float t) -> Tensor '[batchSize] ('Typ 'Float t)

type AttentionScoring' t batchSize keySize valueSize nValues = 
  Tensor '[keySize,batchSize] ('Typ 'Float t) -> Tensor '[nValues,valueSize,batchSize] ('Typ 'Float t) -> Tensor '[nValues,batchSize] ('Typ 'Float t)

type AttentionFunction batchSize keySize valueSize =
  T '[keySize,batchSize] Float32 -> Gen (T '[valueSize,batchSize] Float32)

-- | @attnExample1 θ h st@ combines each element of the vector h with
-- s, and applies a dense layer with parameters θ. The "winning"
-- element of h (using softmax) is returned.
uniformAttn :: ∀ valueSize m keySize batchSize. KnownNat m => 
               T '[batchSize] Int32 ->
               AttentionScoring 'B32 batchSize keySize valueSize ->
               T '[m,valueSize,batchSize] Float32 -> AttentionFunction batchSize keySize valueSize
uniformAttn lengths score hs_ ht = do
  xx <- mapT (score ht) hs_
  let   αt :: T '[m,batchSize] Float32
        αt = softmax0 (mask ⊙ xx)
        ct :: T '[valueSize,batchSize] Float32
        ct = squeeze0 (matmul hs_ (expandDim0 αt))
        mask = cast (sequenceMask @m lengths) -- mask according to length
  return ct


-- | @attnExample1 θ h st@ combines each element of the vector h with
-- s, and applies a dense layer with parameters θ. The "winning"
-- element of h (using softmax) is returned.
uniformAttn' :: ∀ valueSize m keySize batchSize. KnownNat m => 
               T '[batchSize] Int32 ->
               AttentionScoring' 'B32 batchSize keySize valueSize m ->
               T '[m,valueSize,batchSize] Float32 -> AttentionFunction batchSize keySize valueSize
uniformAttn' lengths score hs_ ht = do
  let   αt :: T '[m,batchSize] Float32
        xx = score ht hs_
        αt = softmax0 (mask ⊙ xx)
        ct :: T '[valueSize,batchSize] Float32
        ct = squeeze0 (matmul hs_ (expandDim0 αt))
        mask = cast (sequenceMask @m lengths) -- mask according to length
  return ct


-- -- | Add some attention, but feed back the attention vector back to
-- -- the next iteration in the rnn. (This follows the diagram at
-- -- https://github.com/tensorflow/nmt#background-on-the-attention-mechanism
-- -- commit 75aa22dfb159f10a1a5b4557777d9ff547c1975a).  The main
-- -- difference with 'addAttention' above is that the attn function is
-- -- that the final result depends on the attention vector rather than the output of the underlying cell.
-- -- (This yields to exploding loss in my tests.)
-- addAttentionWithFeedback ::KnownShape s => 
--                 ((T (b ': s) t) -> Gen (T (x ': s) t)) ->
--                 RnnCell state                    (T ((a+x) ': s) t) (T (b ': s) t) ->
--                 RnnCell (T (x ': s) t ': state)  (T ( a    ': s) t) (T (x ': s) t)
-- addAttentionWithFeedback attn cell ((I prevAttnVector :* s),a) = do
--   (s',y) <- cell (s,concat0 a prevAttnVector)
--   focus <- attn y
--   return ((I focus :* s'),focus)

-- -- | @addAttention attn cell@ adds the attention function @attn@ to the
-- -- rnn cell @cell@.  Note that @attn@ can depend in particular on a
-- -- constant external value @h@ which is the complete input to pay
-- -- attention to.
-- addAttentionAbove :: KnownShape s =>
--                 ((T (b ': s) t) -> Gen (T (x ': s) t)) ->
--                 RnnCell states (T (a ': s) t) (T (b ': s) t) ->
--                 RnnCell states (T (a ': s) t) (T (b+x ': s) t)
-- addAttentionAbove attn cell (s,a) = do
--   (s',y) <- cell (s,a)
--   focus <- attn y
--   return (s',concat0 y focus)


-- addAttentionBelow ::KnownShape s => (t ~ Float32) =>
--                 ((T (b ': s) t) -> Gen (T (x ': s) t)) ->
--                 RnnCell state                    (T ((a+x) ': s) t) (T (b ': s) t) ->
--                 RnnCell ((b ': s) ': state)  (T ( a    ': s) t) (T (b ': s) t)
-- addAttentionBelow attn cell ((F prevY :* s),a) = do
--   focus <- attn prevY
--   (s',y) <- cell (s,concat0 a focus)
--   return ((F y :* s'),y)

 


-- | Luong attention model (following
-- https://github.com/tensorflow/nmt#background-on-the-attention-mechanism
-- commit 75aa22dfb159f10a1a5b4557777d9ff547c1975a)
luongAttention :: ∀ attnSize d m e batchSize. KnownNat m => ( KnownNat batchSize) =>
               Tensor '[batchSize] Int32 ->
               AttentionScoring 'B32 batchSize e d ->
               Tensor '[d+e,attnSize] Float32 ->
               T '[m,d,batchSize] Float32 -> T '[e,batchSize] Float32 -> Gen (T '[attnSize,batchSize] Float32)
luongAttention lens score w hs_ ht = do
  ct <- uniformAttn lens score hs_ ht
  return (tanh (w ∙ (concat0 ct ht)))
-- This is essentially a dense layer on top of uniform attention; consider removing it.


-- | A multiplicative scoring function. See 
-- https://github.com/tensorflow/nmt#background-on-the-attention-mechanism
-- commit 75aa22dfb159f10a1a5b4557777d9ff547c1975a
multiplicativeScoring :: forall valueSize keySize batchSize t.
  T [keySize,valueSize] ('Typ 'Float t) ->  AttentionScoring t batchSize keySize valueSize
multiplicativeScoring w dt h = h · ir
  where ir :: T '[valueSize,batchSize] ('Typ 'Float t)
        ir = w ∙ dt

multiplicativeScoring' :: forall valueSize keySize batchSize nValues t.
  KnownNat batchSize => T [keySize,valueSize] ('Typ 'Float t) ->  AttentionScoring' t batchSize keySize valueSize nValues 
multiplicativeScoring' w dt hs = squeeze1 (matmul (expandDim1 ir) hs)
  where ir :: T '[valueSize,batchSize] ('Typ 'Float t)
        ir = w ∙ dt


-- | An additive scoring function. See https://arxiv.org/pdf/1412.7449.pdf
data AdditiveScoringP sz keySize valueSize t = AdditiveScoringP
  (Tensor '[sz, 1]         ('Typ 'Float t))
  (Tensor '[keySize, sz]   ('Typ 'Float t))
  (Tensor '[valueSize, sz] ('Typ 'Float t))

instance (KnownNat n, KnownNat k, KnownNat v, KnownBits t) => KnownTensors (AdditiveScoringP k v n t) where
  travTensor f s (AdditiveScoringP x y z) = AdditiveScoringP <$> travTensor f (s<>"_v") x <*> travTensor f (s<>"_w1") y <*> travTensor f (s<>"_w2") z
instance (KnownNat n, KnownNat k, KnownNat v, KnownBits t) => ParamWithDefault (AdditiveScoringP k v n t) where
  defaultInitializer = AdditiveScoringP glorotUniform glorotUniform glorotUniform

additiveScoring :: AdditiveScoringP sz keySize valueSize t -> AttentionScoring t batchSize valueSize keySize
additiveScoring (AdditiveScoringP v w1 w2) dt h = squeeze0 (v ∙ tanh ((w1 ∙ h) ⊕ (w2 ∙ dt)))

additiveScoring' :: forall sz keySize valueSize t nValues batchSize. KnownNat sz => KnownNat keySize => (KnownNat nValues, KnownNat batchSize) =>
  AdditiveScoringP sz keySize valueSize t -> AttentionScoring' t batchSize valueSize keySize nValues
additiveScoring' (AdditiveScoringP v w1 w2) dt h = transpose r''
  where w1h :: Tensor '[sz,batchSize, nValues] ('Typ 'Float t)
        w1h = transposeN01 @'[sz] (reshape @'[sz,nValues, batchSize] w1h')
        w1h' = matmul (reshape @'[keySize, nValues*batchSize] (transpose01 h)) (transpose01 w1)
        w2dt = w2 ∙ dt
        z' = reshape @'[sz,batchSize*nValues] (tanh (w1h + w2dt))
        r'' = reshape @[batchSize,nValues] (matmul z' (transpose v))

-- | A cell in an rnn. @state@ is the state propagated through time.
type RnnCell states input output = (HTV Float32 states , input) -> Gen (HTV Float32 states , output)

-- | A layer in an rnn. @n@ is the length of the time sequence. @state@ is the state propagated through time.
type RnnLayer n state input t output u = FHTV state -> Tensor (n ': input) t -> Gen (FHTV state , Tensor (n ': output) u)

-- | Build a RNN by repeating a cell @n@ times.
rnn :: ∀ n state input output t u.
       (KnownNat n, KnownShape input, KnownShape output) =>
       RnnCell state (T input t) (T output u) -> RnnLayer n state input t output u
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
rnnBackward :: ∀ n state input output t u.
       (KnownNat n, KnownShape input, KnownShape output) =>
       RnnCell state (T input t) (T output u) -> RnnLayer n state input t output u

rnnBackward cell s0 t = do
  xs <- unstack t
  (sFin,us) <- chainBackward cell (s0,xs)
  return (sFin,stack us)

-- note: an alternative design would be to reverse the input and
-- output tensors instead. (thus we could make 'backwards' a
-- completely separate function that does not do any chaining.)
-- Reversing the tensors in TF may be slow though, so we should change
-- the in/out from tensors to vectors of tensors.

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


-- | Compose two rnn layers in parallel.
bothRnnLayers,(.++.)  :: forall s1 s2 a t b u c n bs. KnownNat bs => KnownLen s1 =>
                  RnnLayer n s1 a t '[b,bs] u -> RnnLayer n s2 a t '[c,bs] u -> RnnLayer n (s1 ++ s2) a t '[b+c,bs] u
bothRnnLayers f g (hsplit @s1 -> (s0,s1)) x = do
  (s0',y) <- f s0 x
  (s1',z) <- g s1 x
  return (happ s0' s1',concat1 y z)


infixr .++.
(.++.) = bothRnnLayers


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

-- | RNN helper
chainBackwardWithState ::
  ∀ state a b n. ((state , a) -> Gen (state , b)) → (state , V n a) -> Gen (state , V n b, V n state)
chainBackwardWithState _ (s0 , V []) = return (s0 , V [], V [])
chainBackwardWithState f (s0 , V (x:xs)) = do
  (s1,V xs',V ss') <- chainBackwardWithState f (s0,V xs)
  (sFin, x') <- f (s1,x)
  return (sFin,V (x':xs'),V (sFin:ss'))

-- | RNN helper
transposeV :: forall n xs. All KnownLen xs =>
               SList xs -> V n (HTV Float32 xs) -> HTV Float32 (Ap (FMap (Cons n)) xs)
transposeV LZ _ = Unit
transposeV (LS _ n) xxs  = F ys' :* yys'
  where (ys,yys) = help @(Tail xs) xxs
        ys' = stack ys
        yys' = transposeV n yys

        help :: forall ys x t. V n (HTV t (x ': ys)) -> (V n (T x t) , V n (HTV t ys))
        help (V xs) = (V (map (fromF . hhead) xs),V (map htail xs))

-- | @(gatherFinalStates dynLen states)[i] = states[dynLen[i]]@
gatherFinalStates :: KnownLen x => KnownNat n => LastEqual bs x => T '[bs] Int32 -> T (n ': x) t -> T x t
gatherFinalStates dynLen states = nth0 0 (reverseSequences dynLen states)

-- a more efficient algorithm (perhaps:)
-- gatherFinalStates' :: forall x n bs t. KnownLen x => KnownNat n => LastEqual bs x => T '[bs] Int32 -> T (x ++ '[n,bs]) t -> T x (x ++ '[bs])
-- gatherFinalStates' (T dynLen)t = gather (flattenN2 @x @n @bs t) indexInFlat
--  where indexInFlat = (dynLen - 1) + tf.range(0, bs) * n

gathers :: forall n bs xs. All (LastEqual bs) xs => All KnownLen xs => KnownNat n =>
            SList xs -> T '[bs] Int32 -> FHTV (Ap (FMap (Cons n)) xs) -> FHTV xs
gathers LZ _ Unit = Unit
gathers (LS _ n) ixs (F x :* xs) = F (gatherFinalStates ixs x) :* gathers @n n ixs xs

-- | @rnnWithCull dynLen@ constructs an RNN as normal, but returns the
-- state after step @dynLen@ only.
rnnWithCull :: forall n bs x y t u ls.
  KnownLen ls => KnownNat n => KnownLen x  => KnownLen y => All KnownLen ls =>
  All (LastEqual bs) ls =>
  T '[bs] Int32 -> RnnCell ls (T x t) (T y u) -> RnnLayer n ls x t y u
rnnWithCull dynLen cell s0 t = do
  xs <- unstack t
  (us,ss) <- chainForwardWithState cell (s0,xs)
  let sss = transposeV @n (shapeSList @ls) ss
  return (gathers @n (shapeSList @ls) dynLen sss,stack us)

-- | Like @rnnWithCull@, but states are threaded backwards.
rnnBackwardsWithCull :: forall n bs x y t u ls.
  KnownLen ls => KnownNat n => KnownLen x  => KnownLen y => All KnownLen ls =>
  All (LastEqual bs) ls => LastEqual bs x => LastEqual bs y =>
  T '[bs] Int32 -> RnnCell ls (T x t) (T y u) -> RnnLayer n ls x t y u
rnnBackwardsWithCull dynLen cell s0 t = do
  (sFin,hs) <- rnnWithCull dynLen cell s0 (reverseSequences dynLen t)
  hs' <- assign (reverseSequences dynLen hs)
  return (sFin, hs')


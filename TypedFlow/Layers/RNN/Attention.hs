{-|
Module      : TypedFlow.Layers.RNN.Attention
Description : Attention combinators to be used with RNN cells
Copyright   : (c) Jean-Philippe Bernardy, 2018
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

module TypedFlow.Layers.RNN.Attention (
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
  ) where

import Prelude hiding (tanh,Num(..),Floating(..),floor)
import GHC.TypeLits
import TypedFlow.TF
import TypedFlow.Types
import Data.Monoid ((<>))
import TypedFlow.Layers.RNN.Base

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
attentiveWithFeedback ::forall attSize cellSize inputSize bs w ss. KnownLen ss =>
  AttentionFunction w bs cellSize attSize ->
  RnnCell w ss                      (T '[inputSize+attSize,bs] (Flt w)) (T '[cellSize,bs] (Flt w)) ->
  RnnCell w ('[attSize,bs] ': ss)   (T '[inputSize        ,bs] (Flt w)) (T '[attSize,bs] (Flt w))
attentiveWithFeedback attn cell = appEmpty @ss $ withFeedback (cell .-. timeDistribute' attn)


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
  KnownNat batchSize => T [keySize,valueSize] ('Typ 'Float t) -- ^ weights
  ->  AttentionScoring t batchSize keySize valueSize nValues
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


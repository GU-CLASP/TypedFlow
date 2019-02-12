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

import Prelude hiding (RealFrac(..))
import GHC.TypeLits
import TypedFlow.TF
import TypedFlow.Types
import TypedFlow.Types.Proofs (appRUnit,(#>))
import Data.Monoid ((<>))
import TypedFlow.Layers.RNN.Base

-- | An attention scoring function. This function should produce a
-- score (between 0 and 1).
type AttentionScoring t keySize valueSize = 
  Tensor '[keySize] ('Typ 'Float t) -> Tensor '[valueSize] ('Typ 'Float t) -> Tensor '[] ('Typ 'Float t)

-- | A function which attends to an external input. Typically a
-- function of this type is a closure which has the attended input in
-- its environment. This environment is interpreted as an associative
-- memory form key to value.
type AttentionFunction t keySize valueSize =
  T '[keySize] (Flt t) -> T '[valueSize] (Flt t)

-- | @attnExample1 θ h st@ combines each element of the vector h with
-- s, and applies a dense layer with parameters θ. The "winning"
-- element of h (using softmax) is returned.
uniformAttn :: ∀ valueSize m keySize t. KnownNat valueSize => KnownNat m => KnownBits t
       => AttentionScoring t keySize valueSize -- ^ scoring function
       -> T '[] Int32 -- ^ length of the input
       -> T '[m,valueSize] (Flt t) -- ^ input (what we're attending to)
       -> AttentionFunction t keySize valueSize
uniformAttn score len hs key = c
  where xx,α :: T '[m] (Flt t)
        xx = mapT (score key) hs
        α = softmax0 (mask ⊙ xx)
        c :: T '[valueSize] (Flt t)
        c = hs ∙ α
        mask = cast (sequenceMask @m len) -- mask according to length

-- | Add some attention to an RnnCell, and feed the attention vector to
-- the next iteration in the rnn. (This follows the diagram at
-- https://github.com/tensorflow/nmt#background-on-the-attention-mechanism
-- commit 75aa22dfb159f10a1a5b4557777d9ff547c1975a).
attentiveWithFeedback ::forall attSize cellSize inputSize w ss. KnownNat inputSize => KnownNat attSize => KnownLen ss =>
  KnownBits w =>
  AttentionFunction w cellSize attSize ->
  RnnCell w ss                   (T '[inputSize+attSize] (Flt w)) (T '[cellSize] (Flt w)) ->
  RnnCell w ('[attSize] ': ss)   (T '[inputSize        ] (Flt w)) (T '[attSize] (Flt w))
attentiveWithFeedback attn cell = appRUnit @ss #> withFeedback (cell .-. timeDistribute attn)


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
luongAttention :: ∀ attnSize d m e w. KnownNat e => KnownNat d => KnownNat attnSize => KnownNat m => KnownBits w
  => Tensor '[d+e,attnSize] (Flt w)     -- ^ weights for the dense layer
  -> AttentionScoring w e d -- ^ scoring function
  -> Tensor '[] Int32          -- ^ length of the input
  -> T '[m,d] (Flt w)         -- ^ inputs
  -> AttentionFunction w e attnSize
luongAttention w scoring lens hs_ ht = 
  let ct = uniformAttn scoring lens hs_ ht
  in (tanh (w ∙ (concat0 ct ht)))

-- | Multiplicative scoring function
multiplicativeScoring :: forall valueSize keySize t.
  KnownBits t => KnownNat valueSize => KnownNat keySize
  => T [keySize,valueSize] ('Typ 'Float t) -- ^ weights
  ->  AttentionScoring t keySize valueSize
multiplicativeScoring w dt h = ir · h
  where ir :: T '[valueSize] ('Typ 'Float t)
        ir = w ∙ dt


data AdditiveScoringP sz keySize valueSize t = AdditiveScoringP
  (Tensor '[1,sz]         ('Typ 'Float t))
  (Tensor '[keySize, sz]   ('Typ 'Float t))
  (Tensor '[valueSize, sz] ('Typ 'Float t))

instance (KnownNat n, KnownNat k, KnownNat v, KnownBits t) => KnownTensors (AdditiveScoringP k v n t) where
  travTensor f s (AdditiveScoringP x y z) = AdditiveScoringP <$> travTensor f (s<>"_v") x <*> travTensor f (s<>"_w1") y <*> travTensor f (s<>"_w2") z
instance (KnownNat n, KnownNat k, KnownNat v, KnownBits t) => ParamWithDefault (AdditiveScoringP k v n t) where
  defaultInitializer = AdditiveScoringP <$> glorotUniform <*> glorotUniform <*> glorotUniform

-- | An additive scoring function. See https://arxiv.org/pdf/1412.7449.pdf
additiveScoring :: forall sz keySize valueSize t. KnownNat valueSize => KnownNat sz => KnownNat keySize => KnownBits t =>
  AdditiveScoringP sz keySize valueSize t -> AttentionScoring t valueSize keySize
additiveScoring (AdditiveScoringP v w1 w2) dt h =  r''
  where w1h :: Tensor '[sz] ('Typ 'Float t)
        w1h = w1 ∙ h
        w2dt = w2 ∙ dt
        z' :: Tensor '[sz] ('Typ 'Float t)
        z' = tanh (w1h + w2dt)
        r'' = z' · squeeze0 v


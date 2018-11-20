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
{-|
Module      : TypedFlow.Models.Topic
Description : Topic models
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental
-}


module TypedFlow.Models.Topic where
import Prelude hiding (RealFrac(..))
import TypedFlow.TF
import TypedFlow.Layers
import TypedFlow.Types
import TypedFlow.Learn
import GHC.TypeLits
import Data.Monoid ((<>))

-- | A convolutional document summary function. Described in
-- 'Topically Driven Neural Language Model' by Lau, Baldwin and Cohn.
tldmDocsummary :: forall
  (vocSize :: Nat) -- number of words
  (e :: Nat) -- size of the embedding
  (a :: Nat) -- number of features of the document vector summary 
  (n :: Nat) -- length of the document
  (filterSize :: Nat) -- size of the convolution filter
  (t :: NBits) -- size of floats
  .  KnownNat vocSize => KnownNat filterSize => KnownNat e => KnownNat a => KnownNat n => KnownBits t
  => (EmbeddingP vocSize e t)
  -> (ConvP t a e '[filterSize])
  -> DropProb
  -> Gen (T '[n] Int32 -> T '[a] (Flt t))
tldmDocsummary embs filters dropProb = do
  drpEmb <- mkDropout dropProb
  return $ \document ->
    let embeddedDoc :: Tensor [n,e] (Flt t)
        embeddedDoc = mapT (drpEmb . embedding @e @vocSize embs) document
    in reduceMax axis0 (conv' @'[n] filters embeddedDoc)

-- | Parameter for topics. This is effectively map from document
-- features (a) to topic representations (vectors of size b) via k
-- topic distributions.
data TopicP t a k b = TopicP {topicDistributions :: (T '[a,k] (Flt t))  -- ^ a linear map from documents features (a) to topic distributions (k)
                             ,topicRepresentations :: (T '[k,b] (Flt t)) -- ^ a linear map from topic distributions (k) to topic representations (b)
                             }

instance (KnownNat a, KnownNat k, KnownNat b, KnownBits t) => KnownTensors (TopicP t a k b) where
  travTensor f s (TopicP x y) = TopicP <$> travTensor f (s<>"_A") x <*> travTensor f (s<>"_B") y
instance (KnownNat a, KnownNat k, KnownNat b, KnownBits t) => ParamWithDefault (TopicP t a k b) where
  defaultInitializer = TopicP <$> glorotUniform <*> glorotUniform

-- | A topic modeler. Described 'Topically Driven Neural Language
-- Model' by Lau, Baldwin and Cohn.  Returns a function converting raw
-- representations (eg. document summaries) to topic representations.
-- This representation can be used as input to a dense layer to
-- predict a word, or as input to an LSTM (initial state) to predict
-- sentences.
mkTdlmTopic :: forall
  (kk :: Nat) -- number of topics
  (a :: Nat) -- document vector summary size
  (b :: Nat) -- topic representation size
  (t :: NBits) -- size of floats
  . KnownNat kk => KnownNat a => KnownNat b => KnownBits t
  => Float -> TopicP t a kk b -> Gen (T '[a] (Flt t) -> Tensor '[b] (Flt t))
mkTdlmTopic separationConstant (TopicP topicInput topicOutput) = do
  drpS   <- mkDropout (DropProb 0.1)
  let topicNormalized :: T '[kk,b] (Flt t)
      topicNormalized = mapT normalize topicOutput
      -- matrix of correlation between the topics
      topicCorrelation :: T '[kk,kk] (Flt t)
      topicCorrelation = matmul topicNormalized (transpose01 topicNormalized)
      -- max correlation between two distinct topics
      topicOverlap = reduceMaxAll (square (topicCorrelation ⊝ eye))
  addRegularizer (constant separationConstant ⊙ cast topicOverlap) -- regularizer which ensures that topics are disjoint

  return (\d -> let p :: T '[kk] (Flt t)
                    p = softmax0 (topicInput ∙ d) -- attention distribution (among the topics)
                in drpS (topicOutput ∙ p))

-- | Gating unit which can be used to mix a RNN hidden state with an
-- external information source (eg. topic representation).  Described
-- 'Topically Driven Neural Language Model' by Lau, Baldwin and Cohn;
-- formula (3)
tldmGatingUnit :: KnownNat n => KnownBits t => KnownNat m => (GRUP t m n) -> T '[n] (Flt t) -> T '[m] (Flt t) -> (T '[m] (Flt t))
tldmGatingUnit (GRUP wz wr w) s h = 
  let x = concat0 h s
      z = sigmoid (wz ∙ x)
      r = sigmoid (wr ∙ x)
      hTilda = tanh (w ∙ (concat0 (r ⊙ h) s))
  in ((ones ⊝ z) ⊙ h + z ⊙ hTilda)

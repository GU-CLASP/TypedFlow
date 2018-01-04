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
import Prelude hiding ((/), sqrt)
import TypedFlow.TF
import TypedFlow.Layers
import TypedFlow.Types
import GHC.TypeLits

-- -- | create a document summarization function with appropriate parameters.
-- mkDocumentSummary
--   :: String -> -- ^ prefix for parameter names
--      Gen (T '[n,e] (Flt t) -> T '[a] (Flt t)) -- ^ document vector (summary)
-- mkDocumentSummary prefix = do
--   filter <- parameter (prefix ++ "_filter") (truncatedNormal 0.1 )
--   return $ (relu . conv filter)



-- p = softmax (A d)
-- s = B p

-- | An implementation of 'Topically Driven Neural Language Model' by
-- Lau, Baldwin and Cohn. This is the first part; the topic modelling
-- itself.
tdlmTopic :: forall
  (vocSize :: Nat) -- number of words
  (e :: Nat) -- size of the embedding
  (n :: Nat) -- length of the document
  (kk :: Nat) -- number of topics
  (a :: Nat) -- document vector summary size
  (b :: Nat) -- topic representation size
  (filterSize :: Nat) -- size of the convolution filter.
  (t :: NBits) -- size of floats
  (batchSize :: Nat)
  . KnownNat kk => KnownNat filterSize => KnownNat n => KnownNat a => KnownNat b => KnownNat e => KnownNat vocSize => KnownBits t => KnownNat batchSize
  => T '[n,batchSize] Int32 -- ^ document
  -> Gen (Tensor '[b, batchSize] (Flt t), Scalar (Flt t))
tdlmTopic inputDoc = do
  embs <- parameterDefault "embs"
  drpEmb <- mkDropout (DropProb 0.1)
  drpS   <- mkDropout (DropProb 0.1)
  filters <- parameterDefault "conv"
  topicInput :: T '[a,kk] (Flt t) <- parameter "A" glorotUniform -- mapping from document representations to topics
  topicOutput :: T '[kk,b] (Flt t) <- parameter "B" glorotUniform  -- all possible topics
  let docInputs = drpEmb (embedding @e @vocSize embs inputDoc)
      conv'd = conv @a @'[filterSize] filters docInputs -- in the code they do this for several filter sizes (ie. phrase sizes)
      max'd = reduceMax @Dim1 conv'd
      d :: T '[a,batchSize] (Flt t)
      d = max'd -- document summary
      p :: T '[kk,batchSize] (Flt t)
      p = softmax0 (topicInput ∙ d) -- attention distribution (among the topics)
      s :: T '[b,batchSize] (Flt t)
      s = drpS (topicOutput ∙ p)  -- document topic representation
      topicNormalized :: T '[b,kk] (Flt t)
      topicNormalized = transpose01 topicOutput / (sqrt (reduceSum @Dim0 (topicOutput ⊙ topicOutput)) :: T '[b] (Flt t))
      topicCorrelation :: T '[b,b] (Flt t)
      topicCorrelation = matmul (transpose01 topicNormalized) topicNormalized
      topicUniqueness = reduceMaxAll (topicCorrelation ⊝ eye)


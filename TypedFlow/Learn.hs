{-|
Module      : TypedFlow.Learn
Description : Loss functions and optimization strategies
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental
-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
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
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UnicodeSyntax #-}

module TypedFlow.Learn where

import TypedFlow.Types
import TypedFlow.TF
import qualified Prelude (Float)
import Prelude (($),return,Maybe(..),(=<<))
import Text.PrettyPrint.Compact (text)
import Data.Monoid hiding (Last)
import GHC.TypeLits
import Control.Monad.State (modify, gets)

-- | Triple of values that are always output in a model: prediction, loss and accuracy.
data ModelOutput s t = forall n.
                       ModelOutput {modelY :: T s t -- ^ prediction
                                   ,modelLoss :: Scalar Float32
                                   ,modelCorrect :: T '[n] Float32 -- ^ correctness of a bit of the output, within [0,1]
                                   }

-- | A standard modelling function: (input value, gold value) ↦ (prediction, accuracy, loss)
type Model input tIn output tOut = T input tIn -> T output tOut -> Gen (ModelOutput output tOut)

modelBoth :: forall n m s t. KnownLen s => ModelOutput (n ': s) t -> ModelOutput (m ': s) t -> ModelOutput (n + m ': s) t
modelBoth (ModelOutput y1 l1 c1) (ModelOutput y2 l2 c2) = ModelOutput (concat0 y1 y2) (l1 + l2) (concat0 c1 c2)

-- | First type argument is the number of classes.
-- @categorical logits gold@
-- return (prediction, accuraccy, loss)
-- accuracy and prediction are averaged over the batch.
categorical :: forall nCat bs. KnownNat nCat => Model '[nCat,bs] Float32 '[bs] Int32
categorical logits' y = do
  logits <- assign logits'
  let y_ = argmax0 logits
      modelY = y_
      modelCorrect = cast (equal y_ y)
      modelLoss = reduceMeanAll (sparseSoftmaxCrossEntropyWithLogits y logits)
  return ModelOutput{..}

-- | First type argument is the number of classes.
-- @categoricalDistribution logits gold@
-- return (prediction, accuraccy, loss)
-- accuracy and prediction are averaged over the batch.
categoricalDistribution :: forall nCat bs. Model '[nCat,bs] Float32 '[nCat,bs] Float32
categoricalDistribution logits' y = do
  logits <- assign logits'
  let y_ = softmax0 logits
      modelY = y_
      modelCorrect = cast (equal (argmax0 @'B32 logits) (argmax0 y))
      modelLoss = reduceMeanAll (softmaxCrossEntropyWithLogits y logits)
  return ModelOutput{..}

-- | @timedCategorical targetWeights logits y@
--
-- targetWeights: a zero-one matrix of the same size as
-- decoder_outputs. It is intended to mask padding positions outside
-- of the target sequence lengths with values 0.
--
-- Note that the accuracy is computed by weigthing the accuracies at
-- individual time steps with the targetWeights.

timedCategorical :: forall len nCat bs bits. KnownNat nCat => KnownNat bs => KnownNat len => KnownBits bits =>
  Tensor '[len,bs] (Flt bits) -> Tensor '[len,nCat,bs] (Flt bits) -> Tensor '[len,bs] Int32 -> Gen (ModelOutput '[len,nCat,bs] (Flt bits))
timedCategorical targetWeights logits' y = do
  logits <- assign logits'
  let y_ = argmax1 logits
      modelY = softmax1 logits
      correctPrediction = equal y_ y
      modelCorrect = cast ((reduceSum @Axis0 (cast @(Flt bits) correctPrediction ⊙ targetWeights)) ⊘ reduceSum @Axis0 targetWeights)
      crossEntropies = sparseSoftmaxCrossEntropyWithLogits y (transpose01 logits)
      modelLoss = cast @Float32 (reduceMeanAll (crossEntropies ⊙ targetWeights))
  return ModelOutput{..}

-- | Model with several binary outputs.
binary :: forall n bs. KnownNat n => (KnownNat bs) => Model '[n,bs] Float32 '[n,bs] Int32
binary logits y = do
  sigy_ <- assign (sigmoid logits)
  let y_ = cast @Int32 (round sigy_)
      modelY = y_
      modelCorrect = cast (flatten2 (equal y_ y))
      modelLoss = reduceMeanAll (sigmoidCrossEntropyWithLogits (cast @Float32 y) logits)
  return ModelOutput{..}

-- | Model compiler options
data Options = Options {maxGradientNorm :: Maybe Prelude.Float -- ^ apply gradient clipping
                       }

-- | default model compiler options
defaultOptions :: Options
defaultOptions = Options {maxGradientNorm = Nothing}

-- | compile a standard model
compile :: forall sx tx sy ty sy_ ty_.
           (KnownShape sx, KnownTyp tx, KnownShape sy, KnownTyp ty, KnownShape sy_) =>
           Options -> (Tensor sx tx -> Tensor sy ty -> Gen (ModelOutput sy_ ty_))
           -- Model input tIn output tOut
        -> Gen ()
compile options f = compileGen options $ do
  x <- placeholder "x"
  f x =<< placeholder "y"


-- | Generic a model with non-standard parameters ("x" and "y" must be
-- provided as placeholders manually).
compileGen :: forall sy ty. (KnownShape sy) =>
           Options -> Gen (ModelOutput sy ty) -> Gen ()
compileGen Options{..} model = knownLast @sy $ do
  gen (text "import tensorflow as tf")
  genFun "mkModel" [text "optimizer=tf.train.AdamOptimizer()"] $ do
    peekAt "optimizer" (T (text "optimizer"))
    peekAt "batch_size" (T (showDim @ (Last sy)))
    trainingPhasePlaceholder <- placeholder "training_phase"
    modify $ \GState{..} -> GState{genTrainingPlaceholder = trainingPhasePlaceholder,..}
    ModelOutput{..} <- model
    y_ <- assign modelY
    peekAt "y_" y_ 
    loss <- assign modelLoss
    peekAt "loss" loss
    accuracy <- assign (reduceMeanAll (cast @Float32 modelCorrect))
    peekAt "accuracy" accuracy
    params <- getParameters
    peekAt "params" (T params)
    trainStep <- assign $ case maxGradientNorm of
      Nothing -> T (funcall "optimizer.minimize" [fromTensor loss])
      Just clip -> T (funcall "optimizer.apply_gradients" [funcall "zip" [clipByGlobalNorm clip (grad loss params),params]])
    peekAt "train" trainStep
    peeks <- gets genPeeks
    gen (text "return " <> dict peeks)

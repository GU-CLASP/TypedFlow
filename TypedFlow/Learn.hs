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
import TypedFlow.Python
import TypedFlow.TF
import qualified Prelude (Float)
import Prelude (($),return,Maybe(..),(=<<),(.))
import Text.PrettyPrint.Compact (text)
import Data.Monoid hiding (Last)
import GHC.TypeLits
import Control.Monad.State (modify, gets)

-- | Triple of values that are always output in a model: prediction, loss and accuracy.
data ModelOutput p s t = ModelOutput {modelY :: T (s++p) t -- ^ prediction (which can contain p-shaped info)
                                     ,modelLoss :: T s Float32 -- ^ loss associated with the prediction
                                     ,modelCorrect :: T s Float32 -- ^ is the above prediction correct?
                                     }

-- | A standard modelling function: (input value, gold value) ↦ (prediction, accuracy, loss)
type Model input tIn g p output tOut = T input tIn -> T (g++output) tOut -> ModelOutput p output tOut

-- modelBoth :: forall n m s t. KnownTyp t => KnownShape s => KnownNat m => KnownNat n =>
--     ModelOutput (n ': s) t -> ModelOutput (m ': s) t -> ModelOutput (n + m ': s) t
-- modelBoth (ModelOutput y1 l1 c1) (ModelOutput y2 l2 c2) = ModelOutput (concat0 y1 y2) (l1 + l2) (concat0 c1 c2)

-- | First type argument is the number of classes.  @categorical
-- logits gold@ return (prediction, accuraccy, loss)

sparseCategorical :: forall nCat. KnownNat nCat => Model '[nCat] Float32 '[] '[] '[] Int32
sparseCategorical logits y =
  let y_ = argmax0 logits
      modelY = y_
      modelCorrect = cast (equal y_ y)
      modelLoss = sparseSoftmaxCrossEntropyWithLogits y logits
  in ModelOutput{..}

-- | First type argument is the number of classes.
-- @categoricalDistribution logits gold@ return (prediction,
-- accuraccy, loss) accuracy is reported as predicting the same class
-- as the input 'winning' class.
categoricalDistribution :: forall nCat. KnownNat nCat => Model '[nCat] Float32 '[nCat] '[nCat] '[] Float32
categoricalDistribution logits y =
  ModelOutput{modelY = softmax0 logits
             ,modelCorrect = cast (equal (argmax0 @'B32 logits) (argmax0 y))
             ,modelLoss = softmaxCrossEntropyWithLogits y logits
             }

-- | @timedCategorical targetWeights logits y@
--
-- targetWeights: a zero-one matrix of the same size as
-- decoder_outputs. It is intended to mask padding positions outside
-- of the target sequence lengths with values 0.
--
-- Note that the accuracy is computed by weigthing the accuracies at
-- individual time steps with the targetWeights.

timedCategorical :: forall len nCat bits. KnownNat nCat => KnownNat len => KnownBits bits =>
  Tensor '[len] (Flt bits) -> Tensor '[len,nCat] (Flt bits) -> Tensor '[len] Int32 -> ModelOutput '[nCat] '[len] (Flt bits)
timedCategorical targetWeights logits y =
  let y_ :: Tensor '[len] Int32
      y_ = argmax1 logits
      modelY = softmax1 logits
      correctPrediction :: Tensor '[len] TFBool
      correctPrediction = equal y_ y
      correctPredictionWeighted :: Tensor '[len] (Flt bits)
      correctPredictionWeighted = cast @(Flt bits) correctPrediction ⊙ targetWeights
      modelCorrect :: Tensor '[len] Float32
      modelCorrect = cast (mapT (⊘ reduceSumAll targetWeights) correctPredictionWeighted )
      crossEntropies = zipWithT sparseSoftmaxCrossEntropyWithLogits y logits
      modelLoss = cast @Float32 (crossEntropies ⊙ targetWeights)
  in ModelOutput{..}

-- | Model with several binary outputs.
binary :: forall n. KnownNat n => Model '[n] Float32 '[] '[] '[n] Int32
binary logits y =
  let y_ = cast @Int32 (round sigy_)
      sigy_ = sigmoid logits
  in ModelOutput {modelY = y_
                 ,modelCorrect = cast (equal y_ y)
                 ,modelLoss = sigmoidCrossEntropyWithLogits (cast @Float32 y) logits}

-- | Model compiler options
data Options = Options {maxGradientNorm :: Maybe Prelude.Float -- ^ apply gradient clipping
                       }

-- | default model compiler options
defaultOptions :: Options
defaultOptions = Options {maxGradientNorm = Nothing}

-- | batchify and compile a simple model
compile :: forall batchSize sx tx sy ty sy_ ty_ p.
           (KnownNat batchSize, KnownShape sx, KnownTyp tx, KnownShape sy, KnownTyp ty, KnownShape sy_, KnownTyp ty_, KnownShape p) =>
           Options -> Gen (Tensor sx tx -> Tensor sy ty -> ModelOutput p sy_ ty_)
           -- Model input tIn output tOut
        -> Gen ()
compile options fGen =
  compileGen @batchSize @p @sy_ @ty_ options $
  knownAppend @sy_ @p $ do
    x <- placeholder "x"
    y <- placeholder "y"
    f <- fGen
    return ModelOutput {modelLoss = zipWithT @batchSize (\x' y' -> modelLoss (f x' y')) x y
                       ,modelY = zipWithT @batchSize (\x' y' -> modelY (f x' y')) x y
                       ,modelCorrect = zipWithT @batchSize (\x' y' -> modelCorrect (f x' y')) x y}


-- | Add a term to the loss. This function is intendend to add
-- regularizers, ie. losses that do not depend on the predicted
-- output, but rather on the structure of a parameter.
addRegularizer :: Scalar Float32 -> Gen ()
addRegularizer r = modify $ \GState{..} -> GState{genRegularizers=r:genRegularizers,..}

-- | Generic a model with non-standard parameters ("x" and "y" must be
-- provided as placeholders manually).
compileGen :: forall bs p sy ty. KnownNat bs => (KnownShape sy, KnownShape p, KnownTyp ty) =>
           Options -> Gen (ModelOutput p (bs ': sy) ty) -> Gen ()
compileGen Options{..} model =
  knownAppend @sy @p $ do
  gen (text "import tensorflow as tf")
  genFun "mkModel" [text "optimizer=tf.train.AdamOptimizer()"] $ do
    peekAtAny "optimizer" (text "optimizer")
    peekAtAny "batch_size" (showDim @ bs)
    trainingPhasePlaceholder <- placeholder "training_phase"
    modify $ \GState{..} -> GState{genTrainingPlaceholder = trainingPhasePlaceholder,..}
    ModelOutput{..} <- model
    y_ <- assign modelY
    peekAt "y_" y_
    regularizers <- gets genRegularizers
    loss <- assign (reduceMeanAll modelLoss ⊕ addN regularizers)
    peekAt "loss" loss
    accuracy <- assign (reduceMeanAll (cast @Float32 modelCorrect))
    peekAt "accuracy" accuracy
    params <- getParameters
    peekAtAny "params" params
    l <- generatePure loss
    trainStep <- assignAny $ case maxGradientNorm of
      Nothing -> funcall "optimizer.minimize" [l]
      Just clip -> funcall "optimizer.apply_gradients" [funcall "zip" [clipByGlobalNorm clip (grad l params),params]]
    peekAtAny "train" trainStep
    peeks <- gets genPeeks
    gen (text "return " <> dict peeks)

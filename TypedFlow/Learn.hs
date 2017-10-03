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
import Prelude (($),return,Maybe(..),id,(=<<))
import Text.PrettyPrint.Compact (text)
import Data.Monoid hiding (Last)
import GHC.TypeLits (KnownNat)
import Control.Monad.State (modify, gets)


binaryCrossEntropy :: KnownNat bs => Tensor '[bs] Float32 -> Tensor '[bs] Float32 -> Tensor '[bs] Float32
binaryCrossEntropy t y = negate (t ⊙ log y) ⊝ (ones ⊝ t) ⊙ log (ones ⊝ y) -- FIXME: add epsilon to avoid NaN

--------------------------------
-- Model maker.

-- | First type argument is the number of classes.
-- @categorical logits gold@
-- return (prediction, accuraccy, loss)
-- accuracy and prediction are averaged over the batch.
categorical :: forall nCat bs. KnownNat nCat => Model '[nCat,bs] Float32 '[bs] Int32
categorical logits' y = do
  logits <- assign logits'
  let y_ = argmax0 logits
      modelY = y_
  correctPrediction <- assign (equal y_ y)
  modelAccuracy <- assign (reduceMeanAll (cast @Float32 correctPrediction))
  modelLoss <- assign (reduceMeanAll (sparseSoftmaxCrossEntropyWithLogits y logits))
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
  correctPrediction <- assign (equal (argmax0 @'B32 logits) (argmax0 y))
  modelAccuracy <- assign (reduceMeanAll (cast @Float32 correctPrediction))
  modelLoss <- assign (reduceMeanAll (softmaxCrossEntropyWithLogits y logits))
  return ModelOutput{..}

-- | @timedCategorical targetWeights logits y@
--
-- targetWeights: a zero-one matrix of the same size as
-- decoder_outputs. It is intended to mask padding positions outside
-- of the target sequence lengths with values 0.

timedCategorical :: forall len nCat bs bits. KnownNat nCat => KnownNat bs => KnownNat len => KnownBits bits =>
  Tensor '[len,bs] (Flt bits) -> Tensor '[len,nCat,bs] (Flt bits) -> Tensor '[len,bs] Int32 -> Gen (ModelOutput '[len,nCat,bs] (Flt bits))
timedCategorical targetWeights logits' y = do
  logits <- assign logits'
  let y_ = argmax1 logits
      modelY = softmax1 logits
  correctPrediction <- assign (equal y_ y)
  modelAccuracy <- assign (cast @Float32 (reduceSumAll (flatten2 (cast @(Flt bits) correctPrediction ⊙ targetWeights)) ⊘ reduceSumAll targetWeights)) --   does not work
  let crossEntropies = sparseSoftmaxCrossEntropyWithLogits y (transpose01 logits)
  modelLoss <- assign (cast @Float32 (reduceMeanAll (crossEntropies ⊙ targetWeights)))
  return ModelOutput{..}

-- | Triple of values that are always output in a model: prediction, loss and accuracy.
data ModelOutput s t = ModelOutput {modelY :: T s t -- ^ prediction
                                   ,modelLoss :: Scalar Float32
                                   ,modelAccuracy :: Scalar Float32
                                   }

-- | A standard modelling function: (input value, gold value) ↦ (prediction, accuracy, loss)
type Model input tIn output tOut = T input tIn -> T output tOut -> Gen (ModelOutput output tOut)

-- | Model with binary output.
binary :: forall bs. (KnownNat bs) => Model '[bs] Float32 '[bs] Int32
binary score y = do
  sigy_ <- assign (sigmoid score)
  let y_ = cast @Int32 (round sigy_)
      modelY = y_
  correctPrediction <- assign (equal y_ y)
  modelAccuracy <- assign (reduceMeanAll (cast @Float32 correctPrediction))
  modelLoss <- assign (reduceMeanAll (binaryCrossEntropy (cast @Float32 y) sigy_))
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
    trainingPhasePlaceholder <- placeholder "training_phase"
    modify $ \GState{..} -> GState{genTrainingPlaceholder = trainingPhasePlaceholder,..}
    ModelOutput{..} <- model
    y_ <- assign modelY
    loss <- assign modelLoss
    accuracy <- assign modelAccuracy
    params <- getParameters
    gradients <- newVar
    let clipping = case maxGradientNorm of
                     Nothing -> id
                     Just clip -> clipByGlobalNorm clip
    gradients <-- clipping (grad modelLoss params)
    peekAt "accuracy" accuracy
    peekAt "loss" loss
    peekAt "y_" y_
    peeks <- gets genPeeks
    trainStep <- newVar
    trainStep <-- funcall "optimizer.apply_gradients" [funcall "zip" [gradients,params]]
    gen (text "return " <> dict ([("train",trainStep)
                                 ,("params",params)
                                 ,("optimizer",text "optimizer")
                                 ,("batch_size",showDim @ (Last sy))
                                 ,("gradients",gradients)] <> peeks))

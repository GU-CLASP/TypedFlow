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



-- crossEntropy :: Tensor '[n,bs] Float32 -> Tensor '[n,bs] Float32 -> Tensor '[bs] Float32
-- crossEntropy y_ y = negate (reduceSum0 (y_ ⊙ log y))

  -- (- t * log(y) - (1 - t) * log(1 - y))

binaryCrossEntropy :: KnownNat bs => Tensor '[bs] Float32 -> Tensor '[bs] Float32 -> Tensor '[bs] Float32
binaryCrossEntropy t y = negate (t ⊙ log y) ⊝ (ones ⊝ t) ⊙ log (ones ⊝ y)

--------------------------------
-- Model maker.

type Batch s batchSize = Tensor (s++'[batchSize])

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

timedCategoricalSparse :: forall len nCat bs. KnownNat nCat => KnownNat bs => KnownNat len =>
  Tensor '[len,bs] Float32 -> Tensor '[len,nCat,bs] Float32 -> Tensor '[len,bs] Int32 -> Gen (ModelOutput '[len,nCat,bs] Float32)
timedCategoricalSparse targetWeights logits' y = do
  logits <- assign logits'
  let y_ = argmax1 logits
      modelY = softmax1 logits
  correctPrediction <- assign (equal y_ y)
  modelAccuracy <- assign (reduceMeanAll (flatten2 (cast @Float32 correctPrediction)))
  let crossEntropies = sparseSoftmaxCrossEntropyWithLogits y (transpose01 logits)
  modelLoss <- assign (reduceMeanAll (crossEntropies ⊙ targetWeights))
  return ModelOutput{..}
  
-- TODO: add a variant of timedCategorical with sampled_softmax_loss

data ModelOutput s t = ModelOutput {modelY :: T s t -- ^ prediction
                                   ,modelLoss :: Scalar Float32
                                   ,modelAccuracy :: Scalar Float32
                                   }
-- | (input value, gold value) ↦ (prediction, accuracy, loss)
type Model input tIn output tOut = T input tIn -> T output tOut -> Gen (ModelOutput output tOut)


binary :: forall bs. (KnownNat bs) => Model '[bs] Float32 '[bs] Int32
binary score y = do
  sigy_ <- assign (sigmoid score)
  let y_ = cast @Int32 (round sigy_)
      modelY = y_
  correctPrediction <- assign (equal y_ y)
  modelAccuracy <- assign (reduceMeanAll (cast @Float32 correctPrediction))
  modelLoss <- assign (reduceMeanAll (binaryCrossEntropy (cast @Float32 y) sigy_))
  return ModelOutput{..}

data Options = Options {maxGradientNorm :: Maybe Prelude.Float}

defaultOptions :: Options
defaultOptions = Options {maxGradientNorm = Nothing}

compile :: forall sx tx sy ty sy_ ty_.
           (KnownShape sx, KnownTyp tx, KnownShape sy, KnownTyp ty, KnownShape sy_) =>
           Options -> (Tensor sx tx -> Tensor sy ty -> Gen (ModelOutput sy_ ty_))
           -- Model input tIn output tOut
        -> Gen ()
compile options f = compileGen options $ do
  x <- placeholder "x"
  f x =<< placeholder "y"


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

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
import Prelude (($),return,Maybe(..),id)
import Text.PrettyPrint.Compact (text)
import Data.Monoid hiding (Last)
import GHC.TypeLits (KnownNat)



-- crossEntropy :: Tensor '[n,bs] Float32 -> Tensor '[n,bs] Float32 -> Tensor '[bs] Float32
-- crossEntropy y_ y = negate (reduceSum0 (y_ ⊙ log y))

  -- (- t * log(y) - (1 - t) * log(1 - y))

binaryCrossEntropy :: KnownNat bs => Tensor '[bs] Float32 -> Tensor '[bs] Float32 -> Tensor '[bs] Float32
binaryCrossEntropy t y = negate (t ⊙ log y) ⊝ (ones ⊝ t) ⊙ log (ones ⊝ y)

--------------------------------
-- Model maker.

type Batch s batchSize = Tensor (s++'[batchSize])

-- data ModelOutput {x , y,  loss :: , accuracy, y_, }

-- | First type argument is the number of classes.
-- @categorical logits gold@
-- return (prediction, accuraccy, loss)
-- accuracy and prediction are averaged over the batch.
categorical :: forall nCat bs. KnownNat nCat => Model '[nCat,bs] Float32 '[bs] Int64
categorical logits' y = do
  logits <- assign logits'
  let y_ = argmax0 logits
      modelY = y_
  correctPrediction <- assign (equal y_ y)
  modelAccuracy <- assign (reduceMeanAll (cast @Float32 correctPrediction))
  modelLoss <- assign (reduceMeanAll (softmaxCrossEntropyWithLogits (oneHot y) logits))
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
  correctPrediction <- assign (equal (argmax0 logits) (argmax0 y))
  modelAccuracy <- assign (reduceMeanAll (cast @Float32 correctPrediction))
  modelLoss <- assign (reduceMeanAll (softmaxCrossEntropyWithLogits y logits))
  return ModelOutput{..}

timedCategoricatDistribution :: forall len nCat bs. KnownNat bs => KnownNat len => Model '[len,nCat,bs] Float32 '[len,nCat,bs] Float32
timedCategoricatDistribution logits' y = do
  logits <- assign logits'
  let y_ = softmax1 logits
      modelY = y_
  correctPrediction <- assign (equal (argmax1 logits) (argmax1 y))
  modelAccuracy <- assign (reduceMeanAll (linearize2 (cast @Float32 correctPrediction)))
  crossEntropies <- zipWithT softmaxCrossEntropyWithLogits y logits
  modelLoss <- assign (reduceMeanAll crossEntropies)
  return ModelOutput{..}
  -- TODO: use sentence length to mask "useless" loss?

type Scalar t = T '[] t

data ModelOutput s t = ModelOutput {modelY :: T s t -- ^ prediction
                                   ,modelLoss :: Scalar Float32
                                   ,modelAccuracy :: Scalar Float32}
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

compile :: (KnownShape input, KnownTyp tIn, KnownShape output, KnownTyp tOut) =>
           Options ->
           Model input tIn output tOut  -> Gen ()
compile Options{..} model = do
  gen (text "import tensorflow as tf")
  genFun "mkModel" [] $ do
    x <- placeholder "x"
    y <- placeholder "y"
    ModelOutput{..} <- model x y
    y_ <- assign modelY
    loss <- assign modelLoss
    accuracy <- assign modelAccuracy
    params <- getParameters
    gradients <- newVar
    let clipping = case maxGradientNorm of
                     Nothing -> id
                     Just clip -> clipByGlobalNorm clip
    gradients <-- clipping (grad modelLoss params)
    gen (text "return " <> tuple [fromTensor x,fromTensor y,fromTensor y_,fromTensor accuracy,fromTensor loss,params,gradients])

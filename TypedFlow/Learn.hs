{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE PatternSynonyms #-}
{-|
Module      : TypedFlow.Learn
Description : Loss functions and optimization strategies
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental
-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE UnicodeSyntax #-}

module TypedFlow.Learn
  (-- losses:
    sparseCategorical, binary, timedCategorical, categoricalDistribution,sparseCategoricalDensePredictions,
    -- types
    Model,ModelOutput,PreparedModel(..), Options(..), defaultOptions,
    -- other
    simpleModel,
    addRegularizer,
    prepare,
    -- utils
    placeholderName,
  ) where

import Data.Proxy
import TypedFlow.Types
import TypedFlow.Types.Proofs (knownAppend, knownAppendS, (?>))
import TypedFlow.Abstract (G,runBC,broadcastGen, type BatchedPlaceholders, type BPH, doExtractVars, ConsSh)
import TypedFlow.TF
import Prelude hiding (RealFrac(..))
import GHC.TypeLits

-- | Triple of values that are always output in a model: prediction, loss and accuracy.
-- @t@ is the type of the prediction.
-- @s@ is the shape of the loss and accuracy
type ModelOutput t predictionShape s
  = Placeholders '[ '("loss",s,Float32) -- loss associated with the prediction
                  , '("accuracy",s,Float32)  -- is the prediction correct?
                  , '("y_",s++predictionShape,t) -- prediction (which can contain prediction-shaped info)
                  ]

pattern ModelOutput ::  T (s++predictionShape) t -> T s Float32 -> T s Float32 -> ModelOutput t predictionShape s
pattern ModelOutput y loss accur = PHT loss :* PHT accur :* PHT y :* Unit

-- | A standard modelling function: (input value, gold value) ↦ (prediction, accuracy, loss).
-- input is the shape of the input.
-- output is the shape of the output (one element per individual loss and accuracy)
-- p is the shape of each output element.
-- g is the shape of each gold output --- often equal to p.
type Model input tIn g p output tOut
  = T input tIn -> T (g++output) tOut -> ModelOutput tOut p output

-- | First type argument is the number of classes.  @categorical
-- logits gold@ return (prediction, accuraccy, loss)

sparseCategorical :: forall nCat. KnownNat nCat => Model '[nCat] Float32 '[] '[] '[] Int32
sparseCategorical logits y =
  let y_ = argmax0 logits
      modelCorrect = cast (equal y_ y)
      modelLoss = sparseSoftmaxCrossEntropyWithLogits y logits
  in ModelOutput y_ modelCorrect modelLoss

-- | First type argument is the number of classes.  @categorical
-- logits gold@ return (prediction, accuracy, loss)
sparseCategoricalDensePredictions :: forall nCat. KnownNat nCat
  => Tensor '[nCat] Float32
  -> Tensor '[] Int32
  -> ModelOutput  Float32 '[nCat] '[]
sparseCategoricalDensePredictions logits y =
  let y_ :: T '[nCat] Float32
      y_ = softmax0 logits
      modelCorrect = cast (equal (argmax0 logits) y)
      modelLoss = sparseSoftmaxCrossEntropyWithLogits y logits
  in ModelOutput y_ modelCorrect modelLoss


-- | First type argument is the number of classes.
-- @categoricalDistribution logits gold@ return (prediction,
-- accuraccy, loss) accuracy is reported as predicting the same class
-- as the input 'winning' class.
categoricalDistribution :: forall nCat. KnownNat nCat => Model '[nCat] Float32 '[nCat] '[nCat] '[] Float32
categoricalDistribution logits y =
  ModelOutput (softmax0 logits)
              (cast (equal (argmax0 @'B32 logits) (argmax0 y)))
              (softmaxCrossEntropyWithLogits y logits)
  

-- | @timedCategorical targetWeights logits y@
--
-- targetWeights: a zero-one matrix of the same size as
-- decoder_outputs. It is intended to mask padding positions outside
-- of the target sequence lengths with values 0.
--
-- Note that the accuracy is computed by multiplying the accuracies at
-- individual time steps with the targetWeights.

timedCategorical :: forall len nCat bits. KnownNat nCat => KnownNat len => KnownBits bits =>
  Tensor '[len] (Flt bits) -> Tensor '[len,nCat] (Flt bits) -> Tensor '[len] Int32 -> ModelOutput  (Flt bits) '[len,nCat] '[]
timedCategorical targetWeights logits y =
  let y_ :: Tensor '[len] Int32
      y_ = argmax1 logits
      modelY = softmax1 logits
      -- correct prediction for each position
      correctPrediction :: Tensor '[len] TFBool
      correctPrediction = equal y_ y
      -- total number of correct predictions
      correctPredictionWeighted :: Tensor '[] (Flt bits)
      correctPredictionWeighted = reduceSumAll (cast @(Flt bits) correctPrediction ⊙ targetWeights)
      weightSum = reduceSumAll targetWeights
      modelCorrect :: Tensor '[] Float32
      modelCorrect = cast (correctPredictionWeighted / weightSum)
      crossEntropies = zipWithT sparseSoftmaxCrossEntropyWithLogits y logits
      modelLoss = cast @Float32 (reduceSumAll (crossEntropies ⊙ targetWeights) / weightSum)
  in ModelOutput modelY modelLoss modelCorrect

-- | Model with @n@ binary outputs.
binary :: KnownNat n => Model '[n] Float32 '[] '[] '[n] Int32
binary logits y =
  let y_ = cast @Int32 (round sigy_)
      sigy_ = sigmoid logits
  in ModelOutput (y_)
                 (cast (equal y_ y))
                 (sigmoidCrossEntropyWithLogits (cast @Float32 y) logits)

-- | Model compiler options
data Options = Options {maxGradientNorm :: Maybe Prelude.Float -- ^ apply gradient clipping
                       }

-- | default model compiler options
defaultOptions :: Options
defaultOptions = Options {maxGradientNorm = Nothing}

type family Concatenate xs where
  Concatenate (x ': xs) = x ++ Concatenate xs
  Concatenate '[] = '[]

genPlaceholders :: All KnownPlaceholder shapesAndTypes => SList shapesAndTypes -> Placeholders shapesAndTypes
genPlaceholders Unit = Unit
genPlaceholders (ph :* names) = PHT (T (ExternalVar (Ref (placeholderName ph) typeSShape typeSTyp))) :* genPlaceholders names

placeholderName :: forall (ph :: PH)  p. KnownPlaceholder ph => p ph -> String
placeholderName proxy = refName (placeHolderRef proxy)

simpleModel :: forall p sx tx sy ty sy_ ty_. KnownLen sy_ => (Tensor sx tx -> Tensor sy ty -> ModelOutput  ty_ p sy_) ->
            (Placeholders '[ '("x",sx, tx), '("y",sy, ty)] -> ModelOutput ty_ p sy_)
simpleModel f = f'
  where f' :: Placeholders '[ '("x",sx,tx), '("y",sy,ty)] -> ModelOutput ty_ p sy_
        f' (PHT x :* PHT y :* Unit) = f x y

-- | Add a term to the loss. This function is intendend to add
-- regularizers, ie. losses that do not depend on the predicted
-- output, but rather on the structure of a parameter.
addRegularizer :: Scalar Float32 -> Gen ()
addRegularizer r = GPState  $ \GState{..} -> ((),GState{genRegularizers=r:genRegularizers,..})

-- | Batch the model (adding one dimension)
batchModel :: forall batchSize shapesAndTypes resShapesAndTypes.
           (KnownNat batchSize, KnownLen shapesAndTypes, All KnownPlaceholder shapesAndTypes, All KnownPlaceholder resShapesAndTypes)
         => (Placeholders shapesAndTypes -> Placeholders resShapesAndTypes)
         -> (BatchedPlaceholders batchSize shapesAndTypes -> BatchedPlaceholders batchSize resShapesAndTypes)
batchModel f xs = runBC u (broadcastGen @batchSize @resShapesAndTypes u True (Proxy @batchSize) (f (unbroadcastPlacehoders @batchSize typeSList xs)))
 where u = -777 -- unique identifier for the batch dimension

       unbroadcastPlacehoders :: forall n r. KnownNat n => SList r -> BatchedPlaceholders n r -> Placeholders r
       unbroadcastPlacehoders Unit Unit = Unit
       unbroadcastPlacehoders (_ :* ss) (PHT x :* xs) = PHT (Unbroadcast batchSize u x) :* unbroadcastPlacehoders @n ss xs
         where batchSize = natSat @n
       
knownBatchModel :: forall n ps. KnownNat n => NP (Sat KnownPlaceholder) ps -> NP (Sat KnownPlaceholder) (Ap (FMap (ConsSh n)) ps)
knownBatchModel Unit = Unit
knownBatchModel (Sat :* xs) = Sat :* knownBatchModel @n xs

-- | take the mean of loss/accur, and add regulariser to loss
consolidate :: forall s rest. KnownShape s
            => Scalar Float32
            -> Placeholders ( '("loss",s  ,Float32) ': '("accuracy",s  ,Float32) ': rest)
            -> Placeholders ( '("loss",'[],Float32) ': '("accuracy",'[],Float32) ': rest)
consolidate extraLoss (PHT loss :* PHT accur :* rest) = (PHT (reduceMeanAll loss + extraLoss) :* PHT (reduceMeanAll accur) :* rest)

class (All KnownPlaceholder ps, KnownLen ps) => KnownPHS ps
instance (All KnownPlaceholder ps, KnownLen ps) => KnownPHS ps

data PreparedModel = PreparedModel {pmBatchSize :: Integer,
                                    pmParams :: [VarInfo],
                                    pmPlaceHolders :: SomeSuch KnownPHS Placeholders,
                                    pmResults :: SomeSuch KnownPHS Placeholders}

-- | Prepare compilation by:
-- extracting and exposing parameters 
-- batching the model
-- exposing placeholders
-- consolidating loss and accuracy
-- adding regularizers to the loss
prepare :: forall bs st1 st2 s. KnownShape s => (KnownLen st1, KnownLen st2, All KnownPlaceholder st1, All KnownPlaceholder st2, KnownNat bs)
        => Gen (Placeholders st1 -> Placeholders ('("loss",s,Float32) ': '("accuracy",s,Float32) ': st2)) -> PreparedModel
prepare fGen =
  knownAll (knownBatchModel @bs (allKnown @KnownPlaceholder @st1)) $ -- prove known
  knownAll (knownBatchModel @bs (allKnown @KnownPlaceholder @st2)) $ -- prove known
  let placeHolders = genPlaceholders typeSList
  in PreparedModel
       {pmBatchSize = natVal (Proxy @bs)
       ,pmParams = filter varTrainable vars
       ,pmPlaceHolders = SomeSuch placeHolders
       ,pmResults = SomeSuch (consolidate @(bs ': s) @(BPH bs st2) regular (batchModel @bs f placeHolders))
       }
  where (f,finalState,vars) = doExtractVars fGen
        regular = sum (genRegularizers finalState)



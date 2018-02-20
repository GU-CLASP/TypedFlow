{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE InstanceSigs #-}
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
import TypedFlow.Abstract (Batched(..),broadcastGen)
import TypedFlow.Python
import TypedFlow.TF
import qualified Prelude (Float)
import Prelude (($),return,Maybe(..),(=<<),(.),Bool(True),String)
import Text.PrettyPrint.Compact (text)
import Data.Monoid hiding (Last,All)
import GHC.TypeLits
import Control.Monad.State (modify, gets)
-- | Triple of values that are always output in a model: prediction, loss and accuracy.
data ModelOutput t predictionShape s = ModelOutput {modelY :: T (s++predictionShape) t -- ^ prediction (which can contain p-shaped info)
                                                   ,modelLoss :: T s Float32 -- ^ loss associated with the prediction
                                                   ,modelCorrect :: T s Float32 -- ^ is the above prediction correct?
                                                   }

instance (KnownShape p, KnownTyp t) => Batched (ModelOutput t p) where
  batchify :: forall n r. KnownNat n => KnownShape r
    => (forall s u. KnownTyp u => KnownShape s => T s u -> T (n:s) u) -> ModelOutput t p  r -> ModelOutput t p (n:r)
  batchify f (ModelOutput{..}) = ModelOutput {modelLoss = f modelLoss
                                             ,modelY = knownAppend @r @p (f modelY)
                                             ,modelCorrect = f modelCorrect}

-- | A standard modelling function: (input value, gold value) ↦ (prediction, accuracy, loss)
type Model input tIn g p output tOut = T input tIn -> T (g++output) tOut
                                       -> ModelOutput tOut p output 

-- modelBoth :: -- forall n m s t. KnownTyp t => KnownShape s => KnownNat m => KnownNat n =>
--     ModelOutput t '[p] s -> ModelOutput t '[q] s -> ModelOutput t '[p + q] s
-- modelBoth (ModelOutput y1 l1 c1) (ModelOutput y2 l2 c2) = ModelOutput (concatT (lengthAsAxis @s) y1 y2) (l1 + l2) (c1 + c2)

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
  Tensor '[len] (Flt bits) -> Tensor '[len,nCat] (Flt bits) -> Tensor '[len] Int32 -> ModelOutput  (Flt bits) '[nCat] '[len]
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
binary :: Model '[] Float32 '[] '[] '[] Int32
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




data HolderName a = HolderName String

class (KnownShape (Fst r), KnownTyp (Snd r)) => KnownPair r where

instance (KnownShape x, KnownTyp y) => KnownPair (x ':& y) where

genBatchedPlaceholders :: All KnownPair shapesAndTypes => Sat KnownNat n -> SList' HolderName shapesAndTypes -> Gen (HHTV shapesAndTypes)
genBatchedPlaceholders _ LZ = return Unit
genBatchedPlaceholders n@Sat (LS (HolderName name) names) = do
  x <- (placeholder name)
  xs <- genBatchedPlaceholders n names
  return (Uncurry (Unbroadcast n x) :* xs) 

compile' :: forall batchSize shapesAndTypes sy_ ty_ p.
           (KnownNat batchSize, All KnownPair shapesAndTypes, KnownShape sy_, KnownTyp ty_, KnownShape p) =>
           Options -> SList' HolderName shapesAndTypes -> Gen (HHTV shapesAndTypes -> ModelOutput  ty_ p sy_)
         -> Gen ()
compile' options names fGen = 
  compileAlreadyBatched @batchSize @p @sy_ @ty_ options $
  knownAppend @sy_ @p $ do
  xs <- genBatchedPlaceholders batchSize names
  f <- fGen
  return $ broadcastGen True batchSize (f xs)
 where batchSize = natSat @batchSize


-- | batchify and compile a simple model
compile :: forall batchSize sx tx sy ty sy_ ty_ p.
           (KnownNat batchSize, KnownShape sx, KnownTyp tx, KnownShape sy, KnownTyp ty, KnownShape sy_, KnownTyp ty_, KnownShape p) =>
           Options -> Gen (Tensor sx tx -> Tensor sy ty -> ModelOutput  ty_ p sy_)
           -- Model input tIn output tOut
        -> Gen ()
compile options fGen = do
  compile' @batchSize options (LS (HolderName "x") (LS (HolderName "y") LZ)) $ do
    f <- fGen
    let f' :: HHTV '[sx ':& tx, sy ':& ty] -> ModelOutput ty_ p sy_
        f' (Uncurry x :* Uncurry y :* Unit) = f x y
    return f'



-- | Add a term to the loss. This function is intendend to add
-- regularizers, ie. losses that do not depend on the predicted
-- output, but rather on the structure of a parameter.
addRegularizer :: Scalar Float32 -> Gen ()
addRegularizer r = modify $ \GState{..} -> GState{genRegularizers=r:genRegularizers,..}

-- | Generic a model with non-standard parameters ("x" and "y" must be
-- provided as placeholders manually).
compileAlreadyBatched :: forall bs p sy ty. KnownNat bs => (KnownShape sy, KnownShape p, KnownTyp ty) =>
           Options -> Gen (ModelOutput ty p (bs ': sy)) -> Gen ()
compileAlreadyBatched Options{..} model =
  knownAppend @sy @p $ do
  gen (text "import tensorflow as tf")
  genFun "mkModel" [text "optimizer=tf.train.AdamOptimizer()"] $ do
    peekAtAny "optimizer" (text "optimizer")
    peekAtAny "batch_size" (showDim @ bs)
    trainingPhasePlaceholder <- placeholder "training_phase"
    modify $ \GState{..} -> GState{genTrainingPlaceholder = trainingPhasePlaceholder,..}
    ModelOutput{..} <- model
    peekAt "y_"  modelY
    regularizers <- gets genRegularizers
    loss <- generatePure (reduceMeanAll modelLoss ⊕ addN regularizers)
    peekAtAny "loss" loss
    peekAt "accuracy" (reduceMeanAll (cast @Float32 modelCorrect))
    params <- getParameters
    peekAtAny "params" params
    trainStep <- assignAny $ case maxGradientNorm of
      Nothing -> funcall "optimizer.minimize" [loss]
      Just clip -> funcall "optimizer.apply_gradients" [funcall "zip" [clipByGlobalNorm clip (grad loss params),params]]
    peekAtAny "train" trainStep
    peeks <- gets genPeeks
    gen (text "return " <> dict peeks)

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

import Data.Proxy
import System.IO.Unsafe
import Data.Unique
import TypedFlow.Types
import TypedFlow.Types.Proofs (knownAppend, knownAppendS)
import TypedFlow.Abstract (Batched(..),broadcastGen)
import TypedFlow.Python
import TypedFlow.TF
import Prelude hiding (RealFrac(..))
import Text.PrettyPrint.Compact (text)
import Data.Monoid hiding (Last,All)
import GHC.TypeLits
import Control.Monad.State (modify, gets)
-- | Triple of values that are always output in a model: prediction, loss and accuracy.
data ModelOutput t predictionShape s = ModelOutput {modelY :: T (s++predictionShape) t -- ^ prediction (which can contain prediction-shaped info)
                                                   ,modelLoss :: T s Float32 -- ^ loss associated with the prediction
                                                   ,modelCorrect :: T s Float32 -- ^ is the above prediction correct?
                                                   }

-- | A standard modelling function: (input value, gold value) ↦ (prediction, accuracy, loss).
-- input is the shape of the input.
-- output is the shape of the output (one element per individual loss and accuracy)
-- p is the shape of each output element.
-- g is the shape of each gold output --- often equal to p.
type Model input tIn g p output tOut = T input tIn -> T (g++output) tOut
                                       -> ModelOutput tOut p output

modelBoth :: forall p q s t. 
    KnownShape s => KnownTyp t => KnownNat q => KnownNat p => ModelOutput t '[p] s -> ModelOutput t '[q] s -> ModelOutput t '[p + q] s
modelBoth (ModelOutput y1 l1 c1) (ModelOutput y2 l2 c2) = ModelOutput arst (l1 + l2) (c1 + c2)
    where arst :: T (s ++ '[p + q]) t
          arst = zipWithTT @s @'[p] @'[q] concat0 y1 y2

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

-- | Model with @n@ binary outputs.
binary :: KnownNat n => Model '[n] Float32 '[] '[] '[n] Int32
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

-- | Name of a placeholder of a given shape and type.
data HolderName (st :: (Shape,Typ)) = HolderName String


-- | A fancily-typed pair of a model output and updateable variables (as an HTV)
data StateAndOutput t p ss where
  StateAndOutput :: SList s -> ModelOutput t p s -> HTV t ss -> StateAndOutput t p (s ': ss)

instance (KnownTyp t, KnownShape p) => Batched (StateAndOutput t p) where
  batchify n f (StateAndOutput s ModelOutput{..} xs)
    = StateAndOutput (n :* s)
      ModelOutput{modelLoss = f modelLoss
                 ,modelY = knownAppendS s (Proxy @p) (f modelY)
                 ,modelCorrect = f modelCorrect}
      (batchify n f xs)

genBatchedPlaceholders :: All KnownPair shapesAndTypes => Unique -> Sat KnownNat n -> SList' HolderName shapesAndTypes -> Gen (HHTV shapesAndTypes)
genBatchedPlaceholders _ _ Unit = return Unit
genBatchedPlaceholders u n@Sat (HolderName name :* names) = do
  x <- placeholder name
  xs <- genBatchedPlaceholders u n names
  return (Uncurry (Unbroadcast n u x) :* xs)


knownCons :: KnownNat x => Sat KnownShape s -> Sat KnownShape (x ': s)
knownCons Sat = Sat

compileGen :: forall batchSize shapesAndTypes sy_ ty_ p stateShapes.
           (KnownNat batchSize, All KnownPair shapesAndTypes, KnownLen stateShapes,
            All KnownShape stateShapes, KnownShape sy_, KnownTyp ty_, KnownShape p)
         => Options
         -> SList' HolderName shapesAndTypes -- ^ names for the inputs
         -> Gen (HHTV shapesAndTypes -> HTV ty_ stateShapes -> (StateAndOutput ty_ p (sy_ ': stateShapes)) )
         -> Gen ()
compileGen options names fGen = do
  let u = unsafePerformIO newUnique -- unique identifier for the batch dimension
      -- (state :: HTV ty_ stateShapes) <- travTensor (persistent False) "state" (repeatT (Unbroadcast batchSize u zeros))
      unbroadcastStates :: forall ss. SList ss -> HTV ty_ (Ap (FMap (Cons batchSize)) ss) -> HTV ty_ ss
      unbroadcastStates Unit Unit = Unit
      unbroadcastStates (_ :* ss) (F x :* xs) = F (Unbroadcast batchSize u x) :* unbroadcastStates ss xs
      batchedShapesKnown = mapFMap @(Cons batchSize) knownCons (allKnown @KnownShape @stateShapes typeSList)
  knownAll batchedShapesKnown $
    compileAlreadyBatched @batchSize @p @sy_ @ty_ @(Ap (FMap (Cons batchSize)) stateShapes) options $ \stateVars ->
    knownAppend @sy_ @p $ do
      xs <- genBatchedPlaceholders u batchSize names
      f <- fGen
      return $ broadcastGen u True (Proxy @batchSize) (f xs (unbroadcastStates (typeSList) stateVars))
 where batchSize = natSat @batchSize

-- | Turn a stateless modelling function into a trivially stateful one.
stateless :: KnownLen s => (inputs -> ModelOutput t p s) -> inputs -> HTV t '[] -> StateAndOutput t p '[ s ]
stateless f x Unit = StateAndOutput typeSList (f x) Unit

-- | Batchify and compile a model with simple input to output mapping.
compile :: forall batchSize sx tx sy ty sy_ ty_ p.
           (KnownNat batchSize, KnownShape sx, KnownTyp tx, KnownShape sy, KnownTyp ty, KnownShape sy_, KnownTyp ty_, KnownShape p) =>
           Options -> Gen (Tensor sx tx -> Tensor sy ty -> ModelOutput  ty_ p sy_)
           -- Model input tIn output tOut
        -> Gen ()
compile options fGen = do
  compileGen @batchSize options (HolderName "x" :* HolderName "y" :* Unit) $ do
    f <- fGen
    let f' :: HHTV '[ '(sx,tx), '(sy,ty)] -> ModelOutput ty_ p sy_ 
        f' (Uncurry x :* Uncurry y :* Unit) = f x y
    return (stateless f')

-- | @updateStates xs ys@ assigns to the tensor (variables!) xs the values ys.
updateStates :: forall xs ty. KnownTyp ty => All KnownShape xs => HTV ty xs -> HTV ty xs -> Gen (HTV ty xs)
updateStates Unit Unit = pure Unit
updateStates (F x :* xs) (F y :* ys) = (:*) <$> (F <$> modifyPersistent x y) <*> updateStates xs ys

-- | Add a term to the loss. This function is intendend to add
-- regularizers, ie. losses that do not depend on the predicted
-- output, but rather on the structure of a parameter.
addRegularizer :: Scalar Float32 -> Gen ()
addRegularizer r = modify $ \GState{..} -> GState{genRegularizers=r:genRegularizers,..}

-- | Generic model preparation, with non-standard parameters ("x", "y"
--  must be provided as placeholders manually).
compileAlreadyBatched :: forall bs p sy ty stateShapes. KnownNat bs
           => All KnownShape stateShapes
           => KnownLen stateShapes
           => (KnownShape sy, KnownShape p, KnownTyp ty)
           => Options
           -> (HTV ty stateShapes -> Gen (StateAndOutput ty p ((bs ': sy) ': stateShapes))) -> Gen ()
compileAlreadyBatched Options{..} model =
  knownAppend @sy @p $ do
  gen (text "import tensorflow as tf")
  genFun "mkModel" [text "optimizer=tf.train.AdamOptimizer()"] $ do
    peekAtAny "optimizer" (text "optimizer")
    peekAtAny "batch_size" (showDim @ bs)
    trainingPhasePlaceholder <- placeholder "training_phase"
    modify $ \GState{..} -> GState{genTrainingPlaceholder = trainingPhasePlaceholder,..}
    (stateVars :: HTV ty stateShapes) <- travTensor (persistent False) "state" (repeatT zeros)
    (StateAndOutput _ ModelOutput{..} newStates) <- model stateVars
    updates <- updateStates @stateShapes stateVars newStates
    peekAtMany "update" updates
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

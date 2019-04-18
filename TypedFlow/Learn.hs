{-# LANGUAGE ConstraintKinds #-}
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
import TypedFlow.Types.Proofs (knownAppend, knownAppendS, (?>))
import TypedFlow.Abstract (Batched(..),broadcastGen,defaultT)
import TypedFlow.TF
import Prelude hiding (RealFrac(..))
import GHC.TypeLits
import Control.Monad.State (modify, gets)

-- | Triple of values that are always output in a model: prediction, loss and accuracy.
-- @t@ is the type of the prediction.
-- @s@ is the shape of the loss and accuracy
data ModelOutput t predictionShape s =
  ModelOutput {modelY :: T (s++predictionShape) t -- ^ prediction (which can contain prediction-shaped info)
              ,modelLoss :: T s Float32 -- ^ loss associated with the prediction
              ,modelCorrect :: T s Float32 -- ^ is the above prediction correct?
              ,modelName :: String
              }

-- | Several model outputs (for multitask models)
type ModelOutputs t ps s = NP (F (ModelOutput t) s) ps

type OneOutput t p s = ModelOutputs t '[ p ] s

singleModel :: ModelOutput t p s -> ModelOutputs t '[ p ] s
singleModel m = F m :* Unit

-- | A standard modelling function: (input value, gold value) ↦ (prediction, accuracy, loss).
-- input is the shape of the input.
-- output is the shape of the output (one element per individual loss and accuracy)
-- p is the shape of each output element.
-- g is the shape of each gold output --- often equal to p.
type Model input tIn g p output tOut
  = T input tIn -> T (g++output) tOut -> OneOutput tOut p output

modelBoth :: forall p q s t. 
    KnownShape s => KnownTyp t => KnownNat q => KnownNat p => ModelOutput t '[p] s -> ModelOutput t '[q] s -> ModelOutput t '[p + q] s
modelBoth (ModelOutput y1 l1 c1 n1) (ModelOutput y2 l2 c2 _) = ModelOutput arst (l1 + l2) (c1 + c2) n1
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
      modelName = ""
  in singleModel (ModelOutput{..})

-- | First type argument is the number of classes.  @categorical
-- logits gold@ return (prediction, accuraccy, loss)
sparseCategoricalDensePredictions :: forall nCat. KnownNat nCat
  => Tensor '[nCat] Float32
  -> Tensor '[] Int32
  -> ModelOutput  Float32 '[nCat] '[]
sparseCategoricalDensePredictions logits y =
  let y_ :: T '[nCat] Float32
      y_ = softmax0 logits
      modelY = y_
      modelCorrect = cast (equal (argmax0 logits) y)
      modelLoss = sparseSoftmaxCrossEntropyWithLogits y logits
      modelName = ""
  in ModelOutput{..}


-- | First type argument is the number of classes.
-- @categoricalDistribution logits gold@ return (prediction,
-- accuraccy, loss) accuracy is reported as predicting the same class
-- as the input 'winning' class.
categoricalDistribution :: forall nCat. KnownNat nCat => Model '[nCat] Float32 '[nCat] '[nCat] '[] Float32
categoricalDistribution logits y = singleModel
  ModelOutput{modelY = softmax0 logits
             ,modelCorrect = cast (equal (argmax0 @'B32 logits) (argmax0 y))
             ,modelLoss = softmaxCrossEntropyWithLogits y logits
             ,modelName = ""
             }

-- | @timedCategorical targetWeights logits y@
--
-- targetWeights: a zero-one matrix of the same size as
-- decoder_outputs. It is intended to mask padding positions outside
-- of the target sequence lengths with values 0.
--
-- Note that the accuracy is computed by multiplying the accuracies at
-- individual time steps with the targetWeights.

timedCategorical :: forall len nCat bits. KnownNat nCat => KnownNat len => KnownBits bits =>
  Tensor '[len] (Flt bits) -> Tensor '[len,nCat] (Flt bits) -> Tensor '[len] Int32 -> ModelOutputs  (Flt bits) '[ '[len,nCat]] '[]
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
      modelName = ""
  in singleModel ModelOutput{..}

-- | Model with @n@ binary outputs.
binary :: KnownNat n => Model '[n] Float32 '[] '[] '[n] Int32
binary logits y =
  let y_ = cast @Int32 (round sigy_)
      sigy_ = sigmoid logits
  in singleModel ModelOutput {modelY = y_
                             ,modelName = ""
                             ,modelCorrect = cast (equal y_ y)
                             ,modelLoss = sigmoidCrossEntropyWithLogits (cast @Float32 y) logits}

-- | Model compiler options
data Options = Options {maxGradientNorm :: Maybe Prelude.Float -- ^ apply gradient clipping
                       }

-- | default model compiler options
defaultOptions :: Options
defaultOptions = Options {maxGradientNorm = Nothing}

type family Concatenate xs where
  Concatenate (x ': xs) = x ++ Concatenate xs
  Concatenate '[] = '[]


-- | A fancily-typed pair of several model outputs and updateable variables (as an HTV)
data StateAndOutputs t ps ss where
  StateAndOutputs :: SList s -> ModelOutputs t ps s -> HTV t ss -> StateAndOutputs t ps (s ': ss)

instance (KnownTyp t, All KnownShape ps) => Batched (StateAndOutputs t ps) where
  batchify :: forall n r. KnownNat n => All KnownShape r
    => Proxy n -> (forall s u. KnownTyp u => KnownShape s => T s u -> T (n:s) u)
    -> (StateAndOutputs t ps) r  -> (StateAndOutputs t ps) (Ap (FMap (Cons n)) r)
  batchify n f (StateAndOutputs s ms xs) = StateAndOutputs (n :* s) (hmapK @KnownShape (h s) ms) (batchify n f xs)
    where h :: forall x s. KnownShape s => KnownShape x => SList s -> F (ModelOutput t) s x -> F (ModelOutput t) (n ': s) x
          h s' (F ModelOutput{..}) = F ModelOutput{modelLoss = f modelLoss
                                                  ,modelY = knownAppendS s' (Proxy @x) ?> f modelY
                                                  ,modelName = modelName
                                                  ,modelCorrect = f modelCorrect}

-- | Prefix the model names
nameModels :: String -> ModelOutputs t ps s -> ModelOutputs t ps s
nameModels _ Unit = Unit
nameModels prefix (F ModelOutput {..} :* xs) = F ModelOutput {modelName = prefix ++ modelName, ..} :* nameModels prefix xs

-- | Name of a placeholder of a given shape and type.
data HolderName (st :: (Symbol,Shape,Typ)) = HolderName String

holderName :: forall (st :: (Symbol,Shape,Typ)) proxy. KnownSymbol (Frst3 st) => proxy st -> String
holderName _ = symbolVal (Proxy @(Frst3 st))

genBatchedPlaceholders :: All KnownPlaceholder shapesAndTypes
  => Unique -> Sat KnownNat n -> SList shapesAndTypes -> Gen (Placeholders shapesAndTypes)
genBatchedPlaceholders _ _ Unit = return Unit
genBatchedPlaceholders u n@Sat (name :* names) = do
  x <- placeholder (holderName name)
  xs <- genBatchedPlaceholders u n names
  return (PHT (Unbroadcast n u x) :* xs)


knownCons :: KnownNat x => Sat KnownShape s -> Sat KnownShape (x ': s)
knownCons Sat = Sat

-- | Turn a stateless modelling function into a trivially stateful one.
stateless :: KnownLen s => (inputs -> ModelOutputs t p s) -> inputs -> HTV t '[] -> StateAndOutputs t p '[ s ]
stateless f x Unit = StateAndOutputs typeSList (f x) Unit

simpleModel :: forall sx tx sy ty sy_ ty_ p. KnownLen sy_ => (Tensor sx tx -> Tensor sy ty -> ModelOutputs  ty_ p sy_) ->
            (Placeholders '[ '("x",sx, tx), '("y",sy, ty)] -> HTV ty_ '[] -> (StateAndOutputs ty_ p (sy_ ': '[])))
simpleModel f = stateless f'
  where f' :: Placeholders '[ '("x",sx,tx), '("y",sy,ty)] -> ModelOutputs ty_ p sy_
        f' (PHT x :* PHT y :* Unit) = f x y

-- | @updateStates xs ys@ assigns to the tensor (variables!) xs the values ys.
updateStates :: forall xs ty. KnownTyp ty => All KnownShape xs => HTV ty xs -> HTV ty xs -> Gen (HTV ty xs)
updateStates Unit Unit = pure Unit
updateStates (F x :* xs) (F y :* ys) = (:*) <$> (F <$> modifyPersistent x y) <*> updateStates xs ys

-- | Add a term to the loss. This function is intendend to add
-- regularizers, ie. losses that do not depend on the predicted
-- output, but rather on the structure of a parameter.
addRegularizer :: Scalar Float32 -> Gen ()
addRegularizer r = modify $ \GState{..} -> GState{genRegularizers=r:genRegularizers,..}


-- | Prepares the (already batched) model for compilation:
-- - add training phase placeholder
-- - create the state variables
-- - compute final accuracy and loss (adding eventual regularizers), and expose them.
precompile :: forall ps sy ty stateShapes.
              All KnownShape stateShapes
           => KnownLen stateShapes
           => (KnownShape sy, All KnownShape ps, KnownTyp ty, KnownLen ps)
           => (HTV ty stateShapes -> Gen (StateAndOutputs ty ps (sy ': stateShapes)))
           -> (Gen (NP (K (String, HTV ty stateShapes,Scalar Float32)) ps))
precompile model =
 do regularizers <- gets genRegularizers
    trainingPhasePlaceholder <- placeholder "training_phase"
    modify $ \GState{..} -> GState{genTrainingPlaceholder = trainingPhasePlaceholder,..}
    (stateVars :: HTV ty stateShapes) <- travTensor (persistent False) "state" (repeatT defaultT)
    (StateAndOutputs _ models newStates) <- model stateVars
    updates <- updateStates @stateShapes stateVars newStates
    let go :: forall xs. NP (Sat KnownShape) xs -> ModelOutputs ty xs sy -> Gen (NP (K (String,HTV ty stateShapes,Scalar Float32)) xs)
        go Unit Unit = return Unit
        go (x@Sat :* xs) (F ModelOutput{..} :* ms) = knownAppendS (typeSList @sy) x ?> do
          let loss = reduceMeanAll modelLoss ⊕ addN regularizers
              accuracy = reduceMeanAll (cast @Float32 modelCorrect)
              y_ = modelY
          peekAt (modelName ++ "y_")  y_
          peekAt (modelName ++ "accuracy") accuracy
          (K (modelName,updates,loss) :*) <$> go xs ms
    go (allKnown @KnownShape typeSList) models



-- | Batch the model (adding one dimension), create placeholders for the inputs.
batchModel :: forall batchSize shapesAndTypes resShapes ty_ stateShapes f.
           (KnownNat batchSize, KnownLen shapesAndTypes, All KnownPlaceholder shapesAndTypes, KnownLen stateShapes,
            All KnownShape stateShapes, KnownTyp ty_, All KnownShape resShapes, Batched f)
         => Gen (Placeholders shapesAndTypes -> HTV ty_ stateShapes -> f resShapes )
         -> HTV ty_ (Ap (FMap (Cons batchSize)) stateShapes) -- ^ state variables
         -> Gen (f (Ap (FMap (Cons batchSize)) resShapes))
batchModel fGen stateVars =
  let u = unsafePerformIO newUnique -- unique identifier for the batch dimension
      unbroadcastStates :: forall ss. SList ss -> HTV ty_ (Ap (FMap (Cons batchSize)) ss) -> HTV ty_ ss
      unbroadcastStates Unit Unit = Unit
      unbroadcastStates (_ :* ss) (F x :* xs) = F (Unbroadcast batchSize u x) :* unbroadcastStates ss xs
  in do xs <- genBatchedPlaceholders u batchSize (typeSList @shapesAndTypes)
        f <- fGen
        return $ broadcastGen u True (Proxy @batchSize) (f xs (unbroadcastStates (typeSList) stateVars))
 where batchSize = natSat @batchSize

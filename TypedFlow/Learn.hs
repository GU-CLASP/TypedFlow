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

module TypedFlow.Learn where

import Data.Proxy
import TypedFlow.Types
import TypedFlow.Types.Proofs (knownAppend, knownAppendS, (?>))
import TypedFlow.Abstract (Batched(..),defaultT,G(..),runBC,broadcastGen)
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

-- | A standard modelling function: (input value, gold value) ↦ (prediction, accuracy, loss).
-- input is the shape of the input.
-- output is the shape of the output (one element per individual loss and accuracy)
-- p is the shape of each output element.
-- g is the shape of each gold output --- often equal to p.
type Model input tIn g p output tOut
  = T input tIn -> T (g++output) tOut -> ModelOutput tOut p output

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
  in ModelOutput{..}

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
categoricalDistribution logits y =
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
      modelName = ""
  in ModelOutput{..}

-- | Model with @n@ binary outputs.
binary :: KnownNat n => Model '[n] Float32 '[] '[] '[n] Int32
binary logits y =
  let y_ = cast @Int32 (round sigy_)
      sigy_ = sigmoid logits
  in ModelOutput {modelY = y_
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
data StateAndOutput t p ss where
  StateAndOutput :: SList s -> ModelOutput t p s -> HTV t ss -> StateAndOutput t p (s ': ss)

unpairStateAndOutput :: StateAndOutput t p (s ': ss) -> (ModelOutput t p s, HTV t ss)
unpairStateAndOutput (StateAndOutput _ a b) = (a,b)

instance (KnownTyp t, KnownShape ps) => Batched (StateAndOutput t ps) where
  batchify :: forall n r. KnownNat n => All KnownShape r
    => Proxy n -> (forall s u. KnownTyp u => KnownShape s => T s u -> G (T (n:s) u))
    -> (StateAndOutput t ps) r  -> G ((StateAndOutput t ps) (Ap (FMap (Cons n)) r))
  batchify n f (StateAndOutput s ms xs) = StateAndOutput (n :* s) <$> (fromF <$> (h s (F ms)))  -- (hmapK @KnownShape (h s) ms)
                                                                  <*> (batchify n f xs)
    where h :: forall x s. KnownShape s => KnownShape x => SList s -> F (ModelOutput t) s x -> G (F (ModelOutput t) (n ': s) x)
          h s' (F ModelOutput{..}) = do
            modelLoss' <- f modelLoss
            modelY' <- knownAppendS s' (Proxy @x) ?> f modelY
            modelCorrect' <- f modelCorrect
            return $ F ModelOutput{modelLoss = modelLoss'
                         ,modelY = modelY'
                         ,modelName = modelName
                         ,modelCorrect = modelCorrect'}

-- | Name of a placeholder of a given shape and type.
data HolderName (st :: (Symbol,Shape,Typ)) = HolderName String

holderName :: forall (st :: (Symbol,Shape,Typ)) proxy. KnownSymbol (Frst3 st) => proxy st -> String
holderName _ = symbolVal (Proxy @(Frst3 st))

genBatchedPlaceholders :: All KnownPlaceholder shapesAndTypes
  => Unique -> Sat KnownNat n -> SList shapesAndTypes -> Gen (Placeholders shapesAndTypes)
genBatchedPlaceholders _ _ Unit = pure Unit
genBatchedPlaceholders u n@Sat (name :* names) = do
  x <- placeholderInternal (holderName name)
  xs <- genBatchedPlaceholders u n names
  return (PHT (Unbroadcast n u x) :* xs)


placeholderInternal :: ∀ (shape :: Shape) t. (KnownTyp t,KnownShape shape) => String -> Gen (T shape t)
placeholderInternal name = T . ExternalVar <$> GPVariable False name Nothing


knownCons :: KnownNat x => Sat KnownShape s -> Sat KnownShape (x ': s)
knownCons Sat = Sat

-- | Turn a stateless modelling function into a trivially stateful one.
stateless :: KnownLen s => (inputs -> ModelOutput t p s) -> inputs -> HTV t '[] -> StateAndOutput t p '[ s ]
stateless f x Unit = StateAndOutput typeSList (f x) Unit

simpleModel :: forall sx tx sy ty sy_ ty_ p. KnownLen sy_ => (Tensor sx tx -> Tensor sy ty -> ModelOutput  ty_ p sy_) ->
            (Placeholders '[ '("x",sx, tx), '("y",sy, ty)] -> HTV ty_ '[] -> (StateAndOutput ty_ p (sy_ ': '[])))
simpleModel f = stateless f'
  where f' :: Placeholders '[ '("x",sx,tx), '("y",sy,ty)] -> ModelOutput ty_ p sy_
        f' (PHT x :* PHT y :* Unit) = f x y

-- | @updateStates xs ys@ assigns to the tensor (variables!) xs the values ys.
updateStates :: forall xs ty. KnownTyp ty => All KnownShape xs => HTV ty xs -> HTV ty xs -> Gen (HTV ty xs)
updateStates Unit Unit = pure Unit
updateStates (F x :* xs) (F y :* ys) = (:*) <$> (F <$> modifyPersistent x y) <*> updateStates xs ys

-- | Add a term to the loss. This function is intendend to add
-- regularizers, ie. losses that do not depend on the predicted
-- output, but rather on the structure of a parameter.
addRegularizer :: Scalar Float32 -> Gen ()
addRegularizer r = GPState  $ \GState{..} -> ((),GState{genRegularizers=r:genRegularizers,..})

batchModel :: forall batchSize shapesAndTypes sy_ ty_ stateShapes ps.
           (KnownNat batchSize, KnownLen shapesAndTypes, All KnownPlaceholder shapesAndTypes, KnownLen stateShapes,
            All KnownShape stateShapes, KnownTyp ty_, KnownShape sy_, KnownShape ps)
         => Gen (Placeholders shapesAndTypes -> HTV ty_ stateShapes -> StateAndOutput ty_ ps (sy_ ': stateShapes))
         -> (NP (Sat KnownShape) (Ap (FMap (Cons batchSize)) stateShapes),
             Gen (HTV ty_ (Ap (FMap (Cons batchSize)) stateShapes), -- the state variables
                  HTV ty_ (Ap (FMap (Cons batchSize)) stateShapes) -> (ModelOutput ty_ ps (batchSize ': sy_), HTV ty_ (Ap (FMap (Cons batchSize)) stateShapes)) ))
batchModel f = let batchedShapesKnown = mapFMap @(Cons batchSize) knownCons (allKnown @KnownShape @stateShapes)
               in knownAll batchedShapesKnown
                  (batchedShapesKnown,precompile (batchModel' @batchSize f))

-- | Prepares the (already batched) model for compilation:
-- -- - create the state variables
precompile :: forall ps sy ty stateShapes.
              All KnownShape stateShapes
           => KnownLen stateShapes
           => (KnownShape sy, KnownShape ps, KnownTyp ty, KnownLen ps)
           => (Gen (HTV ty stateShapes -> StateAndOutput ty ps (sy ': stateShapes)))
           -> (Gen (HTV ty stateShapes,HTV ty stateShapes -> (ModelOutput ty ps sy, HTV ty stateShapes) ))
precompile f = do
  (stateVars :: HTV ty stateShapes) <- travTensor (persistent False) "state" (repeatT defaultT)
  f' <- fmap (unpairStateAndOutput . ) f
  return (stateVars,f')

-- -- - add training phase placeholder
-- -- - create the state variables
-- -- - compute final accuracy and loss (adding eventual regularizers), and expose them.
-- precompile :: forall ps sy ty stateShapes.
--               All KnownShape stateShapes
--            => KnownLen stateShapes
--            => (KnownShape sy, KnownShape ps, KnownTyp ty, KnownLen ps)
--            => (Gen (HTV ty stateShapes -> StateAndOutput ty ps (sy ': stateShapes)))
--            -> Gen (HTV ty stateShapes, (NP (K Task) ps)) -- (String, HTV ty stateShapes,Scalar Float32)
-- precompile model =
--  do regularizers <- genGets genRegularizers
--     trainingPhasePlaceholder <- placeholder "training_phase"
--     -- GPState $ \GState{..} -> ((),GState{genTrainingPlaceholder = trainingPhasePlaceholder,..})
--     -- (stateVars :: HTV ty stateShapes) <- travTensor (persistent False) "state" (repeatT defaultT)
--     stateAndOuputs <- model <*> travTensor (persistent False) "state" (repeatT defaultT)
--     -- (StateAndOutput _ models newStates) <- model stateVars
--     -- updates <- updateStates @stateShapes stateVars newStates
--     case stateAndOuputs of
--         StateAndOutput s (ModelOutput {..}) stateVars -> do
--           let taskLoss = SomeT (reduceMeanAll modelLoss ⊕ addN regularizers)
--               taskAccuracy = SomeT (reduceMeanAll (cast @Float32 modelCorrect))
--               taskPrediction = modelY
--               taskName = modelName
--           return (updates,Task {})

-- | Batch the model (adding one dimension), create placeholders for the inputs.
batchModel' :: forall batchSize shapesAndTypes resShapes ty_ stateShapes f.
           (KnownNat batchSize, KnownLen shapesAndTypes, All KnownPlaceholder shapesAndTypes, KnownLen stateShapes,
            All KnownShape stateShapes, KnownTyp ty_, All KnownShape resShapes, Batched f)
         => Gen (Placeholders shapesAndTypes -> HTV ty_ stateShapes -> f resShapes)
         -> Gen (HTV ty_ (Ap (FMap (Cons batchSize)) stateShapes) -> (f (Ap (FMap (Cons batchSize)) resShapes)))
batchModel' fGen =
  let u = -777 -- unique identifier for the batch dimension
      unbroadcastStates :: forall ss. SList ss -> HTV ty_ (Ap (FMap (Cons batchSize)) ss) -> HTV ty_ ss
      unbroadcastStates Unit Unit = Unit
      unbroadcastStates (_ :* ss) (F x :* xs) = F (Unbroadcast batchSize u x) :* unbroadcastStates ss xs
  in do xs <- genBatchedPlaceholders u batchSize (typeSList @shapesAndTypes)
        f <- fGen
        return $ \stateVars -> runBC u (broadcastGen u True (Proxy @batchSize) (f xs (unbroadcastStates (typeSList) stateVars)))
 where batchSize = natSat @batchSize


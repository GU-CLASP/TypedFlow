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
import qualified Prelude ()
import Prelude (($),return)
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
categorical :: forall n bs. KnownNat n => Model '[n,bs] Float32 '[bs] Int64
categorical logits' y = do
  logits <- assign logits'
  let y_ = argmax0 logits
  correctPrediction <- assign (equal y_ y)
  accuracy <- assign (reduceMeanAll (cast @Float32 correctPrediction))
  loss <- assign (reduceMeanAll (softmaxCrossEntropyWithLogits (oneHot y) logits))
  return (y_,accuracy,loss)

-- | First type argument is the number of classes.
-- @categoricalDistribution logits gold@
-- return (prediction, accuraccy, loss)
-- accuracy and prediction are averaged over the batch.
categoricalDistribution :: forall n bs. KnownNat n => Model '[n,bs] Float32 '[n,bs] Float32
categoricalDistribution logits' y = do
  logits <- assign logits'
  let y_ = softmax0 logits
  correctPrediction <- assign (equal (argmax0 logits) (argmax0 y))
  accuracy <- assign (reduceMeanAll (cast @Float32 correctPrediction))
  loss <- assign (reduceMeanAll (softmaxCrossEntropyWithLogits y logits))
  return (y_,accuracy,loss)


type Scalar t = T '[] t


-- | (input value, gold value) ↦ (prediction, accuracy, loss)
type Model input tIn output tOut = T input tIn -> T output tOut -> Gen (T output tOut, Scalar Float32, Scalar Float32)


binary :: forall bs. (KnownNat bs) => Model '[bs] Float32 '[bs] Int32
binary score y = do
  sigy_ <- assign (sigmoid score)
  let y_ = cast @Int32 (round sigy_)
  correctPrediction <- assign (equal y_ y)
  accuracy <- assign (reduceMeanAll (cast @Float32 correctPrediction))
  loss <- assign (reduceMeanAll (binaryCrossEntropy (cast @Float32 y) sigy_))
  return (y_,accuracy,loss)


compile :: (KnownShape input, KnownTyp tIn, KnownShape output, KnownTyp tOut) =>
           Model input tIn output tOut  -> Gen ()
compile model = do
  gen (text "import tensorflow as tf")
  genFun "mkModel" [] $ do
    x <- placeholder "x"
    y <- placeholder "y"
    (prediction,accuracy,loss) <- model x y
    y_ <- assign prediction
    gen (text "return " <> tuple [fromTensor x,fromTensor y,fromTensor y_,fromTensor accuracy,fromTensor loss])


-- Local Variables:
-- dante-project-root: ".."
-- End:

{-# LANGUAGE NamedFieldPuns #-}
{-|
Module      : TypedFlow.Python.Top
Description : Python-generation Functions 
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental

-}

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeInType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module TypedFlow.Python.Top where

import GHC.TypeLits
import Control.Monad.State
import TypedFlow.Types
import TypedFlow.Types.Proofs
import TypedFlow.Learn
import TypedFlow.Python
import Text.PrettyPrint.Compact hiding (All,Last,Product,Sum,Options)

-- | Batchify and compile a model with simple input to output mapping.
compile :: forall batchSize sx tx sy ty sy_ ty_ p.
           (KnownNat batchSize, KnownShape sx, KnownTyp tx, KnownShape sy, KnownTyp ty, KnownShape sy_, KnownTyp ty_, KnownShape p) =>
           Options -> Gen (Tensor sx tx -> Tensor sy ty -> ModelOutput  ty_ p sy_)
           -- Model input tIn output tOut
        -> Python ()
compile options fGen = do
  compileGen @batchSize options (HolderName "x" :* HolderName "y" :* Unit) $ do
    f <- fGen
    let f' :: HHTV '[ '(sx,tx), '(sy,ty)] -> ModelOutput ty_ p sy_ 
        f' (Uncurry x :* Uncurry y :* Unit) = f x y
    return (stateless f')

compileGen :: forall batchSize shapesAndTypes sy_ ty_ p stateShapes.
           (KnownNat batchSize, All KnownPair shapesAndTypes, KnownLen stateShapes,
            All KnownShape stateShapes, KnownShape sy_, KnownTyp ty_, KnownShape p)
         => Options
         -> SList' HolderName shapesAndTypes -- ^ names for the inputs
         -> Gen (HHTV shapesAndTypes -> HTV ty_ stateShapes -> (StateAndOutput ty_ p (sy_ ': stateShapes)) )
         -> Python ()
compileGen options names fGen =
  let batchedShapesKnown = mapFMap @(Cons batchSize) knownCons (allKnown @KnownShape @stateShapes typeSList)
  in knownAll batchedShapesKnown $
     compileAlreadyBatched @batchSize options (precompile @batchSize (batchModel names fGen))

-- | Generic model preparation, with non-standard parameters ("x", "y"
--  must be provided as placeholders manually).
compileAlreadyBatched :: forall bs ty stateShapes. KnownNat bs
           => KnownTyp ty
           => All KnownShape stateShapes
           => Options
           -> (Gen (HTV ty stateShapes,Scalar Float32)) -> Python ()
compileAlreadyBatched Options{..} model = do
  gen (text "import tensorflow as tf")
  genFun "mkModel" [text "optimizer=tf.train.AdamOptimizer()"] $ do
    (updates,lossIn) <- interpGen model
    loss <- generatePure lossIn
    params <- getParameters
    trainStep <- assignAny $ case maxGradientNorm of
      Nothing -> funcall "optimizer.minimize" [loss]
      Just clip -> funcall "optimizer.apply_gradients" [funcall "zip" [clipByGlobalNorm clip (grad loss params),params]]
    peeks <- mapM paramToPeek =<< gets genPeeks
    updates' <- untypedExprs updates
    let peeks2 = [("optimizer", (text "optimizer"))
                 ,("batch_size", (showDim @ bs))
                 ,("params", params)
                 ,("train", trainStep)
                 ,("update", list updates')
                 ]
    gen (text "return " <> dict (peeks ++peeks2))

paramToPeek :: ParamInfo -> Python (String,UntypedExpression)
paramToPeek (ParamInfo name s t x) = do
  x' <- knownSShape s $ knownTyp t $ generatePure x
  return (name,x')

untypedExprs :: All KnownShape xs => KnownTyp t =>  HTV t xs -> Python [DOC]
untypedExprs Unit = return []
untypedExprs (F x :* xs) = (:) <$> generatePure x <*> untypedExprs xs




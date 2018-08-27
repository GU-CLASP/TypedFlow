{-|
Module      : TypedFlow.Haskell
Description : Generation of computation graph using tensorflow haskell. 
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

module TypedFlow.Haskell where

import Data.Type.Equality
import Data.List (genericReplicate)
import GHC.TypeLits
import Control.Monad.State
import TypedFlow.Types
import TypedFlow.Types.Proofs
import TypedFlow.Abstract (newId, permToFun, unopInputShape)
import TypedFlow.Memo
import System.Mem.StableName
import System.IO.Unsafe

import qualified Data.Int as Backend

import qualified TensorFlow.Core        as Backend
import qualified TensorFlow.GenOps.Core as BackCore
import qualified TensorFlow.Minimize    as Backend
import qualified TensorFlow.Ops         as Backend
import qualified TensorFlow.NN          as Backend
-- import qualified TensorFlow.Variable    as Backend
import qualified TensorFlow.Tensor

import qualified Data.IntMap as IM
import Data.IntMap (IntMap)

type BackendShape = BackendTensor ('Typ 'Int 'B32)
type BackendTensor t = Backend.Tensor Backend.Build (HaskType t)
type BackendVariable t = Backend.Tensor Backend.Ref (HaskType t)
type BackendTensorType t = Backend.TensorType (HaskType t)

shapeFromType :: ∀ (s :: Shape). KnownShape s => BackendShape
shapeFromType = shapeVector (typeSShape @s)

-- | Show a shape, but "None" is replaced by "-1"
shapeVector :: forall (s::Shape) proxy. All KnownNat s => SList' proxy s -> BackendShape
shapeVector s = shapeFromList (shapeToList'' s)

permToTensor :: SShape s -> Permutation s t -> Backend.Tensor Backend.Build Backend.Int32
permToTensor s p = Backend.vector (map (fromInteger . permToFun p) [0.. sListLength s])

shapeFromList :: [Integer] -> BackendShape
shapeFromList = Backend.vector . map convertNone

showShapeLen :: ∀ (s::Shape). KnownLen s => Backend.Int32
showShapeLen = fromIntegral (listTypeLen @ s)

convertNone :: Num a => Integer -> a
convertNone n = (if n == 514229 then (-1) else fromIntegral n)

-- runWithFeeds

data BT (s :: Shape) (t :: Typ) where
  BT :: forall s t. (BackendTensor t) -> BT s t

data HState = HState {genVars :: IntMap Var
                     ,genPureTable :: SNMap22 Shape Typ T BT
                     -- alternative: use tensorRefFromName and make this closer to the python backed.
                     }

type BM a = Backend.BuildT (StateT HState (State GState)) a

data Var = forall s t v. TensorFlow.Tensor.TensorKind v => Var (SShape s) (STyp t) (Backend.Tensor v (HaskType t))

initializedVariable :: forall s a. KnownShape s => KnownTyp a => T s a -> BM (Ref s a)
initializedVariable initVal = do
  BT i <- interpretPure initVal
  x <- lift (lift newId)
  v <- backendTensor (typeSTyp @a) $ Backend.initializedVariable i
  let var = (Var (typeSShape @s) (typeSTyp @a) v)
  lift (modify $ \HState{..} -> HState {genVars = IM.insert (fromIntegral x) var genVars,..})
  return (Ref (fromIntegral x) typeSShape typeSTyp )

placeholder :: forall s a. SShape s -> STyp a -> BM (Ref s a)
placeholder s t = do
  x <- lift (lift newId)
  ph <- backendTensor t $ Backend.placeholder (Backend.Shape (map convertNone $ shapeToList' s))
  let var = (Var s t ph)
  lift (modify $ \HState{..} -> HState {genVars = IM.insert (fromIntegral x) var genVars,..})
  return (Ref (fromIntegral x) s t )

interpGen :: Gen a -> BM a
interpGen (GPReturn x) = return x
interpGen (GPVariable _trainable _name initVal) = initializedVariable initVal
interpGen (GPPlaceholder s t _name) = placeholder s t
interpGen (GPModify _ _) = error "GPModify: TODO"
interpGen (GPState f) = lift (lift (state f))
interpGen (GPBind a b) = do x <- interpGen a
                            interpGen (b x)

listProxyLen :: forall proxy s. KnownLen s => proxy s -> Integer
listProxyLen _ = listTypeLen @s

-- genDistr :: forall s s0 t. KnownTyp t => Distribution s t -> SShape s0 -> SShape s -> DOC
-- genDistr d sh s1 = case d of
--   TruncatedNormalD stddev -> funcall "tf.truncated_normal"
--     [showSShape (sh .+. s1), named "stddev" (float stddev), named "dtype" (showTyp @t)]
--   UniformD low high -> funcall "tf.random_uniform" [showSShape (sh .+. s1)
--                                 ,named "minval" (float low)
--                                 ,named "maxval" (float high)
--                                 ,named "dtype" (showTyp @t)]
--   OrthogonalD ->
--     funcall' (funcall "tf.orthogonal_initializer" [named "dtype" (showTyp @t)]) [named "shape" (showSShape (sh .+. s1))]


knownNumeric :: forall t k. KnownNumeric t => (KnownTyp t => Num (HaskType t) => Backend.OneOf '[Backend.Int32, Float, Double] (HaskType t) => k) -> k
knownNumeric = knownNumeric' (typeSTyp @t)

knownNumeric' :: forall t k. KnownNumeric t => STyp t -> (KnownTyp t => Num (HaskType t) => Backend.OneOf '[Backend.Int32, Float, Double] (HaskType t) => k) -> k
knownNumeric' (STyp tk tb Refl) k = case tk of
  SFloat -> case tb of
    SB32 -> k
    SB64 -> k
  SBool -> error "TFNumeric bug"
  SInt -> case tb of
    SB32 -> k
    SB64 -> error "missing in tensorflow: int64 is not supported in matmul T_T"

knownFloatingB :: forall t k. (KnownTyp t, TypKind t ~ 'Float) => (Backend.OneOf '[Float, Double] (HaskType t) => k) -> k
knownFloatingB k = case bitsVal @(TypBits t) of
    SB32 -> k
    SB64 -> k

knownInt :: forall t k. (KnownTyp t, TypKind t ~ 'Int) => (Backend.OneOf '[Backend.Int32, Backend.Int64] (HaskType t) => k) -> k
knownInt k = case bitsVal @(TypBits t) of
    SB32 -> k
    SB64 -> k

backendTensor :: STyp t ->  (Backend.TensorType (HaskType t) => k) -> k
backendTensor (STyp SFloat SB32 Refl) k = k
backendTensor (STyp SInt SB64 Refl) k = k
backendTensor (STyp SBool _ Refl) k = k
backendTensor (STyp SFloat SB64 Refl) k = k
backendTensor (STyp SInt SB32 Refl) k = k

backendTensor' :: forall t k proxy. KnownTyp t => proxy t -> (Backend.TensorType (HaskType t) => k) -> k
backendTensor' _ = backendTensor (typeSTyp @t)


runUnOp :: forall s s1 t s2 u. KnownTyp u => KnownTyp t => BackendTensorType u => SShape s -> UnOp s1 t s2 u -> BT (s++s1) t -> BT (s++s2) u
runUnOp sL op (BT x) = backendTensor (typeSTyp @t) $ case op of
  SliceOp _ sR lo hi -> BT $ BackCore.slice x
    (shapeFromList (replicate (sListLen  sL) 0 ++ [lo] ++ replicate (sListLen sR) 0))
    (shapeFromList (shapeToList' sL ++ [hi-lo] ++ (shapeToList' sR)))
  Axis1Op aop -> case aop of
    (ArgMax _ _) -> knownNumeric @t $ knownInt @u $ BT $ BackCore.argMax x (Backend.scalar sLLen)
    (OneHot _) -> knownNumeric @u $ knownInt @t $  BT $ Backend.oneHot x (Backend.scalar sLLen) (Backend.scalar 1) (Backend.scalar 0)
    ReduceOp _ _sR rop -> knownNumeric @t $ case rop of
      Max -> BT $ BackCore.max x redindices
      Min -> BT $ BackCore.min x redindices
      Sum -> BT $ Backend.sum x redindices
      Mean -> BT $ Backend.mean x redindices
     where redindices = (Backend.vector [fromIntegral (sListLen sL) :: Backend.Int32 ])
  StopGradient -> BT $ BackCore.stopGradient x
  Cast -> BT $ Backend.cast x
  (Num1Op numop) -> knownNumeric @t $ case numop of
    Square -> BT (Backend.mul x x)
    Negate -> BT (Backend.neg x)
    Sign -> BT (Backend.sign x)
    Abs -> BT (Backend.abs x)
  Float1Op flop -> knownFloatingB @t $ knownFloating @(TypBits u) $ knownFloatingB @u $ case flop of
     Tanh -> BT (BackCore.tanh x)
     Sin -> BT (BackCore.sin x)
     Exp -> BT (BackCore.exp x)
     Sigmoid -> BT (BackCore.sigmoid x)
     Relu -> BT (BackCore.relu x)
     Floor -> BT (BackCore.floor x)
     Round -> BT (BackCore.round x)
     Cos -> BT (BackCore.cos x)
     Log -> BT (BackCore.log x)
     Asin -> BT (BackCore.asin x)
     Acos -> BT (BackCore.acos x)
     Sinh -> BT (BackCore.sinh x)
     Cosh -> BT (BackCore.cosh x)
     Asinh -> BT (BackCore.asinh x)
     Acosh -> BT (BackCore.acosh x)
     Atan -> BT (BackCore.atan x)
     Atanh -> BT (BackCore.atanh x)
     Sqrt -> BT (BackCore.sqrt x)
     HardSigmoid -> error "Haskell: no hard sigmoid defined yet"
     ClipByValue lo hi -> BT $ BackCore.clipByValue x (Backend.scalar $ realToFrac lo) (Backend.scalar $ realToFrac hi)
  Diag _ -> BT $ BackCore.batchMatrixDiag x
 where sLLen = fromIntegral (sListLen sL) :: Backend.Int32

interpretPure :: forall s t. KnownTyp t => KnownShape s => T s t -> BM (BT s t)
interpretPure x = do
  let sn = unsafePerformIO $ makeStableName x
  mv <- snMap22Lookup sn <$> lift (gets genPureTable)
  case mv of
    Just v -> return v
    Nothing -> do
      e  <- interpretPure' (\s x' -> knownSShape s $ interpretPure x') typeSShape x
      lift $ modify (\g -> g {genPureTable = (snMap22Insert (KV sn e)) (genPureTable g)})
      return e

interpNilOp :: forall s t. Backend.TensorType (HaskType t) => NilOp s t -> BM (BT s t)
interpNilOp = \case
  Constant c -> return $ BT $ Backend.scalar c
  Range n@Sat -> knownNumeric @t $ return $
    let start,limit,delta :: HaskType t
        start = 0
        limit = fromIntegral $ natVal n
        delta = 1
    in BT $ Backend.range (Backend.scalar start) (Backend.scalar limit) (Backend.scalar delta)
  Variable (Ref r sr tr) -> do
     tbl <- lift (gets genVars)
     case IM.lookup r tbl of
       Just (Var sx tx x) -> case (testEq sx sr, testEq tx tr) of
          (Just Refl, Just Refl) -> return (BT (Backend.expr x))
          _ -> error "panic: variable does not have the expected type"
       _ -> error "panic: variable not found" 

interpretPure' :: forall s t. KnownTyp t => (forall s' t'. KnownTyp t' => SShape s' -> T s' t' -> BM (BT s' t')) -> SShape s -> T s t -> BM (BT s t)
interpretPure' rec sR = knownSShape sR $ backendTensor (typeSTyp @t) $ \case
  Unbroadcast{} -> error "broadcasting operation did not complete!"
  DirectBroadcast s0 s1 s2 s3 x -> do
    BT recx <- rec (s0 .+. s2) x
    let expandedShape = shapeFromList
                          (concat [shapeToList' s0, genericReplicate (sListLength s1) 1
                                  ,shapeToList' s2, genericReplicate (sListLength s3) 1 ])
        targetShape = shapeFromList sR
    return $ BT $ BackCore.broadcastTo (Backend.reshape recx expandedShape) targetShape
   --  Noise noiseId s0 s1 x -> do
   --    return $ (genDistr x s0 s1) <+> (text "# " <> integer noiseId)
  T op -> interpNilOp op
  Where c x y -> do
    BT rc <- rec typeSShape c
    BT rx <- rec typeSShape x
    BT ry <- rec typeSShape y
    return $ BT $ BackCore.select rc rx ry
  UnOp operation s0 x -> do
    recx <- rec (s0 .+. unopInputShape operation) x
    return (runUnOp s0 operation recx)
  MatMul s0 a b c x y  -> do
    BT recx <- rec (s0 .+. a :* b :* Unit) x
    BT recy <- rec (s0 .+. b :* c :* Unit) y
    return $ knownNumeric @t $ BT $ BackCore.batchMatMul recx recy
  BinOp operation s0 s1 t s2 u x y -> knownSShape s0 $ knownSShape s1 $ knownSShape s2 $ knownProduct' s0 $ do
   BT recx <- rec (s0 .+. s1) x
   BT recy <- rec (s0 .+. s2) y
   let reshx = backendTensor t $ Backend.reshape recx (shapeVector (satProd s0 :* s1))
       reshy = backendTensor u $ Backend.reshape recy (shapeVector (satProd s0 :* s2))
   return $ case operation of
     Simple2Op sop  -> case sop of
        Add -> knownNumeric @t $ BT $ Backend.add recx recy
        Divide -> knownNumeric @t $ BT $ BackCore.div recx recy
        Equal -> backendTensor u $ BT $ Backend.equal recx recy
        Subtract -> knownNumeric @t $ BT $ Backend.sub recx recy
        Multiply -> knownNumeric @t $ BT $ Backend.mul recx recy
        Minimum -> knownNumeric @t $ BT $ BackCore.minimum recx recy
        Maximum -> knownNumeric @t $ BT $ BackCore.maximum recx recy
        LessThan ->  knownNumeric' u $ BT $ BackCore.less recx recy
     -- WTF moment: the arguments do not seem to be in the same order in python as in haskell
     -- python: https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
     -- haskell: https://tensorflow.github.io/haskell/haddock/tensorflow-core-ops-0.2.0.0/TensorFlow-GenOps-Core.html#v:sparseSoftmaxCrossEntropyWithLogits
     SparseSoftmaxCrossEntropyWithLogits -> case t of
        STyp SInt SB32 Refl -> knownFloatingB @t $ BT $ fst $ BackCore.sparseSoftmaxCrossEntropyWithLogits reshy reshx
     SoftmaxCrossEntropyWithLogits -> knownFloatingB @t $ BT $ fst $ BackCore.softmaxCrossEntropyWithLogits reshy reshx
     -- SigmoidCrossEntropyWithLogits -> knownFloatingB @t $ BT $ Backend.sigmoidCrossEntropyWithLogits recy recx -- type is not as general as necessary
  ReshapeFrom s t -> do
    BT rt <- rec s t
    return $ BT $ BackCore.reshape rt (shapeVector sR)
  Concat s0 s1 xs -> do
    let go :: forall s0 s1 ns. SShape s0 -> SShape s1 -> NP (Catable s0 s1 t) ns -> BM [BackendTensor t]
        go _ _ Unit = return []
        go s0' s1' (Catable n y :* ys) = do
          BT y' <- rec (s0' .+. n :* s1') y
          (y' :) <$> go s0' s1' ys
    rxs <- go s0 s1 xs
    return $ BT $ Backend.concat (Backend.scalar (fromIntegral (sListLength s0))) rxs
  Transpose s p x -> do
    BT rx <- rec s x
    return $ BT $ Backend.transpose rx (permToTensor s p)
 --  Gather indexShape s0 m s1 x ix -> do
 --    rx <- rec (s0 .+. ((:*) m s1)) x
 --    rix <- rec indexShape ix
 --    return (func "tf.gather" [rx, rix] [])
 --  GatherND containerShape elementShape indexShape x ix -> do
 --    rx <- rec (containerShape .+. elementShape) x
 --    rix <- rec (indexShape *: (sListLenAsNat containerShape)) ix
 --    return (func "tf.gather_nd" [rx, rix] [])
  Convolution bs inChans outChans filterShape s0 x filters -> do
    BT recx <- rec (bs :* (s0 *: inChans)) x
    BT recFilters <- rec (filterShape .+. inChans :* outChans :* Unit) filters
    case filterShape of
       _width :* _height :* Unit ->
          return $ BT $ knownFloatingB @t $ BackCore.conv2D recx recFilters
       _ -> error "TypedFlow Haskell backend: convolution on an unsupported number of dims"
 --  Pool bs window typ numChans outSpatial x -> do
 --     rx <- rec ((:*) bs (zipWithMulSShapes window outSpatial .+. (:*) numChans Unit)) x
 --     return (func "tf.nn.pool"
 --                  [rx, showSShape window, typ', text (show ("SAME" :: String))]
 --                  [("strides", showSShape window)])
 --   where typ' = text $ (show $ case typ of MaxPool -> "MAX"; AvgPool -> "AVG" :: String)
 -- -- where rec :: forall s' t'. KnownTyp t' => SShape s' -> T s' t' -> DOC
 -- --       rec = generatePure' 


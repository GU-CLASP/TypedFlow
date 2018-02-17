{-|
Module      : TypedFlow.Python
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

module TypedFlow.Python where

import Data.Proxy
import GHC.TypeLits
import Control.Monad.State
import TypedFlow.Types
import TypedFlow.Memo
import Text.PrettyPrint.Compact hiding (All,Last,Product,Sum)
import qualified Data.Map as M

generateFile :: String -> Gen () -> IO ()
generateFile fname g = do
  putStrLn ("Parameters (total " ++ show (sum [product paramShape | ParamInfo{..} <- params]) ++ "):")
  forM_ params printParam
  writeFile fname output
  where (output,params) = generate g
        printParam ParamInfo{..} = putStrLn (paramName ++ ": " ++ "T " ++ render (showShape' paramShape)  ++ " " ++ show paramDType)

named :: String -> DOC -> DOC
named fname x = text (fname <> "=") <> x

genFun :: forall b. String -> [DOC] -> Gen b -> Gen b
genFun name args body = do
  gen (text "def " <> text name <> tuple args <> text ":")
  withDOC (\b -> text "  " <> b) body


showTyp :: forall t. KnownTyp t => DOC
showTyp = text (show (typVal @t))

showShape' ::  [Integer] -> DOC
showShape' s = list (map (showDim' "None") s)

showShape :: ∀ (s :: Shape). All KnownNat s => SList s -> DOC
showShape s = showShape' (shapeToList'' s)

showSShape :: ∀ (s :: Shape). SShape s -> DOC
showSShape s = showShape' (shapeToList' s)

showShapeType :: ∀ (s :: Shape). KnownShape s => DOC
showShapeType = showSShape (typeSShape @s)

-- | Show a shape, but "None" is replaced by "-1"
showShapeMinus :: forall (s::Shape) proxy. All KnownNat s => SList' proxy s -> DOC
showShapeMinus s = list (map (showDim' "-1") (shapeToList'' s))

showShapeLen :: ∀ (s::Shape). KnownLen s => DOC
showShapeLen = (text . show) (listTypeLen @ s)

showDim' :: String -> Integer -> DOC
showDim' none n = text (if n == 514229 then none else show n)

showDimM :: forall n. KnownNat n => DOC
showDimM = showDim' "-1" (natVal (Proxy @ n))

showDim :: forall n. KnownNat n => DOC
showDim = showDim' "None" (natVal (Proxy @ n))

str :: Show a => a -> DOC
str = text . show

newVar :: Gen DOC
newVar = do
  n <- gets nextVar
  modify $ \GState{..} -> GState {nextVar=nextVar+1,..}
  return (text "var" <> integer n)

gen :: DOC -> Gen ()
gen s = modify $ \GState{..} -> GState {genText=genText $$ s,..}

setGen :: DOC -> Gen ()
setGen d = modify $ \GState{..} -> GState {genText=d,..}

(<--) :: DOC -> UntypedExpression -> Gen ()
x <-- y = gen (x <> text "=" <>  y)

cache :: DOC -> Gen DOC
cache x = do
  let x' = renderWith (Options 92 (const id)) x
  mcache <- M.lookup x' <$> gets genAssignTable
  case mcache of
    Just y -> return y
    Nothing -> do
      v <- newVar
      modify (\g -> g {genAssignTable = M.insert x' v (genAssignTable g)})
      v <-- x
      return v

tuple :: [DOC] -> DOC
tuple = parens . sep . punctuate comma

dict :: [(String,DOC)] -> DOC
dict xs = encloseSep "{" "}" "," [text (show k) <> ":" <> v | (k,v) <- xs]

funcall :: String -> [DOC] -> DOC
funcall = funcall' . text

funcall' :: DOC -> [DOC] -> DOC
funcall' f args = hangWith "" 2 (f <> "(") (as <> ")")
  where as = sep (punctuate comma args)

func :: String -> [DOC] -> [(String,DOC)] -> DOC
func fname positional namedArgs = funcall fname (positional ++ map (uncurry named) namedArgs )

withDOC :: forall a. (DOC -> DOC) -> Gen a -> Gen a
withDOC f g = do
  before <- gets genText
  setGen mempty
  x <- g
  after <- gets genText
  setGen (before $$ f after)
  return x

newParameter :: MonadState GState m => ParamInfo -> m ()
newParameter p =   modify $ \GState{..} -> GState{genParams = p:genParams,..}

-- | Name an expression so that it is made available for session.run.
peekAtAny :: String -> UntypedExpression -> Gen ()
peekAtAny p v = modify $ \GState{..} -> GState{genPeeks = if p `elem` map fst genPeeks then error ("duplicate name: " ++ p) else (p,v):genPeeks,..}


assign :: ∀s t. (KnownShape s, KnownTyp t) => T s t -> Gen (T s t)
assign x = do
  e <- generatePure x
  return (T e)

assignAny :: UntypedExpression -> Gen UntypedExpression
assignAny x = do
  v <- newVar
  v <-- x
  return v

lambda :: (T s t -> T s' t') -> Gen UntypedExpression
lambda f = do
  v <- newVar
  let T body = f (T v)
  return (text "lambda " <> v <> ": " <> body)

generate :: Gen () -> (String,[ParamInfo])
generate s = (renderWith (Options 92 (const id)) genText,genParams)
  where GState{..} =  execState (fromGen s) (GState {nextVar = 0
                                                    ,genText = mempty
                                                    ,genParams=[]
                                                    ,genRegularizers=[]
                                                    ,genTrainingPlaceholder = T "NO TRAINING PLACEHOLDER!"
                                                    ,genPureTable = mempty
                                                    ,genAssignTable = mempty
                                                    ,genPeeks=[]})

-- FIXME: sharing


permToFun :: Permutation s t -> Integer -> Integer
permToFun = \case
  PermId -> \x -> x
  PermTrans a b -> permToFun b . permToFun a
  PermSwap -> \case
    0 -> 1
    1 -> 0
    x -> x
  PermSkip p -> \case
    0 -> 0
    x -> permToFun p (x-1) Prelude.+ 1


listProxyLen :: forall proxy s. KnownLen s => proxy s -> Integer
listProxyLen _ = listTypeLen @s

generatePure :: forall s t. KnownTyp t => KnownShape s => T s t -> Gen DOC
generatePure x = do
  let sn = makeSn2 x
  mv <- snMapLookup2 sn <$> gets genPureTable
  case mv of
    Just v -> return v
    Nothing -> do
      e <- generatePure' (\s x' -> knownSShape s $ generatePure x') typeSShape x
      modify (\g -> g {genPureTable = (snMapInsert2 sn e) (genPureTable g)})
      cache e

generatePure' :: forall s t. KnownTyp t => (forall s' t'. KnownTyp t' => SShape s' -> T s' t' -> Gen DOC) -> SShape s -> T s t -> Gen DOC
generatePure' rec sR = knownSShape sR $ \case
  T x -> return x
  If c x y -> do
    rc <- rec typeSShape c
    rx <- rec typeSShape x
    ry <- rec typeSShape y
    return (func "tf.cond" [rc] [("true_fn", lambda0 rx) ,("false_fn", lambda0 ry) ,("strict","True")])
    where lambda0 z = text "lambda: " <> z
  Where c x y -> do
    rc <- rec typeSShape c
    rx <- rec typeSShape x
    ry <- rec typeSShape y
    return (funcall "tf.where" [rc, rx, ry])
  UnOp operation s0 s1 _s2 x -> do
   recx <- rec (s0 .+. s1) x
   return $ case operation of
    SimpleBroadCast n ->
    -- Nicer implementation upcoming?
    -- https://github.com/tensorflow/tensorflow/pull/15243
    -- https://github.com/tensorflow/tensorflow/issues/14509
    -- TODO: do not do the "add zero" part if the context is a broadcastable operation
      funcall "tf.add" [func "tf.expand_dims" [recx] [("axis", integer (n + sListLength s0))],
                         func "tf.zeros" [showSShape s0] [("dtype", showTyp @t)]]
    Axis1Op op args n -> func op [recx] (("axis",integer (sListLength s0 + n)):args)
    Simple1Op op args -> funcall op (recx:args)
    SliceOp lo hi -> recx <> list (replicate (fromIntegral (sListLength s0)) (text ":") ++ [integer lo <> text ".." <> integer hi])
    IndexOp axis ix -> recx <> list (replicate (fromIntegral (axis + sListLength s0)) (text ":") ++ [integer ix])
  BinOp operation s0 s1 s2 _s3 x y -> do
   recx <- rec (s0 .+. s1) x
   recy <- rec (s0 .+. s2) y
   return $ case operation of
     Axis2Op op n -> funcall op  [list [recx,recy], named "axis" (integer (sListLength s0 + n))]
     Simple2Op op Nothing -> funcall op [recx, recy]
     Simple2Op op (Just (nx,ny)) -> func op [] [(nx,recx), (ny,recy)]
  ReshapeFrom s t -> do
    rt <- rec s t
    return (funcall "tf.reshape" [rt, knownSShape sR (showShapeMinus sR)])
  Stack s0 _m s1 (V xs) -> do
    rxs <- mapM (rec (s0 .+. s1)) xs
    return (funcall "tf.stack" [list rxs, text "axis=" <> integer (sListLength s0)])
  Transpose s p x -> do
    rx <- rec s x
    return (func "tf.transpose" [rx] [("perm",list (map (integer . permToFun p) [0.. sListLength s]))])
  Gather indexShape s0 m s1 x ix -> do
    rx <- rec (s0 .+. (LS m s1)) x
    rix <- rec indexShape ix
    return (func "tf.gather" [rx, rix] [])
  Convolution bs inChans outChans filterShape s0 x filters -> do
    recx <- rec (LS bs (sl s0 inChans)) x
    recFilters <- rec (filterShape .+. (LS inChans (LS outChans LZ))) filters
    return (func "tf.nn.convolution" [recx, recFilters] [("padding",text (show ("SAME"::String))),("data_format", text (show dataFormat))])
   where dataFormat = case sListLength filterShape of
           1 -> ("NWC" :: String)
           2 -> "NHWC"
           3 -> "NDHWC"
           _ -> error "convolution: more than 3 spatial dimensions are not supported!"
  Pool bs window typ numChans outSpatial x -> do
     rx <- rec (LS bs (zipWithMulSShapes window outSpatial .+. LS numChans LZ)) x
     return (func "tf.nn.pool" [rx, showSShape window, typ'] [])
   where typ' = text $ (show $ case typ of MaxPool -> "MAX"; AvgPool -> "AVG" :: String)
 -- where rec :: forall s' t'. KnownTyp t' => SShape s' -> T s' t' -> DOC
 --       rec = generatePure' 


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
import Text.PrettyPrint.Compact hiding (All,Last,Product,Sum)


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
showShape' s = list (map (showDim' "None") (reverse s))

showShape :: ∀ (s :: Shape). KnownShape s => DOC
showShape = showShape' (shapeToList @s)

-- | Show a shape, but "None" is replaced by "-1"
showShapeMinus :: ∀ (s :: Shape). KnownShape s => DOC
showShapeMinus = list (map (showDim' "-1") (reverse (shapeToList @ s)))

showShapeLen :: ∀ (s::Shape). KnownLen s => DOC
showShapeLen = (text . show) (listLen @ s)

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


binOp :: ∀ s1 s2 s3 t1 t2 t3. String -> Tensor s1 t1 -> Tensor s2 t2 -> Tensor s3 t3
binOp op (T x) (T y) = T (funcall op [ x , y])

unOp :: ∀ s1 s2 t1 t2. String -> Tensor s1 t1 -> Tensor s2 t2
unOp op (T x) = T (funcall op [x])

assign :: ∀s t. T s t -> Gen (T s t)
assign (T x) = do
  v <- newVar
  v <-- x
  return (T v)

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
                                                    ,genPeeks=[]})

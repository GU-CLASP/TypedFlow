{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}
{-|
Module      : TypedFlow.Layers.RNN.Cells
Description : RNN cells
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental
-}


module TypedFlow.Layers.RNN.Cells (
  -- * RNN Cells
  cellInitializerBit,
  LSTMP(..),
  lstm,
  GRUP(..),
  gru,
  StackP(..),
  stackRU,
  ) where

import TypedFlow.Layers.RNN.Base
import TypedFlow.TF
import TypedFlow.Types
import GHC.TypeLits
import TypedFlow.Layers.Core (DenseP(..),(#))
import Prelude hiding (tanh,Num(..),Floating(..),floor)
import Data.Monoid ((<>))

--------------------------------------
-- Cells

-- | Standard RNN gate initializer. (The recurrent kernel is
-- orthogonal to avoid divergence; the input kernel is glorot)
cellInitializerBit :: ∀ n x t. (KnownNat n, KnownNat x, KnownBits t) => DenseP t (n + x) n
cellInitializerBit = DenseP (concat0 recurrentInitializer kernelInitializer) biasInitializer
  where recurrentInitializer :: Tensor '[n, n] ('Typ 'Float t)
        recurrentInitializer = randomOrthogonal
        kernelInitializer :: Tensor '[x, n] ('Typ 'Float t)
        kernelInitializer = glorotUniform
        biasInitializer = zeros

-- | Parameter for an LSTM
data LSTMP t n x = LSTMP (DenseP t (n+x) n) (DenseP t (n+x) n) (DenseP t (n+x) n) (DenseP t (n+x) n)

instance (KnownNat n, KnownNat x, KnownBits t) => KnownTensors (LSTMP t n x) where
  travTensor f s (LSTMP x y z w) = LSTMP <$> travTensor f (s<>"_f") x <*> travTensor f (s<>"_i") y <*> travTensor f (s<>"_c") z <*> travTensor f (s<>"_o") w
instance (KnownNat n, KnownNat x, KnownBits t) => ParamWithDefault (LSTMP t n x) where
  defaultInitializer = LSTMP forgetInit cellInitializerBit cellInitializerBit cellInitializerBit
    where forgetInit = DenseP (denseWeights cellInitializerBit) ones

-- | Standard LSTM
lstm :: ∀ n x bs t. LSTMP t n x ->
        RnnCell t '[ '[n,bs], '[n,bs]] (Tensor '[x,bs] (Flt t)) (Tensor '[n,bs] (Flt t))
lstm (LSTMP wf wi wc wo) input (VecPair ht1 ct1) = do
  hx <- assign (concat0 ht1 input)
  let f = sigmoid (wf # hx)
      i = sigmoid (wi # hx)
      cTilda = tanh (wc # hx)
      o = sigmoid (wo # hx)
  c <- assign ((f ⊙ ct1) + (i ⊙ cTilda))
  h <- assign (o ⊙ tanh c)
  return (VecPair h c, h)

-- | Parameter for a GRU
data GRUP t n x = GRUP (T [n+x,n] ('Typ 'Float t)) (T [n+x,n] ('Typ 'Float t)) (T [n+x,n] ('Typ 'Float t))

instance (KnownNat n, KnownNat x, KnownBits t) => KnownTensors (GRUP t n x) where
  travTensor f s (GRUP x y z) = GRUP <$> travTensor f (s<>"_z") x <*> travTensor f (s<>"_r") y <*> travTensor f (s<>"_w") z
instance (KnownNat n, KnownNat x, KnownBits t) => ParamWithDefault (GRUP t n x) where
  defaultInitializer = GRUP (denseWeights cellInitializerBit) (denseWeights cellInitializerBit) (denseWeights cellInitializerBit)


-- | Standard GRU cell
gru :: ∀ n x bs t. (KnownNat bs, KnownNat n, KnownBits t) => GRUP t n x ->
        RnnCell t '[ '[n,bs] ] (Tensor '[x,bs] (Flt t)) (Tensor '[n,bs] (Flt t))
gru (GRUP wz wr w) xt (VecSing ht1) = do
  hx <- assign (concat0 ht1 xt)
  let zt = sigmoid (wz ∙ hx)
      rt = sigmoid (wr ∙ hx)
      hTilda = tanh (w ∙ (concat0 (rt ⊙ ht1) xt))
  ht <- assign ((ones ⊝ zt) ⊙ ht1 + zt ⊙ hTilda)
  return (VecSing ht, ht)


data StackP w n = StackP (DenseP w (n + n) 3)

defStackP :: KnownNat n => KnownBits w => StackP w n
defStackP = StackP defaultInitializer
  -- (DenseP glorotUniform (stack0 (V [zeros, constant (-1), zeros]) )) -- demote popping a bit 

instance (KnownNat n, KnownBits w) => KnownTensors (StackP w n) where
  travTensor f s (StackP d) = StackP <$> travTensor f s d

instance (KnownNat n, KnownBits w) => (ParamWithDefault (StackP w n)) where
  defaultInitializer = defStackP

-- | A stack recurrent unit. The input has two purposes: 1. it is
-- saved in a stack. 2. it controls (a dense layer which gives) the
-- operation to apply on the stack.  The first type argument is the
-- depth of the stack.
stackRU :: ∀k n bs w. KnownNat k => KnownNat n => (KnownNat bs) => (KnownBits w) => StackP w n ->
        RnnCell w '[ '[k+1,n,bs]] (Tensor '[n,bs] (Flt w)) (Tensor '[n,bs] (Flt w))
stackRU (StackP w) input (VecSing st1) =
  succPos @k $
  plusComm @k @1 $ do
  let ct1 = nth0' @0 st1
      hx = concat0 ct1 input
      action :: T '[3,bs] (Flt w)
      action = softmax0 (w # hx)
  (_,tl) <- split0 @1 @k st1
  (it,_) <- split0 @k @1 st1
  let stTilda :: T '[3,k+1,n,bs] (Flt w)
      stTilda = stack0 (V [st1, tl `concat0` zeros, (expandDim0 input) `concat0` it])
  st <- assign (squeeze0 (inflate12 (matmul (flatten12 @(k+1) @n stTilda) (expandDim0 action))))
  let ct = nth0' @0 st
  return (VecSing st, ct)


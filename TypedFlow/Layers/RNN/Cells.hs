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
import TypedFlow.Types.Proofs
import GHC.TypeLits
import TypedFlow.Layers.Core (DenseP(..),(#))
import Prelude hiding (RealFrac(..))
import Data.Monoid ((<>))

--------------------------------------
-- Cells

-- | Standard RNN gate initializer. (The recurrent kernel is
-- orthogonal to avoid divergence; the input kernel is glorot)
cellInitializerBit :: ∀ n x t. (KnownNat n, KnownNat x, KnownBits t) => Gen (DenseP t (n + x) n)
cellInitializerBit = DenseP <$> (concat0 <$> recurrentInitializer <*> kernelInitializer) <*> biasInitializer
  where recurrentInitializer :: Gen (Tensor '[n, n] ('Typ 'Float t))
        recurrentInitializer = noise $ OrthogonalD
        kernelInitializer :: Gen (Tensor '[x, n] ('Typ 'Float t))
        kernelInitializer = glorotUniform
        biasInitializer = pure zeros

-- | Parameter for an LSTM
data LSTMP t n x = LSTMP (DenseP t (n+x) n) (DenseP t (n+x) n) (DenseP t (n+x) n) (DenseP t (n+x) n)

instance (KnownNat n, KnownNat x, KnownBits t) => KnownTensors (LSTMP t n x) where
  travTensor f s (LSTMP x y z w) = LSTMP <$> travTensor f (s<>"_f") x <*> travTensor f (s<>"_i") y <*> travTensor f (s<>"_c") z <*> travTensor f (s<>"_o") w
instance (KnownNat n, KnownNat x, KnownBits t) => ParamWithDefault (LSTMP t n x) where
  defaultInitializer = LSTMP <$> forgetInit <*> cellInitializerBit <*> cellInitializerBit <*> cellInitializerBit
    where forgetInit = DenseP <$> (denseWeights <$> cellInitializerBit) <*> pure ones

-- | Standard LSTM
lstm :: ∀ n x t. KnownNat x => KnownNat n => KnownBits t
  => LSTMP t n x -> RnnCell t '[ '[n], '[n]] (Tensor '[x] (Flt t)) (Tensor '[n] (Flt t))
lstm (LSTMP wf wi wc wo) input = C $ \(VecPair ht1 ct1) -> 
  let f = sigmoid (wf # hx)
      hx = (concat0 ht1 input)
      i = sigmoid (wi # hx)
      cTilda = tanh (wc # hx)
      o = sigmoid (wo # hx)
      c = ((f ⊙ ct1) + (i ⊙ cTilda))
      h = (o ⊙ tanh c)
  in (VecPair h c, h)

-- | Parameter for a GRU
data GRUP t n x = GRUP (T [n+x,n] ('Typ 'Float t)) (T [n+x,n] ('Typ 'Float t)) (T [n+x,n] ('Typ 'Float t))

instance (KnownNat n, KnownNat x, KnownBits t) => KnownTensors (GRUP t n x) where
  travTensor f s (GRUP x y z) = GRUP <$> travTensor f (s<>"_z") x <*> travTensor f (s<>"_r") y <*> travTensor f (s<>"_w") z
instance (KnownNat n, KnownNat x, KnownBits t) => ParamWithDefault (GRUP t n x) where
  defaultInitializer = GRUP <$> (denseWeights <$> cellInitializerBit) <*> (denseWeights <$> cellInitializerBit) <*> (denseWeights <$> cellInitializerBit)


-- | Standard GRU cell
gru :: ∀ n x t. KnownNat x => (KnownNat n, KnownBits t) => GRUP t n x ->
        RnnCell t '[ '[n] ] (Tensor '[x] (Flt t)) (Tensor '[n] (Flt t))
gru (GRUP wz wr w) xt = C $ \(VecSing ht1) ->
  let hx =  (concat0 ht1 xt)
      zt = sigmoid (wz ∙ hx)
      rt = sigmoid (wr ∙ hx)
      hTilda = tanh (w ∙ (concat0 (rt ⊙ ht1) xt))
      ht = ((ones ⊝ zt) ⊙ ht1 + zt ⊙ hTilda)
  in (VecSing ht, ht)


data StackP w n = StackP (DenseP w (n + n) 3)

defStackP :: KnownNat n => KnownBits w => Gen (StackP w n)
defStackP = StackP <$> defaultInitializer
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
        RnnCell w '[ '[k+1,n]] (Tensor '[n] (Flt w)) (Tensor '[n] (Flt w))
stackRU (StackP w) input = C $ \(VecSing st1) ->
  succPos @k #>
  plusMono @k @1 #>
  plusComm @k @1 #>
  termCancelation @k @1 #>
  let ct1 = nth0' @0 st1
      hx = concat0 ct1 input
      action :: T '[3] (Flt w)
      action = softmax0 (w # hx)
      tl :: T '[k,n] (Flt w)
      tl = slice0 @1 @(k+1) st1
      it :: T '[k,n] (Flt w)
      it = slice0 @0 @k  st1
      stTilda :: T '[3,k+1,n] (Flt w)
      stTilda = stack0 (st1 :**  (tl `concat0` zeros) :** (expandDim0 input `concat0` it) :** VUnit)
      st :: T '[k+1,n] (Flt w)
      st = inflate2 (flatten12 stTilda ∙ action)
      ct = nth0' @0 st
  in (VecSing st, ct)


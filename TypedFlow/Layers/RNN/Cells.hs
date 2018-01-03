{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
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
  gru) where

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
lstm (LSTMP wf wi wc wo) (VecPair ht1 ct1, input) = do
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
gru (GRUP wz wr w) (VecSing ht1, xt) = do
  hx <- assign (concat0 ht1 xt)
  let zt = sigmoid (wz ∙ hx)
      rt = sigmoid (wr ∙ hx)
      hTilda = tanh (w ∙ (concat0 (rt ⊙ ht1) xt))
  ht <- assign ((ones ⊝ zt) ⊙ ht1 + zt ⊙ hTilda)
  return (VecSing ht, ht)



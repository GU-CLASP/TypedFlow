{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UnicodeSyntax #-}
module MNIST where

import TypedFlow

(#>) :: forall b c a. (a -> b) -> (b -> c) -> a -> c
(#>) = flip (.)

atShape :: forall s t. T s t -> T s t
atShape x = x

mnist :: forall batchSize. KnownNat batchSize => Model [784,batchSize] Float32  '[10,batchSize] Float32
mnist input gold = do
  filters1 <- parameter "f1" convInitialiser
  filters2 <- parameter "f2" convInitialiser
  w1 <- parameter "w1" denseInitialiser
  w2 <- parameter "w2" denseInitialiser
  let nn = arrange3                           #>
           atShape @'[1,28,28,batchSize]      #>
           (relu . conv @32 @'[5,5] filters1) #>
           atShape @'[32,28,28,batchSize]     #>
           maxPool2D @2 @2                    #>
           atShape @'[32,14,14,batchSize]     #>
           (relu . conv @64 @'[5,5] filters2) #>
           maxPool2D @2 @2                    #>
           linearize3                         #>
           (relu . dense @1024 w1)            #>
           dense @10 w2
      logits = nn input
  categoricalDistribution logits gold

main :: IO ()
main = do
  writeFile "mnist_model.py" (generate $ compile (mnist @None))
  putStrLn "done!"

{-> main


-}



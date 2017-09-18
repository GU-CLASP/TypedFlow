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
  filters1 <- parameterDefault "f1"
  filters2 <- parameterDefault "f2"
  w1 <- parameterDefault "w1"
  w2 <- parameterDefault "w2"
  let nn = inflate3                           #>
           atShape @'[1,28,28,batchSize]      #>
           (relu . conv @32 @'[5,5] filters1) #>
           atShape @'[32,28,28,batchSize]     #>
           maxPool2D @2 @2                    #>
           atShape @'[32,14,14,batchSize]     #>
           (relu . conv @64 @'[5,5] filters2) #>
           maxPool2D @2 @2                    #>
           flatten3                           #>
           (relu . dense @1024 w1)            #>
           dense @10 w2
      logits = nn input
  categoricalDistribution logits gold

main :: IO ()
main = do
  generateFile "mnist_model.py" (compile defaultOptions (mnist @None))
  putStrLn "done!"

{-> main


-}



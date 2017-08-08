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

infixr >--
(>--) :: forall (m :: * -> *) t t1 t2 t3 t4.
               Monad m =>
               (t -> m (t3, t1)) -> (t1 -> m (t4, t2)) -> t -> m ((t3, t4), t2)
(l1 >-- l2) input = do
  (s1,x) <- l1 input
  (s2,y) <- l2 x
  return ((s1,s2),y)
  
encoder prefix input = do
  embs <- parameter (prefix++"embs") embeddingInitializer
  lstm1 <- parameter (prefix++"w1") lstmInitializer
  lstm2 <- parameter (prefix++"w2") lstmInitializer
  -- todo: what to do about sentence length?
  (sFinal,_) <-
    (rnn (timeDistribute (embedding @50 @100000 embs)) ()
      >--
     rnn (lstm @512 lstm1) (zeros,zeros)
      >--
     rnnBackwards (lstm @512 lstm2) (zeros,zeros)
     ) input
  return sFinal

decoder prefix thoughtVectors target = do
  embs <- parameter (prefix++"embs") embeddingInitializer
  projs <- parameter (prefix++"proj") denseInitialiser
  -- note: for an intra-language translation the embeddings can be shared.
  lstm1 <- parameter (prefix++"w1") lstmInitializer
  lstm2 <- parameter (prefix++"w2") lstmInitializer
  (_sFinal,outFinal) <-
    (rnn (timeDistribute (embedding @50 @100000 embs))
      .--.
     rnn (lstm @512 lstm1)
      .--.
     rnnBackwards (lstm @512 lstm2)
      .--.
      rnn (timeDistribute (dense projs))
     ) thoughtVectors target
  return outFinal

-- TODO: one also needs to turn the embedding into a projection layer to get
-- TODO: beam search

-- mnist :: forall batchSize. KnownNat batchSize => Model [784,batchSize] Float32  '[10,batchSize] Float32
-- mnist input gold = do
--   filters1 <- parameter "f1" convInitialiser
--   filters2 <- parameter "f2" convInitialiser
--   w1 <- parameter "w1" denseInitialiser
--   w2 <- parameter "w2" denseInitialiser
--   let nn = arrange3                           #>
--            atShape @'[1,28,28,batchSize]      #>
--            (relu . conv @32 @'[5,5] filters1) #>
--            atShape @'[32,28,28,batchSize]     #>
--            maxPool2D @2 @2                    #>
--            atShape @'[32,14,14,batchSize]     #>
--            (relu . conv @64 @'[5,5] filters2) #>
--            maxPool2D @2 @2                    #>
--            linearize3                         #>
--            (relu . dense @1024 w1)            #>
--            dense @10 w2
--       logits = nn input
--   categoricalDistribution logits gold

-- main :: IO ()
-- main = do
--   writeFile "mnist_model.py" (generate $ compile defaultOptions (mnist @None))
--   putStrLn "done!"

{-> main


-}



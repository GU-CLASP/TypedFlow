{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
module TypedFlow.Memo where

import qualified Data.IntMap as I
import System.Mem.StableName
import Data.IORef
import System.IO.Unsafe

type SNMap k v = I.IntMap [(StableName k,v)]

snMapLookup :: StableName k -> SNMap k v -> Maybe v
snMapLookup sn m = do
  x <- I.lookup (hashStableName sn) m
  lookup sn x

snMapInsert :: StableName k -> v -> SNMap k v -> SNMap k v
snMapInsert sn res = I.insertWith (++) (hashStableName sn) [(sn,res)]

memo :: (a -> b) -> a -> b
memo f = unsafePerformIO (
  do { tref <- newIORef (I.empty)
     ; return (applyStable f tref)
     })

applyStable :: (a -> b) -> IORef (SNMap a b) -> a -> b
applyStable f tbl arg = unsafePerformIO (
  do { sn <- makeStableName arg
     ; lkp <- snMapLookup sn <$> readIORef tbl
     ; case lkp of
         Just result -> return result
         Nothing ->
           do { let res = f arg
              ; modifyIORef tbl (snMapInsert sn res)
              ; return res
              }})


memoEffect :: forall a m b. (a -> m b) -> a -> m b
memoEffect f a = 
  let tref = unsafePerformIO (newIORef (I.empty))
  in applyStable f tref a

applyStableEffect :: Monad m => (a -> m b) -> IORef (SNMap a b) -> a -> m b
applyStableEffect f tbl arg = do
  let sn = unsafePerformIO (makeStableName arg)
  let lkp = snMapLookup sn (unsafePerformIO (readIORef tbl))
  case lkp of
    Just result -> return result
    Nothing -> do
      res <- f arg
      return (unsafePerformIO (modifyIORef tbl (snMapInsert sn res)) `seq` res)

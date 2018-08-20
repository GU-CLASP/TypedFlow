{-# LANGUAGE TypeInType #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE GADTs #-}
module TypedFlow.Memo where

import qualified Data.IntMap as I
import qualified Data.Map.Strict as M
import System.Mem.StableName
import Data.IORef
import System.IO.Unsafe
import Unsafe.Coerce
import Data.Kind (Type)
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

memoOrd :: Ord a => (a -> b) -> a -> b
memoOrd f = unsafePerformIO (
  do { tref <- newIORef (M.empty)
     ; return (applyStableOrd f tref)
     })

applyStableOrd :: Ord a => (a -> b) -> IORef (M.Map a b) -> a -> b
applyStableOrd f tbl arg = unsafePerformIO (
  do { lkp <- M.lookup arg <$> readIORef tbl
     ; case lkp of
         Just result -> return result
         Nothing ->
           do { let res = f arg
              ; modifyIORef tbl (M.insert arg res)
              ; return res
              }})


data Some2 k1 k2 (f :: k1 -> k2 -> Type) where
  Some2 :: forall k1 k2 f a b. StableName (f a b) -> Some2 k1 k2 f

instance Eq (Some2 k1 k2 f) where
  Some2 sn1 == Some2 sn2 = eqStableName sn1 sn2

type SSNMap2 k1 k2 (f :: k1 -> k2 -> Type) v = I.IntMap [(Some2 k1 k2 f,v)]

makeSn2 :: f a b -> Some2 k1 k2 f
makeSn2 = Some2 . unsafePerformIO . makeStableName

snMapLookup2 :: Some2 k1 k2 f -> SSNMap2 k1 k2 f v -> Maybe v
snMapLookup2 (Some2 sn) m = do
  x <- I.lookup (hashStableName sn) m
  lookup (Some2 sn) x

snMapInsert2 :: Some2 k1 k2 f -> v -> SSNMap2 k1 k2 f v -> SSNMap2 k1 k2 f v
snMapInsert2 (Some2 sn) res = I.insertWith (++) (hashStableName sn) [(Some2 sn,res)]

data KV k1 k2 (f :: k1 -> k2 -> Type) (v :: k1 -> k2 -> Type)  where
  KV :: forall k1 k2 f v a b. StableName (f a b) -> v a b -> KV k1 k2 f v

type SNMap22 k1 k2 (f :: k1 -> k2 -> Type) (v :: k1 -> k2 -> Type) = I.IntMap [KV k1 k2 f v]

snMap22Lookup :: StableName (f a b) -> SNMap22 k1 k2 f v -> Maybe (v a b)
snMap22Lookup sn  m = do
  x <- I.lookup (hashStableName sn) m
  lkKV sn x

lkKV :: StableName (f a b) -> [KV k1 k2 f v] -> Maybe (v a b)
lkKV _ [] = Nothing
lkKV sn (KV sn' v:kvs) | eqStableName sn sn' = Just (unsafeCoerce v) -- sn == sn' -> a == a' and b == b' 
                       | otherwise = lkKV sn kvs

snMap22Insert :: KV k1 k2 f v -> SNMap22 k1 k2 f v -> SNMap22 k1 k2 f v
snMap22Insert (KV sn res) = I.insertWith (++) (hashStableName sn) [KV sn res]


-- | The type of a memo table for functions of a.
type Memo a = forall r. (a -> r) -> (a -> r)

-- | Memoize a two argument function (just apply the table directly for
-- single argument functions).
memo2 :: Memo a -> Memo b -> (a -> b -> r) -> (a -> b -> r)
memo2 a b = a . (b .)

-- | Memoize a three argument function.
memo3 :: Memo a -> Memo b -> Memo c -> (a -> b -> c -> r) -> (a -> b -> c -> r)
memo3 a b c = a . (memo2 b c .)

{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE GADTs #-}

module TypedFlow.Memo2 where

import Data.Kind (Type)
import qualified Data.Map.Strict as M
import System.Mem.StableName
-- import Data.IORef
-- import System.IO.Unsafe
import Unsafe.Coerce
import qualified Data.IntMap as I
import Data.Type.Equality
import Control.Monad.IO.Class
import Data.IORef
import TypedFlow.Types.Proofs (SingEq(..))
import Data.List (intercalate)

data Map0 k (m :: Type -> Type) f v = forall . Map0 {
  m0Key :: f -> IO k,
  m0Empty :: m v,
  m0lk  :: k -> m v -> Maybe v,
  m0upd :: k -> (Maybe v -> v) -> m v -> m v,
  m0fmap :: forall u w.  (u -> w) -> m u -> m w,
  m0showKey :: k -> String,
  m0showTbl :: (v -> String) -> (m v -> String)
  }


data Map1 (k :: k1 -> Type) (m :: (k1 -> Type) -> Type)  (f :: k1 -> Type) (v :: k1 -> Type) = Map1 {
  m1Key :: forall x. f x -> IO (k x),
  m1Empty :: m v,
  m1lk  :: forall x. k x -> m v -> Maybe (v x),
  m1upd :: forall x. k x -> (Maybe (v x) -> (v x)) -> m v -> m v,
  m1showKey :: forall x . k x -> String,
  m1showTbl :: (forall x . v x -> String) -> (m v -> String)
  }

data Map2 (k :: k1 -> k2 -> Type) (m :: (k1 -> k2 -> Type) -> Type)  (f :: k1 -> k2 -> Type) (v :: k1 -> k2 -> Type) = Map2 {
  m2Key :: forall x y. f x y -> IO (k x y),
  m2Empty :: m v,
  m2lk  :: forall x y. k x y -> m v -> Maybe (v x y),
  m2upd :: forall x y. k x y -> (Maybe (v x y) -> (v x y)) -> m v -> m v,
  -- m2fmap :: forall u w.  (forall x y. u x y -> w x y) -> m u -> m w,
  m2showKey :: forall x y. k x y -> String,
  m2showTbl :: (forall x y. v x y -> String) -> (m v -> String)
  }

data Map3 (k :: k1 -> k2 -> k3 -> Type) (m :: (k1 -> k2 -> k3 -> Type) -> Type)  (f :: k1 -> k2 -> k3 -> Type) (v :: k1 -> k2 -> k3 -> Type) = Map3 {
  m3Key :: forall x y z. f x y z -> IO (k x y z),
  m3Empty :: m v,
  m3lk  :: forall x y z. k x y z -> m v -> Maybe (v x y z),
  m3upd :: forall x y z. k x y z -> (Maybe (v x y z) -> (v x y z)) -> m v -> m v,
  m3showKey :: forall x y z. k x y z -> String,
  m3showTbl :: (forall x y z. v x y z -> String) -> (m v -> String)
  }

newtype Id x = Id x

ordMap :: forall k b. (Ord k, Show k) => Map0 k (M.Map k) k b
ordMap = Map0 {..} where
  m0Key = return
  m0Empty = mempty
  m0lk k = M.lookup k
  m0upd k f m = M.alter (Just . f) k m
  m0fmap = fmap
  m0showKey = show
  m0showTbl sh m = intercalate ";" [(show k) <> "↦" <> (sh v) | (k,v) <- M.assocs m]

data Single1 f g where
  None1 :: Single1 f g
  Single1 :: f a -> g a -> Single1 f g 

verifMap1 :: forall k v. SingEq k => Map1 k (Single1 k) k v
verifMap1 = Map1 {..} where
  m1Key = return
  m1Empty = None1
  m1lk :: k a -> Single1 k b -> Maybe (b a)
  m1lk k = \case
    None1 -> Nothing
    Single1 k' v -> case testEq k k' of
      Just Refl -> Just v
      Nothing -> error "verifMap1: mismatching keys! (1)"
  m1upd :: forall x. k x -> (Maybe (v x) -> (v x)) -> Single1 k v -> Single1 k v
  m1upd k f None1 = Single1 k (f Nothing)
  m1upd k f (Single1 k' v) = case testEq k k' of
      Just Refl -> Single1 k (f (Just v))
      Nothing -> error "verifMap1: mismatching keys! (2)"
  m1showKey _ = "#"
  m1showTbl :: (forall x . v x -> String) -> (Single1 k v -> String)
  m1showTbl _ None1 = "·"
  m1showTbl h (Single1 _ v) = "!" <> (h v)


testStable :: StableName a -> StableName b -> Maybe (a :~: b)
testStable sn sn' | eqStableName sn sn' = Just (unsafeCoerce Refl)
                  | otherwise = Nothing

snMap2 :: forall f v. Map2 (SN2 f) (SNMap22 f) f v
snMap2 = Map2 {..} where
  m2showTbl :: (forall x y. v x y -> String) -> (SNMap22 f v -> String)
  m2showTbl h (SNMap22 m) = intercalate "," [ m2showKey k <> "↦" <> h v | e <- I.elems m, KV k v <- e   ]
  m2showKey (SN2 sn) = show (hashStableName sn)
  m2Key obj = SN2 <$> makeStableName obj
  m2Empty = mempty
  m2lk = snMap22Lookup
  m2upd :: SN2 f x y -> (Maybe (v x y) -> (v x y)) -> SNMap22 f v -> SNMap22 f v
  m2upd (SN2 sn) f (SNMap22 m) = SNMap22 $
                                 I.alter (\case Nothing -> Just [KV (SN2 sn) (f Nothing)]
                                                Just p -> Just (updKV (SN2 sn) f p))
                                 (hashStableName sn)
                                 m

  updKV :: SN2 f' x y -> (Maybe (v' x y) -> (v' x y)) -> [KV k1 k2 (SN2 f') v'] -> [KV k1 k2 (SN2 f') v']
  updKV (SN2 sn) f [] = [KV (SN2 sn) (f Nothing)]
  updKV (SN2 sn) f (v@(KV (SN2 sn') x):xs) = case testStable sn sn' of
    Just Refl -> KV (SN2 sn') (f (Just x)):xs
    Nothing -> v : updKV (SN2 sn) f xs
                                 
  -- m2fmap :: forall u w.  (forall x y. u x y -> w x y) -> SNMap22 f u -> SNMap22 f w
  -- m2fmap h (SNMap22 t) = SNMap22 (fmap (fmap (\(KV k v) -> KV k (h v))) t)

  snMap22Lookup :: forall a b f' v'. SN2 f' a b -> SNMap22 f' v' -> Maybe (v' a b)
  snMap22Lookup (SN2 sn) (SNMap22 m) = do
    x <- I.lookup (hashStableName sn) m
    lkKV sn x

  lkKV :: forall k1 k2 f' v' a b . StableName (f' a b) -> [KV k1 k2 (SN2 f') v'] -> Maybe (v' a b)
  lkKV _ [] = Nothing
  lkKV sn (KV (SN2 sn') v:kvs) = case testStable sn sn' of
                             Just Refl ->  Just (unsafeCoerce v) -- sn == sn' -> a == a' and b == b' 
                             Nothing ->  lkKV sn kvs


data KV k1 k2 (f :: k1 -> k2 -> Type) (v :: k1 -> k2 -> Type)  where
  KV :: forall k1 k2 f v a b. f a b -> v a b -> KV k1 k2 f v

newtype SNMap22  (f :: k1 -> k2 -> Type) (v :: k1 -> k2 -> Type) = SNMap22 (I.IntMap [KV k1 k2 (SN2 f) v]) deriving (Monoid, Semigroup)

newtype SN2 (f :: k1 -> k2 -> Type) a b = SN2 (StableName (f a b)) 

data (:.:) (m1 :: k2 -> Type) (m2 :: k1 -> k2) (h :: k1) = Comp (m1 (m2 h))


data Sig02 f g x y where
  Ex02 :: f -> g x y -> Sig02 f g x y

data Sig03 f g x y z where
  Ex03 :: f -> g x y z -> Sig03 f g x y z

data Sig12 f g x y z where
  Ex12 :: f x -> g y z -> Sig12 f g x y z

data Sig22 f g x y where
  Ex22 :: f x y -> g x y -> Sig22 f g x y

data P33 f g x y z where
  T33 :: f x y z -> g x y z -> P33 f g x y z



containing00 :: (forall v. Map0 k1 m1 f v) -> Map0 k2 m2 g h -> Map0 (k1,k2) (m1 :.: m2)  (f,g) h
containing00 f g  = Map0
   {
   m0Key = (\(a,b) -> (,) <$> m0Key f a <*> m0Key g b),
   m0Empty = Comp (m0Empty f),
   m0lk = \(k1,k2) (Comp t) -> do t' <- m0lk f k1 t; m0lk g k2 t',
   m0upd = \(k1,k2) h (Comp t) -> Comp $ m0upd f k1 (m0upd g k2 h . \case Just tb -> tb; Nothing -> (m0Empty g)) t,
   m0fmap = \h (Comp t) -> Comp $ m0fmap f (m0fmap g h) t,
   m0showKey = \(k1,k0) -> m0showKey f k1 <> "," <> m0showKey g k0,
   m0showTbl = \h (Comp t) -> m0showTbl f (m0showTbl g h) t
   }                      

containing02 :: (forall v. Map0 k1 m1 f v) -> Map2 k2 m2 g h -> Map2 (Sig02 k1 k2) (m1 :.: m2) (Sig02 f g)  h
containing02 f g = Map2
   {
   m2Key = (\(Ex02 a b) -> Ex02 <$> m0Key f a <*> m2Key g b),
   m2Empty = Comp (m0Empty f),
   m2lk = \(Ex02 k1 k2) (Comp t) -> do t' <- m0lk f k1 t; m2lk g k2 t',
   m2upd = \(Ex02 k1 k2) h (Comp t) -> Comp $ m0upd f k1 (m2upd g k2 h . \case Just tb -> tb; Nothing -> (m2Empty g)) t,
   -- m2fmap = \h (Comp t) -> Comp $ m0fmap f (m2fmap g h) t,
   m2showKey = \(Ex02 k1 k2) -> m0showKey f k1 <> "," <> m2showKey g k2,
   m2showTbl = \h (Comp t) -> m0showTbl f (m2showTbl g h) t
   }                      

containing03 :: (forall v. Map0 k1 m1 f v) -> Map3 k2 m2 g h -> Map3 (Sig03 k1 k2) (m1 :.: m2) (Sig03 f g)  h
containing03 f g = Map3
   {
   m3Key = (\(Ex03 a b) -> Ex03 <$> m0Key f a <*> m3Key g b),
   m3Empty = Comp (m0Empty f),
   m3lk = \(Ex03 k1 k3) (Comp t) -> do t' <- m0lk f k1 t; m3lk g k3 t',
   m3upd = \(Ex03 k1 k3) h (Comp t) -> Comp $ m0upd f k1 (m3upd g k3 h . \case Just tb -> tb; Nothing -> (m3Empty g)) t,
   m3showKey = \(Ex03 k1 k2) -> m0showKey f k1 <> "," <> m3showKey g k2
,
   m3showTbl = \h (Comp t) -> m0showTbl f (m3showTbl g h) t
  }                      

newtype Lam' (m2 :: (k2 -> k3 -> Type) -> Type) (h :: k1 -> k2 -> k3 -> Type) (a :: k1) = Lam' {fromLam' :: (m2 (h a))}
data M12 (m1 :: (k1 -> Type) -> Type) (m2 :: (k2 -> k3 -> Type) -> Type) (h :: k1 -> k2 -> k3 -> Type) = M12 (m1 (Lam' m2 h))

containing12 :: (forall v. Map1 k1 m1 f v) -> (forall k4. Map2 k2 m2 g (h k4)) -> Map3 (Sig12 k1 k2) (M12 m1 m2) (Sig12 f g) h
containing12 f g = Map3
   {
   m3Key = (\(Ex12 a b) -> Ex12 <$> m1Key f a <*> m2Key g b),
   m3Empty = M12 (m1Empty f),
   m3lk = \(Ex12 k1 k2) (M12 t) -> do Lam' t' <- m1lk f k1 t; m2lk g k2 t',
   m3upd = \(Ex12 k1 k2) h (M12 t) -> M12 $ m1upd f k1 (Lam' . m2upd g k2 h . (\case Just tb -> tb; Nothing -> m2Empty g) . fmap fromLam') t,
   m3showKey = \(Ex12 k1 k2) -> m1showKey f k1 <> ">" <> m2showKey g k2,
   m3showTbl = \h (M12 t) -> m1showTbl f (m2showTbl g h . fromLam') t
   }



data F2m m g h = F2m (forall x y. g x y -> m (h x y))
data F2m' m g f h = F2m' (forall x y. g x y -> f x y -> m (h x y))

data F3m m g h = F3m (forall x y z. g x y z -> m (h x y z))
data F3m' m g f h = F3m' (forall x y z. g x y z -> f x y z -> m (h x y z))

memo2 :: forall g h k m n. MonadIO n => Map2 k m g h -> ((forall x y. g x y -> n (h x y)) -> forall x y. g x y -> n (h x y)) -> n (F2m n g h)
memo2 Map2{..} f = do
    tblRef <- liftIO $ newIORef m2Empty
    let finished :: forall x y. g x y -> n (h x y)
        finished arg = do
          tbl <- liftIO $ readIORef tblRef
          key <- liftIO $ m2Key arg
          case m2lk key tbl of
            Just result -> do
              -- liftIO $ putStrLn "memo2: hit"
              return result
            Nothing -> do
              -- liftIO $ putStrLn "memo2: miss"
              res <- f finished arg
              liftIO $ modifyIORef tblRef (m2upd key $ \_ -> res)
              return res
    return (F2m finished)
  
memo2' :: forall g f h k m n. MonadIO n => Map2 k m g h -> ((forall x y. g x y -> f x y -> n (h x y)) -> forall x y. g x y -> f x y -> n (h x y)) -> n (F2m' n g f h)
memo2' Map2{..} f = do
    tblRef <- liftIO $ newIORef m2Empty
    let finished :: forall x y. g x y -> f x y -> n (h x y)
        finished arg extra = do
          tbl <- liftIO $ readIORef tblRef
          key <- liftIO $ m2Key arg
          case m2lk key tbl of
            Just result -> return result
            Nothing -> do
              res <- f finished arg extra
              liftIO $ modifyIORef tblRef (m2upd key $ \_ -> res)
              return res
    return (F2m' finished)

memo3' :: forall g f h k m n. MonadIO n => Map3 k m g h -> ((forall x y z. g x y z -> f x y z -> n (h x y z)) -> forall x y z. g x y z -> f x y z -> n (h x y z)) -> n (F3m' n g f h)
memo3' Map3{..} f = do
    tblRef <- liftIO $ newIORef m3Empty
    let finished :: forall x y z. g x y z -> f x y z -> n (h x y z)
        finished arg extra = do
          tbl <- liftIO $ readIORef tblRef
          key <- liftIO $ m3Key arg
          case m3lk key tbl of
            Just result -> do
              liftIO $ putStrLn "memo3: hit"
              return result
            Nothing -> do
              liftIO $ putStrLn ("memo3: miss " <> m3showKey key) --  <> " from " <> m3showTbl (const ".") tbl
              res <- f finished arg extra
              liftIO $ modifyIORef tblRef (m3upd key $ \_ -> res)
              return res
    return (F3m' finished)

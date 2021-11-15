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

data Map0 k (m :: Type -> Type) f v = forall . Map0 {
  m0Key :: f -> IO k,
  m0Empty :: m v,
  m0lk  :: k -> m v -> Maybe v,
  m0upd :: k -> (Maybe v -> v) -> m v -> m v
  }


data Map1 (k :: k1 -> Type) (m :: (k1 -> Type) -> Type)  (f :: k1 -> Type) (v :: k1 -> Type) = Map1 {
  m1Key :: forall x. f x -> IO (k x),
  m1Empty :: m v,
  m1lk  :: forall x. k x -> m v -> Maybe (v x),
  m1upd :: forall x. k x -> (Maybe (v x) -> (v x)) -> m v -> m v
  }

data Map2 (k :: k1 -> k2 -> Type) (m :: (k1 -> k2 -> Type) -> Type)  (f :: k1 -> k2 -> Type) (v :: k1 -> k2 -> Type) = Map2 {
  m2Key :: forall x y. f x y -> IO (k x y),
  m2Empty :: m v,
  m2lk  :: forall x y. k x y -> m v -> Maybe (v x y),
  m2upd :: forall x y. k x y -> (Maybe (v x y) -> (v x y)) -> m v -> m v,
  m2fmap :: forall u w.  (forall x y. u x y -> w x y) -> m u -> m w
  }

data Map3 (k :: k1 -> k2 -> k3 -> Type) (m :: (k1 -> k2 -> k3 -> Type) -> Type)  (f :: k1 -> k2 -> k3 -> Type) (v :: k1 -> k2 -> k3 -> Type) = Map3 {
  m3Key :: forall x y z. f x y z -> IO (k x y z),
  m3Empty :: m v,
  m3lk  :: forall x y z. k x y z -> m v -> Maybe (v x y z),
  m3upd :: forall x y z. k x y z -> (Maybe (v x y z) -> (v x y z)) -> m v -> m v
  }

newtype Id x = Id x

ordMap :: forall k b. Ord k => Map0 k (M.Map k) k b
ordMap = Map0 {..} where
  m0Key = return
  m0Empty = mempty
  m0lk k = M.lookup k
  m0upd k f m = M.alter (Just . f) k m

data Single1 f g where
  None1 :: Single1 f g
  Single1 :: f a -> g a -> Single1 f g 

verifMap1 :: forall k b. SingEq k => Map1 k (Single1 k) k b
verifMap1 = Map1 {..} where
  m1Key = return
  m1Empty = None1
  m1lk :: k a -> Single1 k b -> Maybe (b a)
  m1lk k = \case
    None1 -> Nothing
    Single1 k' v -> case testEq k k' of
      Just Refl -> Just v
      Nothing -> error "verifMap1: mismatching keys!"
  m1upd k f _m = Single1 k (f Nothing)


testStable :: StableName a -> StableName b -> Maybe (a :~: b)
testStable sn sn' | eqStableName sn sn' = Just (unsafeCoerce Refl)
                  | otherwise = Nothing

snMap2 :: forall f v. Map2 (SN2 f) (SNMap22 f) f v
snMap2 = Map2 {..} where
  m2Key obj = SN2 <$> makeStableName obj
  m2Empty = mempty
  m2lk = snMap22Lookup
  m2upd :: SN2 f x y -> (Maybe (v x y) -> (v x y)) -> SNMap22 f v -> SNMap22 f v
  m2upd (SN2 sn) f (SNMap22 m) = SNMap22 $
                                 I.alter (\case Nothing -> Just [KV (SN2 sn) (f Nothing)]
                                                Just p -> Just [KV (SN2 sn') (case testStable sn sn' of Nothing -> x; Just Refl -> f (Just x))
                                                               | KV (SN2 sn') x <- p])
                                 (hashStableName sn)
                                 m
                                 
  m2fmap :: forall u w.  (forall x y. u x y -> w x y) -> SNMap22 f u -> SNMap22 f w
  m2fmap h (SNMap22 t) = SNMap22 (fmap (fmap (\(KV k v) -> KV k (h v))) t)

  snMap22Lookup :: forall a b f v. SN2 f a b -> SNMap22 f v -> Maybe (v a b)
  snMap22Lookup (SN2 sn) (SNMap22 m) = do
    x <- I.lookup (hashStableName sn) m
    lkKV sn x

  lkKV :: forall k1 k2 f v a b . StableName (f a b) -> [KV k1 k2 (SN2 f) v] -> Maybe (v a b)
  lkKV _ [] = Nothing
  lkKV sn (KV (SN2 sn') v:kvs) = case testStable sn sn' of
                             Just Refl ->  Just (unsafeCoerce v) -- sn == sn' -> a == a' and b == b' 
                             Nothing ->  lkKV sn kvs


data KV k1 k2 (f :: k1 -> k2 -> Type) (v :: k1 -> k2 -> Type)  where
  KV :: forall k1 k2 f v a b. f a b -> v a b -> KV k1 k2 f v

newtype SNMap22  (f :: k1 -> k2 -> Type) (v :: k1 -> k2 -> Type) = SNMap22 (I.IntMap [KV k1 k2 (SN2 f) v]) deriving (Monoid, Semigroup)

newtype SN2 (f :: k1 -> k2 -> Type) a b = SN2 (StableName (f a b)) 

data (:.:) (m1 :: Type -> Type) (m2 :: k -> Type) (h :: k) = Comp (m1 (m2 h))


containing00 :: (forall v. Map0 k1 m1 f v) -> Map0 k2 m2 g h -> Map0 (k1,k2) (m1 :.: m2)  (f,g) h
containing00 f g  = Map0
   {
   m0Key = (\(a,b) -> (,) <$> m0Key f a <*> m0Key g b),
   m0Empty = Comp (m0Empty f),
   m0lk = \(k1,k2) (Comp t) -> do t' <- m0lk f k1 t; m0lk g k2 t',
   m0upd = \(k1,k2) h (Comp t) -> Comp $ m0upd f k1 (m0upd g k2 h . \case Just tb -> tb; Nothing -> (m0Empty g)) t
   }                      

data Sig02 f g x y where
  Ex02 :: f -> g x y -> Sig02 f g x y

data Sig12 f g x y z where
  Ex12 :: f x -> g y z -> Sig12 f g x y z

data Sig22 f g x y where
  Ex22 :: f x y -> g x y -> Sig22 f g x y



containing02 :: (forall v. Map0 k1 m1 f v) -> Map2 k2 m2 g h -> Map2 (Sig02 k1 k2) (m1 :.: m2) (Sig02 f g)  h
containing02 f g = Map2
   {
   m2Key = (\(Ex02 a b) -> Ex02 <$> m0Key f a <*> m2Key g b),
   m2Empty = Comp (m0Empty f),
   m2lk = \(Ex02 k1 k2) (Comp t) -> do t' <- m0lk f k1 t; m2lk g k2 t',
   m2upd = \(Ex02 k1 k2) h (Comp t) -> Comp $ m0upd f k1 (m2upd g k2 h . \case Just tb -> tb; Nothing -> (m2Empty g)) t,
   m2fmap = \h t -> _
   }                      

newtype Lam12 h a b c = Lam12 {fromLam12 :: h b c}
newtype Lam' (m2 :: (k2 -> k3 -> Type) -> Type) (h :: k1 -> k2 -> k3 -> Type) (a :: k1) = Lam' {fromLam' :: (m2 (h a))}
data M12 (m1 :: (k1 -> Type) -> Type) (m2 :: (k2 -> k3 -> Type) -> Type) (h :: k1 -> k2 -> k3 -> Type) = M12 (m1 (Lam' m2 h))

containing12 :: (forall v. Map1 k1 m1 f v) -> Map2 k2 m2 g h -> Map3 (Sig12 k1 k2) (M12 m1 m2) (Sig12 f g) (Lam12 h)
containing12 f g = Map3
   {
   m3Key = (\(Ex12 a b) -> Ex12 <$> m1Key f a <*> m2Key g b),
   m3Empty = M12 (m1Empty f),
   m3lk = \(Ex12 k1 k2) (M12 t) -> do Lam' t' <- m1lk f k1 t; Lam12 <$> m2lk g k2 (m2fmap g fromLam12 t'),
   m3upd = \(Ex12 k1 k2) h (M12 t) -> M12 $ m1upd f k1 (Lam' . m2fmap g Lam12 . m2upd g k2 (fromLam12 . h . fmap Lam12) . m2fmap g fromLam12 .
                                                         (\case Just tb -> tb; Nothing -> m2fmap g Lam12 (m2Empty g)) . fmap fromLam') t
   }                      


data F2m m g h = F2m (forall x y. g x y -> m (h x y))

data F3m m g h = F3m (forall x y z. g x y z -> m (h x y z))

memo2 :: forall g h k m n. MonadIO n => Map2 k m g h -> ((forall x y. g x y -> n (h x y)) -> forall x y. g x y -> n (h x y)) -> n (F2m n g h)
memo2 Map2{..} f = do
    tblRef <- liftIO $ newIORef m2Empty
    let finished :: forall x y. g x y -> n (h x y)
        finished arg = do
          tbl <- liftIO $ readIORef tblRef
          key <- liftIO $ m2Key arg
          case m2lk key tbl of
            Just result -> return result
            Nothing -> do
              res <- f finished arg
              liftIO $ modifyIORef tblRef (m2upd key $ \_ -> res)
              return res
    return (F2m finished)
  

memo3 :: forall g h k m n. MonadIO n => Map3 k m g h -> ((forall x y z. g x y z -> n (h x y z)) -> forall x y z. g x y z -> n (h x y z)) -> n (F3m n g h)
memo3 Map3{..} f = do
    tblRef <- liftIO $ newIORef m3Empty
    let finished :: forall x y z. g x y z -> n (h x y z)
        finished arg = do
          tbl <- liftIO $ readIORef tblRef
          key <- liftIO $ m3Key arg
          case m3lk key tbl of
            Just result -> return result
            Nothing -> do
              res <- f finished arg
              liftIO $ modifyIORef tblRef (m3upd key $ \_ -> res)
              return res
    return (F3m finished)

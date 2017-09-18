import Control.Applicative
import Test.QuickCheck.Gen
import Data.List
import Data.Array
data Abs a = Bin a (Abs a) (Abs a) | Leaf a deriving Show

type Method a =  a -> [a] -> [a] -> [a]

parens :: String -> String
parens xs = "(" ++ xs ++ ")"

preorder :: Char -> [Char] -> [Char] -> String
preorder x l r =  (x : l ++ r)
postorder :: Char -> [Char] -> [Char] -> String
postorder x l r =  (l ++ r ++ [x])
reversePO :: Char -> [Char] -> [Char] -> String
reversePO x l r =  (x : r ++ l)

linearize _ (Leaf x) = [x]
linearize m (Bin x l r) = parens (m x (lin l) (lin r))
  where lin = linearize m

mkMethods :: Eq a => [(a->Bool,Method a)] -> Method a
mkMethods ms x = case find (\(p,_) -> p x) ms of
  Just (_,m) -> m x
  Nothing -> error "no applicable linearization method"

linPO :: Abs Char -> [Char]
linPO = linearize (mkMethods [(const True,preorder)])

lin1 :: Abs Char -> [Char]
lin1 = linearize (mkMethods [(\x -> x < '3',reversePO),(const True,preorder)])


ex :: Abs Char
ex = Bin 'a' (Bin '1' (Leaf 'b') (Leaf 'c')) (Leaf 'd')

guard :: Alternative f => Bool -> f a -> f a
guard True x = x
guard False _ = empty


arb :: Gen (Abs Char)
arb = sized $ \n -> do
  oneof (take (max 1 n) [(Leaf <$> elements ['a'..'e'])
                        ,resize (n-1) (Bin <$> elements ['0'..'4'] <*> arb <*> arb)])

arbOkSize :: Gen (Abs Char)
arbOkSize = do
  x <- resize 6 arb
  let xx = linPO x
  if (length xx > 2 && length xx < 22)
    then return x
    else arbOkSize

mySample :: Int -> IO [Abs Char]
mySample n = generate (sequence $ replicate n arbOkSize)

showEx :: Abs Char -> String
showEx x = linPO x ++ "\t" ++ lin1 x


test :: IO ()
test = mapM_ putStrLn . map showEx =<< mySample 10


main :: IO ()
main = writeFile "synthtrees.txt" . unlines . map showEx =<< mySample 100000

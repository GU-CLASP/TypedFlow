{-|
Module      : TypedFlow.Models.Topic
Description : Topic models
Copyright   : (c) Jean-Philippe Bernardy, 2017
License     : LGPL-3
Maintainer  : jean-philippe.bernardy@gu.se
Stability   : experimental
-}


module TypedFlow.Models.Topic where


-- -- | create a document summarization function with appropriate parameters.
-- mkDocumentSummary
--   :: String -> -- ^ prefix for parameter names
--      Gen (T '[n,e] (Flt t) -> T '[a] (Flt t)) -- ^ document vector (summary)
-- mkDocumentSummary prefix = do
--   filter <- parameter (prefix ++ "_filter") (truncatedNormal 0.1 )
--   return $ (relu . conv filter)



-- p = softmax (A d)
-- s = B p

-- | An implementation of 'Topically Driven Neural Language Model' by Lau, Baldwin and Cohn.
tdlm
  :: T '[n] (Flt t) -- ^ document
  -> Gen _
tdlm d = do
  embs <- parameterDefault "embs"
  drpEmb <- mkDropout dropProb
  filters <- parameterDefault "conv"
  
  let docInputs = embedding embs d
      conv'd = convValid @nFilters @[embeddingSize,filterSize] filters
      max'd = reduceMax @Dim

module Aggr where

agreement :: KnownNat batchSize => Tensor '[20,batchSize] Int32 -> Gen (Tensor '[20,batchSize] Float32)
agreement input' = do
  let input = expandDim1 input'
  (embs,lstm1,lstm2,w) <- parameter "params"
  (_sFi,out) <- rnn (timeDistribute (embedding @50 @100000 embs)
                     .--.
                     (lstm @150 lstm1)
                     .--.
                     (lstm @150 lstm2)
                     .--.
                     timeDistribute (sigmoid . squeeze0 . dense  w))
                (() |> (zeros,zeros) |> (zeros,zeros) |> (),input)
  return out


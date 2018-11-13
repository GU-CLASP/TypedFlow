{ mkDerivation, base, containers, ghc-typelits-knownnat, mtl
, pretty-compact, stdenv
}:
mkDerivation {
  pname = "typedflow";
  version = "0.9";
  src = /home/jyp/repo/gu/TypedFlow;
  libraryHaskellDepends = [
    base containers ghc-typelits-knownnat mtl pretty-compact
  ];
  description = "Typed frontend to TensorFlow and higher-order deep learning";
  license = stdenv.lib.licenses.lgpl3;
}

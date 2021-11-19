{ mkDerivation, base, containers, ghc-typelits-knownnat, lib, mtl
, prettyprinter
}:
mkDerivation {
  pname = "typedflow";
  version = "0.9";
  src = /home/jyp/repo/gu/TypedFlow;
  libraryHaskellDepends = [
    base containers ghc-typelits-knownnat mtl prettyprinter
  ];
  description = "Typed frontend to TensorFlow and higher-order deep learning";
  license = lib.licenses.lgpl3Only;
}

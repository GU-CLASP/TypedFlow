{ mkDerivation, aeson, base, base-compat, bytestring, containers
, criterion, deepseq, fetchgit, pretty, stdenv, text
, unordered-containers, wl-pprint
}:
mkDerivation {
  pname = "pretty-compact";
  version = "3.0";
  src = fetchgit {
    url = "git@github.com:jyp/prettiest.git";
    sha256 = "0m8bjpc1pwzfkdzq7fgji81yffwn91ywybvmnazmy2b47rg24wjf";
    rev = "a36f4ea19eed4ece78f7c939a1bc73a3393386a2";
  };
  libraryHaskellDepends = [ base base-compat containers ];
  benchmarkHaskellDepends = [
    aeson base base-compat bytestring criterion deepseq pretty text
    unordered-containers wl-pprint
  ];
  description = "Pretty-printing library";
  license = "GPL";
}

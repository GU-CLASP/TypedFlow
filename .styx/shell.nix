{ nixpkgs ? import <nixpkgs> {}

 }:
let nixpkgs_source =
fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-21.05.tar.gz";
  nixpkgs' = (import nixpkgs_source){};
in with nixpkgs'.pkgs;
let hp = haskellPackages.override{
    overrides = self: super: {
      typedflow = self.callPackage ./typedflow.nix {};
      };};
     getHaskellDeps = ps: path:
        let f = import path;
            gatherDeps = { buildDepends ? [], libraryHaskellDepends ? [], executableHaskellDepends ? [], libraryToolDepends ? [], executableToolDepends ? [], ...}:
               buildDepends ++ libraryHaskellDepends ++ executableHaskellDepends ++ libraryToolDepends ++ executableToolDepends;
            x = f (builtins.intersectAttrs (builtins.functionArgs f)
                                               (ps // 
                                                nixpkgs'.pkgs) # can also depend on non-haskell packages
                   // {lib = lib; mkDerivation = gatherDeps;});
        in x;
ghc = hp.ghcWithPackages (ps: with ps; lib.lists.subtractLists
[typedflow]
([ cabal-install 
QuickCheck hscolour
  ]  ++ getHaskellDeps ps ./typedflow.nix));
in
pkgs.stdenv.mkDerivation {
  name = "my-haskell-env-0";
  buildInputs = [ glibcLocales ghc ];
  shellHook = ''
 export LANG=en_US.UTF-8
 eval $(egrep ^export ${ghc}/bin/ghc)
'';
}

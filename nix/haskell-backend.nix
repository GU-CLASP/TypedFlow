{ bootstrap ? import <nixpkgs> {} }:

let # pkgsSource = fetchTarball "https://github.com/NixOS/nixpkgs/archive/7098bcac278a2d028036bb3a23508fd1c52155ac.tar.gz";
    pkgsSource = fetchTarball "https://github.com/NixOS/nixpkgs-channels/archive/nixos-18.03.tar.gz";
    gPostUnpack = (drv: ''
                 mv source source0
                 mv source0/${drv.pname} source
                 mv source0/third_party .
                 '');
    overlays = [(self: super:
         let tensorflow-source = super  .fetchFromGitHub {
                      owner = "tensorflow";
                      repo = "haskell";
                      rev = "baa501b26257c4d4fbf92b439a3f0c1440597187";
                      sha256 = "0xckb99aach3vn71r0d63dz0wapja0l981dlj5z4qj7fp6aw8d36";
                      fetchSubmodules = true;
                   };
         in {
           tensorflow-c = super.callPackage ./tf-c.nix { };
           haskellPackages = super.haskellPackages.extend (selfHS: superHS: {
             proto-lens-protoc = self.haskell.lib.overrideCabal superHS.proto-lens-protoc (drv: {
                    version = "0.2.2.3";});
             proto-lens = self.haskell.lib.overrideCabal superHS.proto-lens (drv: {
                    version = "0.2.2.0";});
             proto-lens-protobuf-types = self.haskell.lib.overrideCabal superHS.proto-lens-protobuf-types (drv: {
                 buildTools = [ self.protobuf3_4 ];
                 version = "0.2.2.0";});
             tensorflow-proto = self.haskell.lib.overrideCabal superHS.tensorflow-proto (drv: {
               buildTools = [ self.protobuf3_4 ];
               setupHaskellDepends = with selfHS; [ base Cabal proto-lens-protoc proto-lens-protobuf-types ];
               libraryHaskellDepends = with selfHS; [ base proto-lens proto-lens-protoc proto-lens-protobuf-types ];
               # src = tensorflow-source; postUnpack = gPostUnpack drv;
               sha256 = "0s3gkis2m3ciia83ziz7rca61czzj77racmcb8si9jxxgw3rxhkc";
               version = "0.2.0.0";
               });
             tensorflow = self.haskell.lib.overrideCabal superHS.tensorflow (drv: {
               buildTools = [ self.protobuf3_4 ];
               # src = tensorflow-source; postUnpack = gPostUnpack drv;
               librarySystemDepends = [self.tensorflow-c];
               sha256 = "0qlz4fxq2dv5l8krmi8q2g61ng1lhxjyzmv3bcxnc1nz4a1438dl";
               version = "0.2.0.0";});
             tensorflow-opgen = self.haskell.lib.overrideCabal superHS.tensorflow-opgen (drv: {
               buildTools = [ self.protobuf3_4 ];
               # src = tensorflow-source; postUnpack = gPostUnpack drv;
               sha256 = "16d4bgc665synpwcapzffd1kqzvpwvfs97k0fwkxda0lzziy87xq";
               version = "0.2.0.0";});
             tensorflow-core-ops = self.haskell.lib.overrideCabal superHS.tensorflow-core-ops (drv: {
               buildTools = [ self.protobuf3_4 ];
               # src = tensorflow-source; postUnpack = gPostUnpack drv;
               sha256 = "0ii5n2fxx6frkk6cscbn2fywx9yc914n6y9dp84rr4v3vr08ixf0";
               version = "0.2.0.0";});
             tensorflow-ops = self.haskell.lib.overrideCabal superHS.tensorflow-ops (drv: {
               # src = tensorflow-source; postUnpack = gPostUnpack drv;
               sha256 = "12x37bh8172xkgnp5ahr87ykad8gbsnb4amchpjcwxap33n9h19c";
               version = "0.2.0.0";});
            });
    })];
  pkgs = import pkgsSource {inherit overlays; config = { allowUnfree = true; };};
in
  pkgs.stdenv.mkDerivation {
    name = "tensorflow-env";
    buildInputs = [(pkgs.haskellPackages.ghcWithPackages (hps: with hps; [ tensorflow-ops tensorflow-core-ops tensorflow-opgen tensorflow base containers ghc-typelits-knownnat mtl pretty-compact ]))];
  }

{ bootstrap ? import <nixpkgs> {} }:
let nixpkgs_source = fetchTarball https://github.com/NixOS/nixpkgs/archive/nixos-20.03.tar.gz;
    # nixpkgs_source = fetchTarball https://github.com/NixOS/nixpkgs/archive/4cf0b6ba5d5ab5eb20a88449e0612f4dad8e4c29.tar.gz;
    # nixpkgs_source = bootstrap.fetchFromGitHub { # for safety of checking the hash
    #    owner = "jyp";
    #    repo = "nixpkgs";
    #    rev = "6b911c2d99ad116fca338fc26de86b8859079322";
    #    sha256 = "1bhwjkynya653mvpc4wwqks6kxnc06gyw6sbpwp8dbyr444ms4bd";
    #  };
    # nixpkgs_source = ~/repo/nixpkgs;

in with (import nixpkgs_source {}).pkgs;
let py = (pkgs.python37.withPackages (ps: [ps.tensorflow-bin_2 ps.nltk]));

in pkgs.stdenv.mkDerivation {
  name = "my-env-0";
  buildInputs = [ py ];
}


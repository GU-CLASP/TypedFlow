{ fetchurl, stdenv, cudatoolkit, cudnn, symlinkJoin, zlib, linuxPackages }:

let cudatoolkit_joined = symlinkJoin {
        name = "${cudatoolkit.name}-unsplit";
        paths = [ cudatoolkit.out cudatoolkit.lib ];
    };
cudaSupport = false; # does not work because the pre-compiled binary is for CUDA 9.0 and we have CUDA 9.1 T_T
rpath = if cudaSupport
        then stdenv.lib.makeLibraryPath [ stdenv.cc.cc.lib zlib cudatoolkit_joined cudnn linuxPackages.nvidia_x11 ]
        else stdenv.lib.makeLibraryPath [ stdenv.cc.cc.lib zlib  ];
in stdenv.mkDerivation rec {
  pname = "tensorflow-c";
  version = "1.9.0";
  name  = "${pname}-${version}";

  src = if cudaSupport
        then fetchurl {
               url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-${version}.tar.gz";
               sha256 = "0m1g4sqr9as0jgfx7wlyay2nkad6wgvsyk2gvhfkqkq5sm1vbx85";
              }
        else fetchurl {
              url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-${version}.tar.gz";
              sha256 = "0l9ps115ng5ffzdwphlqmj3jhidps2v5afppdzrbpzmy41xz0z21";
             };

  buildCommand = ''
   . $stdenv/setup
   mkdir -pv $out
   tar -C $out -xzf $src
   rrPath="$out/lib/:${rpath}"
   find $out/lib -name '*.so' -exec chmod u+w {} \; # so we can patch later
   echo "RPATH for $out/lib/... = $rrPath"
   find $out/lib -name '*.so' -exec patchelf --set-rpath "$rrPath" {} \;
  '';
}


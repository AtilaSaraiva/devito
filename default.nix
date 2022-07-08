{ sources ? import ./nix/sources.nix }:
with import sources.nixpkgs {
  overlays = [
    (import (builtins.fetchTarball https://github.com/AtilaSaraiva/myNixPythonPackages/archive/main.tar.gz))
  ];
};
with python39Packages;

buildPythonPackage rec {
  pname = "devito";
  version = "master";

  src = ./.;

  postPatch = ''
    # Removing unecessary dependencies
    sed -e "s/flake8.*//g" \
        -e "s/codecov.*//g" \
        -e "s/pytest.*//g" \
        -e "s/pytest-runner.*//g" \
        -e "s/pytest-cov.*//g" \
        -i requirements.txt

    # Relaxing dependencies requirements
    sed -e "s/>.*//g" \
        -e "s/<.*//g" \
        -i requirements.txt
  '';

  doCheck = false;

  propagatedBuildInputs = [
    anytree
    nbval
    multidict
    distributed
    pyrevolve
    codepy
    sympy
    psutil
    py-cpuinfo
    cgen
    click
    scipy
    cached-property
    numpy
  ];

  pythonImportsCheck = [ "devito" ];

  shellHooks = ''
    export DEVITO_ARCH="gcc"
    export DEVITO_LANGUAGE="openmp"
    cd atila
  '';
}

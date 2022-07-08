with import "/home/atila/Files/Códigos/gitRepos/nixpkgs" { };
with python39Packages;

buildPythonPackage rec {
  pname = "devito";
  version = "4.6.2";

  src = ../.;

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
    matplotlib
    numpy
    python39Packages.pytest
  ];

  pythonImportsCheck = [ "devito" ];

  shellHooks = ''
    export DEVITO_ARCH="gcc"
    export DEVITO_LANGUAGE="openmp"
  '';
}

{
  description = "Kosmic Lab development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311;
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          numpy
          scipy
          pandas
          networkx
          tqdm
          pyyaml
          jsonschema
          pytest
          pip
        ]);
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.poetry
            pkgs.git
            # LaTeX for manuscript compilation
            pkgs.texlive.combined.scheme-full
          ];
          shellHook = ''
            export PYTHONUNBUFFERED=1
            export PATH="$HOME/.cargo/bin:$PATH"
            export KOSMIC_LAB_ROOT=${toString ./.}
            echo "Kosmic Lab dev shell activated (with LaTeX)."
          '';
        };
      });
}

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
        projectDir = ./.;
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
          black
        ]);
        runTestsScript = pkgs.writeShellApplication {
          name = "kosmic-run-tests";
          runtimeInputs = [ pythonEnv pkgs.poetry ];
          text = ''
            cd ${projectDir}
            poetry run pytest --maxfail=1 --disable-warnings -q
          '';
        };
        runLintScript = pkgs.writeShellApplication {
          name = "kosmic-run-lint";
          runtimeInputs = [ pythonEnv pkgs.poetry ];
          text = ''
            cd ${projectDir}
            poetry run black --check .
          '';
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.poetry
            pkgs.git
            pkgs.texlive.combined.scheme-full
          ];
          shellHook = ''
            export PYTHONUNBUFFERED=1
            export KOSMIC_LAB_ROOT=${toString projectDir}
            echo "Kosmic Lab dev shell activated (with LaTeX)."
          '';
        };

        apps = {
          run-tests = {
            type = "app";
            program = "${runTestsScript}/bin/kosmic-run-tests";
          };
          run-lint = {
            type = "app";
            program = "${runLintScript}/bin/kosmic-run-lint";
          };
          run-archive-verify = {
            type = "app";
            program = "${pkgs.writeShellApplication {
              name = "kosmic-archive-verify";
              runtimeInputs = [ pythonEnv pkgs.poetry ];
              text = ''
                if [ -z "$1" ]; then
                  echo "Usage: kosmic-archive-verify ARCHIVE.tar.gz"
                  exit 1
                fi
                cd ${projectDir}
                poetry run python scripts/archive_tool.py verify --archive "$1"
              '';
            }}/bin/kosmic-archive-verify";
          };
          run-archive-summary = {
            type = "app";
            program = "${pkgs.writeShellApplication {
              name = "kosmic-archive-summary";
              runtimeInputs = [ pythonEnv pkgs.poetry ];
              text = ''
                if [ -z "$1" ]; then
                  echo "Usage: kosmic-archive-summary ARCHIVE.tar.gz"
                  exit 1
                fi
                cd ${projectDir}
                poetry run python scripts/archive_tool.py summary --archive "$1"
              '';
            }}/bin/kosmic-archive-summary";
          };
          default = apps.run-tests;
        };

        checks.pytest = pkgs.runCommand "kosmic-pytest" { buildInputs = [ pythonEnv pkgs.poetry ]; } ''
          export HOME=$TMPDIR
          cd ${projectDir}
          poetry run pytest --maxfail=1 --disable-warnings -q
          touch $out
        '';

        checks.archive = pkgs.runCommand "kosmic-archive-check" { buildInputs = [ pythonEnv pkgs.poetry ]; } ''
          export HOME=$TMPDIR
          cd ${projectDir}
          mkdir -p tmp/checks
          cat <<'JSON' > tmp/checks/min_checkpoint.json
{"metadata":{"config":{"path":"synthetic","sha256":"dummy"}},"state":{}}
JSON
          poetry run python scripts/archive_tool.py create --checkpoint tmp/checks/min_checkpoint.json --config fre/configs/track_g_threshold.yaml --output tmp/checks/sample_bundle.tar.gz
          poetry run python scripts/archive_tool.py verify --archive tmp/checks/sample_bundle.tar.gz
          touch $out
        '';

        checks.lint = pkgs.runCommand "kosmic-lint" { buildInputs = [ pythonEnv pkgs.poetry ]; } ''
          export HOME=$TMPDIR
          cd ${projectDir}
          poetry run black --check .
          touch $out
        '';

        checks.registry = pkgs.runCommand "kosmic-registry" { buildInputs = [ pythonEnv pkgs.poetry ]; } ''
          cd ${projectDir}
          poetry run python scripts/validate_registry.py --path configs/config_registry.json
          touch $out
        '';
      });
}

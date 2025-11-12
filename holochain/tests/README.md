# Holochain Integration Tests

`run_sandbox.sh` assumes the `hc` CLI is installed and available on PATH.
Install it from the official release: https://github.com/holochain/holochain/releases

Example setup:
```bash
# Install hc CLI (Linux example)
curl -Ls https://github.com/holochain/holochain/releases/download/v0.3.4/hc-x86_64-unknown-linux-gnu.tar.gz |
  tar xz -C ~/.local/bin hc
export PATH="$HOME/.local/bin:$PATH"
```

The sandbox script expects the DNA manifest at `holochain/dna/dna.yaml`.
If you only want to pack the bundle manually:
```bash
hc dna pack ../dna --output sandbox_workdir/bundle.dna
```

Then run:
```bash
./holochain/tests/run_sandbox.sh
```

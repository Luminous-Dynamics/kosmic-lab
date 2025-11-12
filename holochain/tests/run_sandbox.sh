#!/usr/bin/env bash
# Script shell for running Holochain conductor integration tests.
# Requirements: `hc` binary (bundled under holochain/bin) and nix shell.
# Steps:
#   1. Pack DNA bundle from holochain/dna/dna.yaml
#   2. Instantiate sandbox conductor
#   3. Run cargo integration tests (ignored tests)

set -euo pipefail

HC_BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../bin" && pwd)"

if ! command -v hc >/dev/null 2>&1; then
  if [ -x "${HC_BIN_DIR}/hc" ]; then
    export PATH="${HC_BIN_DIR}:$PATH"
  else
    echo "[ERROR] hc CLI not found."
    echo "Install with:"
    echo "  nix develop -c cargo install holochain_cli \\"
    echo "    --locked --git https://github.com/holochain/holochain \\"
    echo "    --tag holochain-0.6.0-dev.32"
    echo "Or place the binary under ${HC_BIN_DIR}"
    exit 1
  fi
fi

if ! command -v hc >/dev/null 2>&1; then
  echo "[ERROR] hc CLI still unavailable after PATH adjustments."
  exit 1
fi

DNA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/dna"
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_WORKDIR="${BUILD_DIR}/sandbox_workdir"

mkdir -p "${TMP_WORKDIR}"

echo "[1/4] Cleaning previous sandboxes..."
hc sandbox clean 2>/dev/null || true

echo "[2/4] Packing DNA bundle..."
hc dna pack "${DNA_DIR}" --output "${TMP_WORKDIR}/bundle.dna"

cat > "${TMP_WORKDIR}/happ.yaml" <<'EOF'
---
manifest_version: "1"
name: fre-simulation-happ
roles:
  - name: "fre-simulation"
    provisioning:
      strategy: create
      deferred: false
    dna:
      bundled: "./bundle.dna"
      modifiers:
        network_seed: ~
        properties: ~
        origin_time: ~
        quantum_time: ~
EOF

hc app pack "${TMP_WORKDIR}" --output "${TMP_WORKDIR}/bundle.happ"

echo "[3/4] Launching sandbox conductor..."
hc sandbox generate --app-id fre-simulation "${TMP_WORKDIR}/bundle.happ" --run=0 &
HC_PID=$!
sleep 10

echo "Listing sandbox instances:"
hc sandbox list || true

echo "[4/4] Running cargo integration tests (ignored)..."
if [ -f Cargo.toml ]; then
  cargo test --test integration -- --ignored
else
  echo "[WARN] Cargo.toml not found; skipping integration tests."
fi

echo "Stopping conductor (PID ${HC_PID})"
kill "${HC_PID}" 2>/dev/null || true
hc sandbox clean 2>/dev/null || true

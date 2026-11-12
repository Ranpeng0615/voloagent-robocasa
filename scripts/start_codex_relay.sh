#!/usr/bin/env bash
# Run on the authenticated local computer while server episodes use Codex.
set -euo pipefail
code="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$code/.runtime/codex-relay"
relay_python="${CODEX_RELAY_PYTHON:-/home/zeno-rp/volo-repro/envs/orchestrator/bin/python}"
export ROBOCASA_ROOT="$code"
export RUN_DIR="$code/.runtime/codex-relay"
export CODEX_BRIDGE_UPSTREAM=""
export CODEX_MODEL_CATALOG="${CODEX_MODEL_CATALOG:-$HOME/.codex/models_cache.json}"
"$relay_python" -m uvicorn codex_bridge:app --app-dir "$code/scripts" \
  --host 127.0.0.1 --port 18111 &
relay_pid=$!
cleanup() {
  kill "$relay_pid" 2>/dev/null || true
  ssh -S "$code/work/server-ssh.sock" -O cancel -p 46442 \
    -R 127.0.0.1:18111:127.0.0.1:18111 root@183.249.99.226 2>/dev/null || true
}
trap cleanup EXIT
ssh -S "$code/work/server-ssh.sock" -O forward -o ExitOnForwardFailure=yes \
  -p 46442 -R 127.0.0.1:18111:127.0.0.1:18111 root@183.249.99.226
wait "$relay_pid"

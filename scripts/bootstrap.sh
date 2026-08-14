#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo 'missing uv; install it before bootstrapping fineQComp' >&2
  exit 2
fi
uv sync --extra dev

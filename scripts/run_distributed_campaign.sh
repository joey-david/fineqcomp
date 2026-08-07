#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
remote_root="${REMOTE_REPO_ROOT:-/home/lamsade/jdavid/fineQComp}"
session="${CAMPAIGN_SESSION:-fineqcomp5}"
shards=5

usage() {
  echo "usage: $0 [--prepare]"
}

case "${1:-}" in
  "") ;;
  --prepare)
    "$python_bin" -m fineqcomp prepare --config configs/campaign.yaml
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ ! -s prepared/manifest.jsonl ]]; then
  echo "missing prepared/manifest.jsonl; run $0 --prepare first" >&2
  exit 2
fi

start_host() {
  local host="$1"
  shift
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" bash -s -- \
    "$remote_root" "$session" "$shards" "$@" <<'REMOTE'
set -euo pipefail

repo_root="$1"
session="$2"
shards="$3"
shift 3

if (( $# == 0 || $# % 2 != 0 )); then
  echo "expected GPU/shard pairs" >&2
  exit 2
fi
if [[ ! -d "$repo_root" ]]; then
  echo "missing remote checkout: $repo_root" >&2
  exit 2
fi
cd "$repo_root"
if [[ ! -s prepared/manifest.jsonl ]]; then
  echo "missing prepared/manifest.jsonl on $HOSTNAME" >&2
  exit 2
fi
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
python_bin="${PYTHON:-$repo_root/.venv/bin/python}"

if tmux has-session -t "$session" 2>/dev/null; then
  echo "$HOSTNAME: tmux session $session already exists; leaving it untouched"
  exit 0
fi

tmux new-session -d -s "$session" -n "shard$2" -c "$repo_root" \
  env CUDA_VISIBLE_DEVICES="$1" "$python_bin" -m fineqcomp run \
    --shard "$2" --shards "$shards"
shift 2
tmux set-option -t "$session" remain-on-exit on
while (( $# )); do
  gpu="$1"
  shard="$2"
  tmux new-window -d -t "$session:" -n "shard$shard" -c "$repo_root" \
    env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m fineqcomp run \
      --shard "$shard" --shards "$shards"
  shift 2
done
echo "$HOSTNAME: started tmux session $session"
REMOTE
}

status=0
start_host "${KAISERTROT_HOST:-kaisertrot}" 0 0 1 1 &
pids=("$!")
start_host "${OURASI_HOST:-ourasi}" 0 2 1 3 &
pids+=("$!")
start_host "${UPNQUICK_HOST:-upnquick}" 1 4 &
pids+=("$!")
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"

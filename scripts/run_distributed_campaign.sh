#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

python_bin="${PYTHON:-$repo_root/../reasoning/.venv/bin/python}"
remote_root="${REMOTE_REPO_ROOT:-/home/lamsade/jdavid/fineQComp}"
session="${CAMPAIGN_SESSION:-fineqcomp8}"

case "${1:-}" in
  "") ;;
  --prepare)
    "$python_bin" -m fineqcomp prepare --config configs/campaign.yaml
    ;;
  -h|--help)
    echo "usage: $0 [--prepare]"
    exit 0
    ;;
  *)
    echo "usage: $0 [--prepare]" >&2
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
    "$remote_root" "$session" "$@" <<'REMOTE'
set -euo pipefail

repo_root="$1"
session="$2"
shift 2

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
python_bin="${PYTHON:-$repo_root/../reasoning/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$repo_root/src:$repo_root/vendor}"
export HF_HOME="${HF_HOME:-$repo_root/../fineQComp_hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
"$python_bin" -c 'import bitsandbytes, fineqcomp, torch; assert torch.cuda.device_count() >= 2'

if tmux has-session -t "$session" 2>/dev/null; then
  echo "$HOSTNAME: tmux session $session already exists; leaving it untouched"
  exit 0
fi

tmux new-session -d -s "$session" -n gpu0 -c "$repo_root" \
  env CUDA_VISIBLE_DEVICES=0 "$python_bin" -m fineqcomp run \
    --shard 0 --shards 1 "$@"
tmux set-option -t "$session" remain-on-exit on
tmux new-window -d -t "$session:" -n gpu1 -c "$repo_root" \
  env CUDA_VISIBLE_DEVICES=1 "$python_bin" -m fineqcomp run \
    --shard 0 --shards 1 "$@"
echo "$HOSTNAME: started tmux session $session"
REMOTE
}

low_vram=(
  --models qwen3_8b_base mistral_7b_base
  --adapters seeded_last1_r4 seeded_last4_r4 seeded_last4_r16
  --backbones nf4
  --max-length 192
  --micro-batch-size 1
)

status=0
start_host "${UPNQUICK_HOST:-upnquick}" &
pids=("$!")
start_host "${OURASI_HOST:-ourasi}" &
pids+=("$!")
start_host "${BOLDEAGLE_HOST:-boldeagle}" "${low_vram[@]}" &
pids+=("$!")
start_host "${READYCASH_HOST:-readycash}" "${low_vram[@]}" &
pids+=("$!")
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"

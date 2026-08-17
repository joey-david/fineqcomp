#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

python_bin="${FINEQCOMP_PYTHON:-$repo_root/../reasoning/.venv/bin/python}"
remote_root="${REMOTE_REPO_ROOT:-/home/lamsade/jdavid/fineQComp}"
session="${CAMPAIGN_SESSION:-fineqcomp-lit-$(date +%m%d-%H%M)}"
launch_id="${CAMPAIGN_LAUNCH_ID:-$session}"
mode="${1:-}"

case "$mode" in
"") ;;
--prepare)
  "$python_bin" -m fineqcomp prepare --config configs/campaign.yaml
  ;;
--pilot-then-full) ;;
-h | --help)
  echo "usage: $0 [--prepare|--pilot-then-full]"
  exit 0
  ;;
*)
  echo "usage: $0 [--prepare|--pilot-then-full]" >&2
  exit 2
  ;;
esac

if [[ ! -s prepared/manifest.jsonl ]]; then
  echo "missing prepared/manifest.jsonl; run $0 --prepare first" >&2
  exit 2
fi

coord="$repo_root/remote_logs/$launch_id"
if [[ -e "$coord" ]]; then
  echo "launch path already exists: $coord" >&2
  exit 2
fi
mkdir -p "$coord"

start_host() {
  local host="$1"
  local first_shard="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" bash -s -- \
    "$remote_root" "$session" "$first_shard" "$launch_id" <<'REMOTE'
set -euo pipefail

repo_root="$1"
session="$2"
first_shard="$3"
launch_id="$4"

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
python_bin="${FINEQCOMP_PYTHON:-$repo_root/../reasoning/.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$repo_root/src:$repo_root/vendor}"
export HF_HOME="${HF_HOME:-$repo_root/../fineQComp_hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
"$python_bin" -c 'import bitsandbytes, fineqcomp, math_verify, rouge_score, torch; assert torch.cuda.device_count() >= 2'

if tmux has-session -t "$session" 2>/dev/null; then
  echo "$HOSTNAME: tmux session $session already exists; leaving it untouched"
  exit 0
fi

pilot_args() {
  case "$1" in
    0) printf '%s %s\n' mistral_7b_base metamath ;;
    1) printf '%s %s\n' qwen25_7b_base magicoder ;;
    2) printf '%s %s\n' mistral_7b_base xsum ;;
    3) printf '%s %s\n' qwen25_7b_base metamath ;;
  esac
}

read -r model0 dataset0 < <(pilot_args "$first_shard")
read -r model1 dataset1 < <(pilot_args "$((first_shard + 1))")
tmux new-session -d -s "$session" -n gpu0 -c "$repo_root" \
  "$repo_root/scripts/run_campaign_worker.sh" \
  0 "$first_shard" "$launch_id" "$model0" "$dataset0"
tmux set-option -t "$session" remain-on-exit on
tmux new-window -d -t "$session:" -n gpu1 -c "$repo_root" \
  "$repo_root/scripts/run_campaign_worker.sh" \
  1 "$((first_shard + 1))" "$launch_id" "$model1" "$dataset1"
echo "$HOSTNAME: started tmux session $session"
REMOTE
}

status=0
start_host "${UPNQUICK_HOST:-upnquick}" 0 &
pids=("$!")
start_host "${A40_HOST:-coktailjet}" 2 &
pids+=("$!")
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"

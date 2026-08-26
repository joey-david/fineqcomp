#!/usr/bin/env bash
# One frozen-model measurement per model x dataset x prepared seed. PANEL can
# be `discovery` (78 original cells), `prospective` (126 development cells),
# `external`/`qwen_external`/`qwen25_external` (24 cells each), `channel` (the
# 30 prospective cells), `replacement` (the three locked Qwen3 PAWS cells),
# `information_volume` (27 locked multiple-choice cells),
# or `all` (306 cells from the original panels).
# All scalar candidates, correction
# spectra and coverage measures share one forward pass; the layer-energy
# profile adds a short backward pass over LAYER_ENERGY_ROWS rows.
#SBATCH --job-name=fqrelinfo
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --array=0-305%36
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
mkdir -p slurm_logs "${OUT_ROOT:-reports/relative_information}"

source "$repo_root/scripts/jean_zay_env.sh"
export PYTHONUNBUFFERED=1

task_id="${SLURM_ARRAY_TASK_ID:?this script must run as an array}"
panel="${PANEL:-discovery}"
cell="$($PYTHON - "$task_id" "$panel" <<'PY'
import json
import sys

panels = {}
panels["discovery"] = (
    ("configs/model_panel.yaml", "prepared/panel-manifest.jsonl"),
    ("configs/high_gain_panel.yaml", "prepared/highgain-manifest.jsonl"),
    ("configs/diversity_sql.yaml", "prepared/diversity-sql-manifest.jsonl"),
    ("configs/diversity_xbrl.yaml", "prepared/diversity-xbrl-manifest.jsonl"),
)
panels["prospective"] = (
    ("configs/cot_panel.yaml", "prepared/cot-panel-manifest.jsonl"),
    ("configs/behavioural_change.yaml", "prepared/behavioural-manifest.jsonl"),
    ("configs/information_lever.yaml", "prepared/information-lever-manifest.jsonl"),
    ("configs/compressibility.yaml", "prepared/compressibility-manifest.jsonl"),
)
panels["external"] = (
    ("configs/external_llama_panel.yaml", "prepared/external-llama-manifest.jsonl"),
)
panels["qwen_external"] = (
    ("configs/external_qwen3_panel.yaml", "prepared/external-qwen3-manifest.jsonl"),
)
panels["qwen25_external"] = (
    ("configs/external_qwen25_panel.yaml", "prepared/external-qwen25-manifest.jsonl"),
)
panels["channel"] = (
    (
        "configs/prospective_channel_bits.yaml",
        "prepared/prospective-channel-manifest.jsonl",
    ),
)
panels["replacement"] = (
    (
        "configs/prospective_channel_paws.yaml",
        "prepared/prospective-channel-paws-manifest.jsonl",
    ),
)
panels["information_volume"] = (
    (
        "configs/prospective_information_volume_mc.yaml",
        "prepared/prospective-information-volume-mc-manifest.jsonl",
    ),
)
panels["all"] = tuple(
    pair
    for key in (
        "discovery",
        "prospective",
        "external",
        "qwen_external",
        "qwen25_external",
        "channel",
    )
    for pair in panels[key]
)
index = int(sys.argv[1])
panel = sys.argv[2]
try:
    pairs = panels[panel]
except KeyError as error:
    raise SystemExit(f"unknown PANEL={panel!r}; choose {sorted(panels)}") from error
cells = []
for config, manifest in pairs:
    with open(manifest) as stream:
        for line in stream:
            if line.strip():
                cells.append((config, manifest, json.loads(line)["run_id"]))
expected = {
    "discovery": 78,
    "prospective": 126,
    "external": 24,
    "qwen_external": 24,
    "qwen25_external": 24,
    "channel": 30,
    "replacement": 3,
    "information_volume": 27,
    "all": 306,
}[panel]
if len(cells) != expected:
    raise SystemExit(f"expected {expected} {panel} cells, found {len(cells)}")
try:
    print(*cells[index], sep="\t")
except IndexError as error:
    raise SystemExit(f"array index {index} is outside 0..{len(cells) - 1}") from error
PY
)"
IFS=$'\t' read -r config manifest run_id <<<"$cell"

out_root="${OUT_ROOT:-reports/relative_information}"
out="$out_root/$run_id.json"
if [[ -s "$out" && -z "${FORCE:-}" ]]; then
  echo "already complete: $out"
  exit 0
fi

echo "panel=$panel task=$task_id run_id=$run_id rows=${ROWS:-256} out=$out"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec "$PYTHON" -m fineqcomp relative-information \
  --config "$config" \
  --manifest "$manifest" \
  --run-id "$run_id" \
  --prepared-root prepared \
  --out "$out" \
  --rows "${ROWS:-256}" \
  --row-sampling "${ROW_SAMPLING:-uniform}" \
  --row-sample-seed "${ROW_SAMPLE_SEED:-271828}" \
  --max-length "${MAX_LENGTH:-1024}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
  --hidden-dim "${HIDDEN_DIM:-64}" \
  --residual-dim "${RESIDUAL_DIM:-32}" \
  --response-tokens "${RESPONSE_TOKENS:-32}" \
  --sketch-seed "${SKETCH_SEED:-1729}" \
  --ntk-ridge 0.1 \
  --layer-energy-rows "${LAYER_ENERGY_ROWS:-64}" \
  --layer-energy-rank "${LAYER_ENERGY_RANK:-16}"

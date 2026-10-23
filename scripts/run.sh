#!/usr/bin/env bash
# One Docker lifetime manager. Roots are portable and passed into the container.
set -euo pipefail
bazza_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$bazza_repo/scripts/lib/docker_job.sh"

bazza_usage() {
 cat <<'HELP'
Usage: scripts/run.sh train|evaluate|demo --output-dir DIR [options]
  --config FILE.py            Python settings (default: config.py)
  --env-file FILE             Explicit trusted shell environment file
  --root DIR                  BAZZA_ROOT: runtime/artifact root
  --project-root DIR          BAZZA_PROJECT_ROOT: optional project parent
  --data-root DIR             BAZZA_DATA_ROOT: prepared data
  --cache-root DIR            BAZZA_CACHE_ROOT: feature banks
  --model-root DIR            BAZZA_MODEL_ROOT: model weights
  --run-root DIR              BAZZA_RUN_ROOT: existing/new experiment artifacts
  --selection FILE            BAZZA_SELECTION_PATH: existing encoded selection
  --detach | --resume         Docker-owned background job / exact continuation
  --stop | --logs | --status  Manage the job identified by its output directory
  --device cpu | --plan-only  CPU checks / inspect the plan without training
Root precedence: command-line roots > explicit env file > inherited env > defaults.
Inputs are mounted read-only; only the selected output directory is writable.
HELP
}

bazza_stage="${1:-}"
case "$bazza_stage" in
 -h|--help|'') bazza_usage;exit 0;;
 train|evaluate|demo) shift;;
 *) bazza_usage >&2;exit 2;;
esac
# Load only an explicitly requested local file, before command-line overrides.
bazza_arguments=("$@")
bazza_env_file=""
for ((bazza_i=0;bazza_i<${#bazza_arguments[@]};bazza_i++));do
 case "${bazza_arguments[bazza_i]}" in
  --help|-h) bazza_usage;exit 0;;
  --env-file)
   ((bazza_i+1<${#bazza_arguments[@]})) || { echo '--env-file requires a path.' >&2;exit 2; }
   bazza_env_file="$(realpath -m -- "${bazza_arguments[bazza_i+1]}")";;
 esac
done
if [[ -n "$bazza_env_file" ]];then
 [[ -f "$bazza_env_file" ]] || { echo "Environment file missing: $bazza_env_file" >&2;exit 2; }
 set -a
 source "$bazza_env_file"
 set +a
fi
bazza_action=run
bazza_detach=0
bazza_output=""
bazza_config="$bazza_repo/config.py"
bazza_extra=()
while (($#));do
 case "$1" in
  --detach) bazza_detach=1;shift;;
  --stop|--logs|--status) bazza_action="${1#--}";shift;;
  --env-file) shift 2;;
  --config|--output-dir|--root|--project-root|--data-root|--cache-root|--model-root|--run-root|--selection)
   (($#>=2)) || { echo "$1 requires a path." >&2;exit 2; }
   bazza_value="$(realpath -m -- "$2")"
   case "$1" in
    --config) bazza_config="$bazza_value";;
    --output-dir) bazza_output="$bazza_value";;
    --root) export BAZZA_ROOT="$bazza_value";;
    --project-root) export BAZZA_PROJECT_ROOT="$bazza_value";;
    --data-root) export BAZZA_DATA_ROOT="$bazza_value";;
    --cache-root) export BAZZA_CACHE_ROOT="$bazza_value";;
    --model-root) export BAZZA_MODEL_ROOT="$bazza_value";;
    --run-root) export BAZZA_RUN_ROOT="$bazza_value";;
    --selection) export BAZZA_SELECTION_PATH="$bazza_value";;
   esac
   shift 2;;
  *) bazza_extra+=("$1");shift;;
 esac
done
[[ -n "$bazza_output" ]] || { echo '--output-dir is required.' >&2;exit 2; }
bazza_key="$(printf '%s' "$bazza_output" | sha256sum | cut -c1-16)"
bazza_container="bazza-$bazza_stage-$bazza_key"
case "$bazza_action" in
 stop) exec docker stop --timeout 180 "$bazza_container";;
 logs) if [[ "$(docker inspect "$bazza_container" --format '{{.State.Running}}' 2>/dev/null || true)" == true ]];then exec docker logs --follow --tail 30 "$bazza_container";fi;exec tail -n 30 "$bazza_output/console.log";;
 status) for bazza_file in status.json summary.json exit_code.txt;do [[ ! -f "$bazza_output/$bazza_file" ]] || cat "$bazza_output/$bazza_file";done;exit 0;;
esac
[[ -f "$bazza_config" ]] || { echo 'Supply an existing --config FILE.py.' >&2;exit 2; }
export BAZZA_ROOT="$(realpath -m -- "${BAZZA_ROOT:-$bazza_repo/build}")"
export BAZZA_PROJECT_ROOT="$(realpath -m -- "${BAZZA_PROJECT_ROOT:-$(dirname -- "$bazza_repo")}")"
export BAZZA_DATA_ROOT="$(realpath -m -- "${BAZZA_DATA_ROOT:-$BAZZA_ROOT/data}")"
export BAZZA_CACHE_ROOT="$(realpath -m -- "${BAZZA_CACHE_ROOT:-$BAZZA_ROOT/features/bazza}")"
export BAZZA_MODEL_ROOT="$(realpath -m -- "${BAZZA_MODEL_ROOT:-$BAZZA_ROOT/weights}")"
export BAZZA_RUN_ROOT="$(realpath -m -- "${BAZZA_RUN_ROOT:-$BAZZA_ROOT/training/bazza}")"
export BAZZA_SELECTION_PATH="$(realpath -m -- "${BAZZA_SELECTION_PATH:-$BAZZA_DATA_ROOT/groundbeef/v0.5/selection.json}")"
mkdir -p "$bazza_output"
bazza_image="${BAZZA_IMAGE:-bazza-cuda:local}"
bazza_options=(--rm --init --name "$bazza_container" --user "$(id -u):$(id -g)" --network host)
[[ "$bazza_detach" == 0 ]] || bazza_options+=(--detach)
bazza_gpu=1
for bazza_arg in "${bazza_extra[@]}";do
 [[ "$bazza_arg" != cpu && "$bazza_arg" != --plan-only ]] || bazza_gpu=0
done
[[ "$bazza_gpu" == 0 ]] || bazza_options+=(--gpus all)
# Mount only the declared roots and settings; never create absent input roots.
declare -A bazza_mounts=()
for bazza_path in "$bazza_repo" "$(dirname -- "$bazza_config")" "$BAZZA_ROOT" "$BAZZA_PROJECT_ROOT" \
 "$BAZZA_DATA_ROOT" "$BAZZA_CACHE_ROOT" "$BAZZA_MODEL_ROOT" "$BAZZA_RUN_ROOT" "$(dirname -- "$BAZZA_SELECTION_PATH")";do
 [[ ! -d "$bazza_path" ]] || bazza_mounts["$bazza_path"]=ro
done
bazza_mounts["$bazza_output"]=rw
for bazza_path in "${!bazza_mounts[@]}";do
 bazza_options+=(-v "$bazza_path:$bazza_path:${bazza_mounts[$bazza_path]}")
done
for bazza_name in BAZZA_ROOT BAZZA_PROJECT_ROOT BAZZA_DATA_ROOT BAZZA_CACHE_ROOT BAZZA_MODEL_ROOT BAZZA_RUN_ROOT BAZZA_SELECTION_PATH;do
 bazza_options+=(-e "$bazza_name=${!bazza_name}")
done
[[ ! -v OVERFIT_URL ]] || bazza_options+=(-e "OVERFIT_URL=$OVERFIT_URL")
bazza_image_id="$(docker image inspect "$bazza_image" --format '{{.Id}}')"
exec docker run "${bazza_options[@]}" -e PYTHONPATH="$bazza_repo" -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
 -e BAZZA_IMAGE_ID="$bazza_image_id" -w "$bazza_repo" "$bazza_image" \
 bash -c "$bazza_job_entrypoint" bazza "$bazza_output" \
 python -u "$bazza_stage.py" --config "$bazza_config" --output-dir "$bazza_output" "${bazza_extra[@]}"

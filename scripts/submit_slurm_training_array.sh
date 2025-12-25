#!/usr/bin/env bash

# Exit immediately on error (-e flag), treat unset variables as errors (-u flag), and ensure that
# failures within pipelines are detected (-o flag). These safeguards keep the script
# predictable when running on remote compute clusters.
set -euo pipefail

show_help() {
    cat <<'USAGE'
Usage: submit_slurm_training_array.sh [options]

Submit a SLURM job array that trains the BusinessStrategySimulator agent for many configs.

Options:
  -m, --manifest PATH        Manifest JSONL file describing each array task (required)
  -u, --num-updates N        Number of PPO updates to run (default: 500)
  -e, --num-envs N           Number of parallel envs (default: 10)
      --use-gpu              Pass --use_gpu to the training script
      --no-gpu               Do not request GPU training (default)
  -j, --job-name NAME        SLURM job name (default: bssim-train-array)
  -t, --time HH:MM:SS        Walltime limit (default: 04:00:00)
  -p, --partition NAME       SLURM partition/queue name
  -a, --account NAME         SLURM account to charge
      --qos NAME             QoS name
  -n, --nodes N              Number of nodes to request (default: 1)
      --cpus-per-task N      CPUs per task (default: 10)
      --mem SIZE             Memory per node (default: 16G)
      --gres SPEC            Generic resources string (e.g., gpu:1)
      --dependency SPEC      Add a dependency (e.g., afterok:1234)
      --module CMD           Module command to run in the job (can be repeated)
      --venv PATH            Virtual environment to activate (default: .venv)
      --build-dir PATH       Directory containing simulator build outputs (default: build)
      --job-script PATH      Where to write the generated sbatch script
      --extra-sbatch ARG     Extra raw #SBATCH line (can be repeated)
      --array-max-concurrent N  Maximum number of concurrent array tasks
      --dry-run              Generate the script without calling sbatch
  -h, --help                 Show this message and exit
USAGE
}

# Determine the absolute path to the repository root so that relative paths in
# the script continue to work even if the script is invoked from a different
# directory.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Default values for simulator configuration, training run behaviour, and
# SLURM-specific submission parameters. These values can be overridden via
# command line options.
MANIFEST=""
NUM_UPDATES=500
NUM_ENVS=10
USE_GPU=0
JOB_NAME="bssim-train-array"
TIME_LIMIT="04:00:00"
PARTITION=""
ACCOUNT=""
QOS=""
NODES=1
CPUS_PER_TASK=10
MEMORY="16G"
GRES=""
DEPENDENCY=""
MODULE_CMDS=()
VENV_PATH=".venv"
BUILD_DIR="build"
JOB_SCRIPT=""
EXTRA_SBATCH=()
ARRAY_MAX_CONCURRENT=""
DRY_RUN=0

parse_long_opt() {
    local opt="$1"
    local value="$2"
    case "$opt" in
        manifest) MANIFEST="$value" ;;
        num-updates) NUM_UPDATES="$value" ;;
        num-envs) NUM_ENVS="$value" ;;
        use-gpu) USE_GPU=1 ;;
        no-gpu) USE_GPU=0 ;;
        job-name) JOB_NAME="$value" ;;
        time) TIME_LIMIT="$value" ;;
        partition) PARTITION="$value" ;;
        account) ACCOUNT="$value" ;;
        qos) QOS="$value" ;;
        nodes) NODES="$value" ;;
        cpus-per-task) CPUS_PER_TASK="$value" ;;
        mem) MEMORY="$value" ;;
        gres) GRES="$value" ;;
        dependency) DEPENDENCY="$value" ;;
        module) MODULE_CMDS+=("$value") ;;
        venv) VENV_PATH="$value" ;;
        build-dir) BUILD_DIR="$value" ;;
        job-script) JOB_SCRIPT="$value" ;;
        extra-sbatch) EXTRA_SBATCH+=("$value") ;;
        array-max-concurrent) ARRAY_MAX_CONCURRENT="$value" ;;
        dry-run) DRY_RUN=1 ;;
        help) show_help; exit 0 ;;
        *) echo "Unknown option --$opt" >&2; show_help; exit 1 ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--manifest) MANIFEST="$2"; shift 2 ;;
        -u|--num-updates) NUM_UPDATES="$2"; shift 2 ;;
        -e|--num-envs) NUM_ENVS="$2"; shift 2 ;;
        --use-gpu) USE_GPU=1; shift ;;
        --no-gpu) USE_GPU=0; shift ;;
        -j|--job-name) JOB_NAME="$2"; shift 2 ;;
        -t|--time) TIME_LIMIT="$2"; shift 2 ;;
        -p|--partition) PARTITION="$2"; shift 2 ;;
        -a|--account) ACCOUNT="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        -n|--nodes) NODES="$2"; shift 2 ;;
        --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
        --mem) MEMORY="$2"; shift 2 ;;
        --gres) GRES="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --module) MODULE_CMDS+=("$2"); shift 2 ;;
        --venv) VENV_PATH="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --job-script) JOB_SCRIPT="$2"; shift 2 ;;
        --extra-sbatch) EXTRA_SBATCH+=("$2"); shift 2 ;;
        --array-max-concurrent) ARRAY_MAX_CONCURRENT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) show_help; exit 0 ;;
        --*)
            opt_name=${1#--}
            if [[ $# -ge 2 && $2 != --* ]]; then
                parse_long_opt "$opt_name" "$2"
                shift 2
            else
                parse_long_opt "$opt_name" ""
                shift 1
            fi
            ;;
        -*)
            echo "Unknown option $1" >&2
            show_help
            exit 1
            ;;
        *)
            echo "Unexpected positional argument: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

if [[ -z "$MANIFEST" ]]; then
    echo "Manifest file is required. Use --manifest PATH." >&2
    exit 1
fi

make_absolute() {
    local path="$1"
    if [[ -z "$path" ]]; then
        echo ""
    elif [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$REPO_ROOT/$path"
    fi
}

MANIFEST=$(make_absolute "$MANIFEST")
BUILD_DIR=$(make_absolute "$BUILD_DIR")
if [[ -n "$VENV_PATH" ]]; then
    VENV_PATH=$(make_absolute "$VENV_PATH")
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest file not found: $MANIFEST" >&2
    exit 1
fi

task_count=$(wc -l < "$MANIFEST")
if [[ "$task_count" -le 0 ]]; then
    echo "Manifest file contains no tasks: $MANIFEST" >&2
    exit 1
fi

if [[ -n "$BUILD_DIR" && ! -d "$BUILD_DIR" ]]; then
    if [[ "$BUILD_DIR" == "$(make_absolute build)" ]]; then
        echo "Info: skipping missing default build directory '$BUILD_DIR'." >&2
        BUILD_DIR=""
    else
        echo "Warning: build directory '$BUILD_DIR' does not exist." >&2
        BUILD_DIR=""
    fi
fi

if [[ -n "$VENV_PATH" && ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "Warning: virtual environment not found at '$VENV_PATH'." >&2
fi

if [[ -z "$JOB_SCRIPT" ]]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    JOB_SCRIPT="$REPO_ROOT/WorkingFiles/SlurmJobs/train_agent_array_${timestamp}.sbatch"
fi

mkdir -p "$(dirname "$JOB_SCRIPT")"

array_spec="0-$((task_count - 1))"
if [[ -n "$ARRAY_MAX_CONCURRENT" && "$ARRAY_MAX_CONCURRENT" != "0" ]]; then
    array_spec="${array_spec}%${ARRAY_MAX_CONCURRENT}"
fi

{
    printf '#!/bin/bash\n'
    printf '#SBATCH --job-name=%s\n' "$JOB_NAME"
    printf '#SBATCH --nodes=%s\n' "$NODES"
    printf '#SBATCH --time=%s\n' "$TIME_LIMIT"
    printf '#SBATCH --cpus-per-task=%s\n' "$CPUS_PER_TASK"
    printf '#SBATCH --mem=%s\n' "$MEMORY"
    printf '#SBATCH --array=%s\n' "$array_spec"
    if [[ -n "$PARTITION" ]]; then
        printf '#SBATCH --partition=%s\n' "$PARTITION"
    fi
    if [[ -n "$ACCOUNT" ]]; then
        printf '#SBATCH --account=%s\n' "$ACCOUNT"
    fi
    if [[ -n "$QOS" ]]; then
        printf '#SBATCH --qos=%s\n' "$QOS"
    fi
    if [[ -n "$GRES" ]]; then
        printf '#SBATCH --gres=%s\n' "$GRES"
    fi
    if [[ -n "$DEPENDENCY" ]]; then
        printf '#SBATCH --dependency=%s\n' "$DEPENDENCY"
    fi
    for extra in "${EXTRA_SBATCH[@]}"; do
        printf '#SBATCH %s\n' "$extra"
    done
    printf '\nset -euo pipefail\n'
    printf 'cd %q\n' "$REPO_ROOT"
    for module_cmd in "${MODULE_CMDS[@]}"; do
        printf '%s\n' "$module_cmd"
    done
    if [[ -n "$VENV_PATH" ]]; then
        printf 'source %q\n' "$VENV_PATH/bin/activate"
    fi
    if [[ -n "$BUILD_DIR" ]]; then
        printf 'export PYTHONPATH=%q:${PYTHONPATH:-}\n' "$BUILD_DIR"
    fi
    printf 'manifest=%q\n' "$MANIFEST"
    printf 'entry=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$manifest")\n'
    printf 'if [[ -z "$entry" ]]; then\n'
    printf '  echo "No manifest entry for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2\n'
    printf '  exit 2\n'
    printf 'fi\n'
    printf 'BSSIM_ARRAY_ENTRY="$entry" python - <<'"'"'PY'"'"'\n'
    printf 'import json\n'
    printf 'import os\n'
    printf 'import subprocess\n'
    printf '\n'
    printf 'entry = json.loads(os.environ["BSSIM_ARRAY_ENTRY"])\n'
    printf 'cmd = [\n'
    printf '    "python",\n'
    printf '    "business_strategy_gym_env.py",\n'
    printf '    "--config",\n'
    printf '    entry["config"],\n'
    printf '    "--output",\n'
    printf '    entry["output"],\n'
    printf '    "--num_updates",\n'
    printf '    str(entry["num_updates"]),\n'
    printf '    "--num_envs",\n'
    printf '    str(entry["num_envs"]),\n'
    printf ']\n'
    if [[ $USE_GPU -eq 1 ]]; then
        printf 'cmd.append("--use_gpu")\n'
    fi
    printf 'cmd.extend(entry.get("extra_args", []))\n'
    printf 'subprocess.run(cmd, check=True)\n'
    printf 'PY\n'
} > "$JOB_SCRIPT"

chmod 700 "$JOB_SCRIPT"

echo "SLURM array script generated at $JOB_SCRIPT"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run enabled; not submitting job."
    exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch command not found. Submit the script manually when on the cluster." >&2
    exit 0
fi

sbatch "$JOB_SCRIPT"

#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'USAGE'
Usage: submit_slurm_training_job.sh [options] [-- extra_training_args]

Submit a SLURM job that trains the BusinessStrategySimulator PPO agent.

Options:
  -c, --config PATH          Simulator configuration file (default: WorkingFiles/Config/default.json)
  -o, --output PATH          Where to save the trained agent (default: AgentFiles/Agent.zip)
  -u, --num-updates N        Number of PPO updates to run (default: 500)
  -e, --num-envs N           Number of parallel envs (default: 10)
      --use-gpu              Pass --use_gpu to the training script
      --no-gpu               Do not request GPU training (default)
  -j, --job-name NAME        SLURM job name (default: bssim-train)
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
      --dry-run              Generate the script without calling sbatch
  -h, --help                 Show this message and exit

Anything after "--" is forwarded verbatim to business_strategy_gym_env.py.
USAGE
}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

CONFIG="WorkingFiles/Config/default.json"
OUTPUT="AgentFiles/Agent.zip"
NUM_UPDATES=500
NUM_ENVS=10
USE_GPU=0
JOB_NAME="bssim-train"
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
DRY_RUN=0
EXTRA_TRAIN_ARGS=()

parse_long_opt() {
    local opt="$1"
    local value="$2"
    case "$opt" in
        config) CONFIG="$value" ;;
        output) OUTPUT="$value" ;;
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
        dry-run) DRY_RUN=1 ;;
        help) show_help; exit 0 ;;
        *) echo "Unknown option --$opt" >&2; show_help; exit 1 ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config) CONFIG="$2"; shift 2 ;;
        -o|--output) OUTPUT="$2"; shift 2 ;;
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
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) show_help; exit 0 ;;
        --)
            shift
            EXTRA_TRAIN_ARGS=("$@")
            break
            ;;
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

CONFIG=$(make_absolute "$CONFIG")
OUTPUT=$(make_absolute "$OUTPUT")
BUILD_DIR=$(make_absolute "$BUILD_DIR")
if [[ -n "$VENV_PATH" ]]; then
    VENV_PATH=$(make_absolute "$VENV_PATH")
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Configuration file not found: $CONFIG" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$BUILD_DIR" && ! -d "$BUILD_DIR" ]]; then
    echo "Warning: build directory '$BUILD_DIR' does not exist." >&2
fi

if [[ -n "$VENV_PATH" && ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "Warning: virtual environment not found at '$VENV_PATH'." >&2
fi

if [[ -z "$JOB_SCRIPT" ]]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    JOB_SCRIPT="$REPO_ROOT/WorkingFiles/SlurmJobs/train_agent_${timestamp}.sbatch"
fi

mkdir -p "$(dirname "$JOB_SCRIPT")"

{
    printf '#!/bin/bash\n'
    printf '#SBATCH --job-name=%s\n' "$JOB_NAME"
    printf '#SBATCH --nodes=%s\n' "$NODES"
    printf '#SBATCH --time=%s\n' "$TIME_LIMIT"
    printf '#SBATCH --cpus-per-task=%s\n' "$CPUS_PER_TASK"
    printf '#SBATCH --mem=%s\n' "$MEMORY"
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
    printf '\n'
    printf 'python business_strategy_gym_env.py --config %q --output %q --num_updates %s --num_envs %s' "$CONFIG" "$OUTPUT" "$NUM_UPDATES" "$NUM_ENVS"
    if [[ $USE_GPU -eq 1 ]]; then
        printf ' --use_gpu'
    fi
    for arg in "${EXTRA_TRAIN_ARGS[@]}"; do
        printf ' %q' "$arg"
    done
    printf '\n'
} > "$JOB_SCRIPT"

chmod 700 "$JOB_SCRIPT"

echo "SLURM script generated at $JOB_SCRIPT"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run enabled; not submitting job."
    exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch command not found. Submit the script manually when on the cluster." >&2
    exit 0
fi

sbatch "$JOB_SCRIPT"

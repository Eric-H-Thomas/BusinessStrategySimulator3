#!/usr/bin/env bash

# Exit immediately on error (-e flag), treat unset variables as errors (-u flag), and ensure that
# failures within pipelines are detected (-o flag). These safeguards keep the script
# predictable when running on remote compute clusters.
set -euo pipefail

# Print usage information describing the command line interface for the
# submission helper. The block of text below is emitted exactly as written.
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

# Determine the absolute path to the repository root so that relative paths in
# the script continue to work even if the script is invoked from a different
# directory.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Default values for simulator configuration, training run behaviour, and
# SLURM-specific submission parameters. These values can be overridden via
# command line options.
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

# Helper used to process "--long" options that may appear without short
# aliases. It accepts the option name and associated value, updating the
# corresponding configuration variable.
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

# Main argument parsing loop. It consumes the provided CLI options, updating
# script configuration or capturing additional training arguments after "--".
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

# Convert a possibly relative path into an absolute one using the repository
# root as the base. Empty strings are returned untouched to allow optional
# parameters.
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

# Normalize key paths that will be used later so that any relative input works
# regardless of the caller's current working directory.
CONFIG=$(make_absolute "$CONFIG")
OUTPUT=$(make_absolute "$OUTPUT")
BUILD_DIR=$(make_absolute "$BUILD_DIR")
if [[ -n "$VENV_PATH" ]]; then
    VENV_PATH=$(make_absolute "$VENV_PATH")
fi

# Ensure that the required simulator configuration exists before writing or
# submitting a job.
if [[ ! -f "$CONFIG" ]]; then
    echo "Configuration file not found: $CONFIG" >&2
    exit 1
fi

# Make sure the output directory is present so the training process can store
# the resulting agent checkpoint without encountering directory errors.
mkdir -p "$(dirname "$OUTPUT")"

# Warn the user about missing optional resources (build directory and virtual
# environment) so they can adjust the configuration prior to submitting the
# SLURM job.
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

# If no custom job script path was provided, construct a timestamped location
# under WorkingFiles/SlurmJobs to keep generated sbatch files organized.
if [[ -z "$JOB_SCRIPT" ]]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    JOB_SCRIPT="$REPO_ROOT/WorkingFiles/SlurmJobs/train_agent_${timestamp}.sbatch"
fi

mkdir -p "$(dirname "$JOB_SCRIPT")"

# Generate the sbatch job file, line-by-line, using printf to avoid issues with
# shell expansion. The script includes resource directives, optional modules
# and environment setup, and finally the training invocation itself.
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
# Reapply strict shell safety options inside the job script since it runs as a
# separate bash process on the cluster.
    printf '\nset -euo pipefail\n'
    printf 'cd %q\n' "$REPO_ROOT"
    for module_cmd in "${MODULE_CMDS[@]}"; do
        printf '%s\n' "$module_cmd"
    done
    if [[ -n "$VENV_PATH" ]]; then
# Activate the desired virtual environment, if provided, so that Python runs
# with the expected dependencies installed.
        printf 'source %q\n' "$VENV_PATH/bin/activate"
    fi
    if [[ -n "$BUILD_DIR" ]]; then
        # When a build directory is provided, append it to PYTHONPATH so Python
        # can import compiled simulator components during training.
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

# Make the generated job script executable to allow manual inspection and
# execution if desired.
chmod 700 "$JOB_SCRIPT"

# Let the user know where the sbatch file lives for debugging or manual reuse.
echo "SLURM script generated at $JOB_SCRIPT"

# Respect the --dry-run flag by exiting after generating the sbatch file and
# avoiding any submission attempts.
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run enabled; not submitting job."
    exit 0
fi

# If sbatch is unavailable (e.g., running locally), inform the user but still
# exit successfully after generating the script so it can be submitted later.
if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch command not found. Submit the script manually when on the cluster." >&2
    exit 0
fi

sbatch "$JOB_SCRIPT"

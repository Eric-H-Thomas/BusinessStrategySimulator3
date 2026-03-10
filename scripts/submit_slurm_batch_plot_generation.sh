#!/usr/bin/env bash

set -euo pipefail

show_help() {
    cat <<'USAGE'
Usage: submit_slurm_batch_plot_generation.sh [options] FOLDER_A [FOLDER_A ...]

Submit a SLURM job that runs scripts/batch_plot_generation.py for each provided
folder. Each FOLDER_A should contain experiment outputs with nested
MasterOutput_MarketOverlap.zip files.

Options:
  -j, --job-name NAME        SLURM job name (default: bssim-batch-plots)
  -t, --time HH:MM:SS        Walltime limit (default: 02:00:00)
  -p, --partition NAME       SLURM partition/queue name
  -a, --account NAME         SLURM account to charge
      --qos NAME             QoS name
  -n, --nodes N              Number of nodes (default: 1)
      --cpus-per-task N      CPUs per task (default: 4)
      --mem SIZE             Memory per node (default: 8G)
      --dependency SPEC      Add dependency (e.g. afterok:1234)
      --module CMD           Module command to run in the job (repeatable)
      --venv PATH            Virtual environment to activate (default: .venv)
      --job-script PATH      Path to generated sbatch script
      --extra-sbatch ARG     Extra raw #SBATCH line (repeatable)
      --dry-run              Generate script without calling sbatch
  -h, --help                 Show this message and exit
USAGE
}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

JOB_NAME="bssim-batch-plots"
TIME_LIMIT="02:00:00"
PARTITION=""
ACCOUNT=""
QOS=""
NODES=1
CPUS_PER_TASK=4
MEMORY="8G"
DEPENDENCY=""
MODULE_CMDS=()
VENV_PATH=".venv"
JOB_SCRIPT=""
EXTRA_SBATCH=()
DRY_RUN=0
FOLDER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--job-name) JOB_NAME="$2"; shift 2 ;;
        -t|--time) TIME_LIMIT="$2"; shift 2 ;;
        -p|--partition) PARTITION="$2"; shift 2 ;;
        -a|--account) ACCOUNT="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        -n|--nodes) NODES="$2"; shift 2 ;;
        --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
        --mem) MEMORY="$2"; shift 2 ;;
        --dependency) DEPENDENCY="$2"; shift 2 ;;
        --module) MODULE_CMDS+=("$2"); shift 2 ;;
        --venv) VENV_PATH="$2"; shift 2 ;;
        --job-script) JOB_SCRIPT="$2"; shift 2 ;;
        --extra-sbatch) EXTRA_SBATCH+=("$2"); shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) show_help; exit 0 ;;
        --*)
            echo "Unknown option: $1" >&2
            show_help
            exit 1
            ;;
        *)
            FOLDER_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#FOLDER_ARGS[@]} -eq 0 ]]; then
    echo "Error: provide at least one FOLDER_A argument." >&2
    show_help
    exit 1
fi

make_absolute() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$REPO_ROOT/$path"
    fi
}

FOLDERS=()
for folder in "${FOLDER_ARGS[@]}"; do
    abs_folder=$(make_absolute "$folder")
    if [[ ! -d "$abs_folder" ]]; then
        echo "Error: folder does not exist or is not a directory: $abs_folder" >&2
        exit 1
    fi
    FOLDERS+=("$abs_folder")
done

if [[ -n "$VENV_PATH" ]]; then
    VENV_PATH=$(make_absolute "$VENV_PATH")
fi

if [[ -z "$JOB_SCRIPT" ]]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    JOB_SCRIPT="$REPO_ROOT/WorkingFiles/SlurmJobs/batch_plots_${timestamp}.sbatch"
else
    JOB_SCRIPT=$(make_absolute "$JOB_SCRIPT")
fi

mkdir -p "$(dirname "$JOB_SCRIPT")"

SBATCH_LINES=(
    "#SBATCH --job-name=${JOB_NAME}"
    "#SBATCH --time=${TIME_LIMIT}"
    "#SBATCH --nodes=${NODES}"
    "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
    "#SBATCH --mem=${MEMORY}"
    "#SBATCH --output=${JOB_SCRIPT}.out"
    "#SBATCH --error=${JOB_SCRIPT}.err"
)

[[ -n "$PARTITION" ]] && SBATCH_LINES+=("#SBATCH --partition=${PARTITION}")
[[ -n "$ACCOUNT" ]] && SBATCH_LINES+=("#SBATCH --account=${ACCOUNT}")
[[ -n "$QOS" ]] && SBATCH_LINES+=("#SBATCH --qos=${QOS}")
[[ -n "$DEPENDENCY" ]] && SBATCH_LINES+=("#SBATCH --dependency=${DEPENDENCY}")
for extra in "${EXTRA_SBATCH[@]}"; do
    SBATCH_LINES+=("#SBATCH ${extra}")
done

{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo
    for line in "${SBATCH_LINES[@]}"; do
        echo "$line"
    done
    echo
    echo "cd \"$REPO_ROOT\""
    for module_cmd in "${MODULE_CMDS[@]}"; do
        echo "module ${module_cmd}"
    done
    if [[ -n "$VENV_PATH" ]]; then
        echo "if [[ -f \"$VENV_PATH/bin/activate\" ]]; then"
        echo "    source \"$VENV_PATH/bin/activate\""
        echo "else"
        echo "    echo \"Warning: virtual environment not found at $VENV_PATH\" >&2"
        echo "fi"
    fi
    echo
    for folder in "${FOLDERS[@]}"; do
        echo "echo \"Running batch_plot_generation.py for: $folder\""
        echo "python scripts/batch_plot_generation.py \"$folder\""
    done
} > "$JOB_SCRIPT"

chmod +x "$JOB_SCRIPT"

echo "Generated sbatch script: $JOB_SCRIPT"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run enabled; not submitting to SLURM."
    exit 0
fi

sbatch_output=$(sbatch "$JOB_SCRIPT")
echo "$sbatch_output"

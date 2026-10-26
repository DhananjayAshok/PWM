# Shared library sourced by every bash script in this project (always run scripts from the project root).
# Loads the project config and the core virtual environment, defines the argument-handling helpers
# (BASH_TEMPLATE.md), the per-environment activation helper `use_env`, and every script's defaults
# (<NAME>_ESSENTIALS / <NAME>_DEFAULTS / <NAME>_ARG_KEYS) at the bottom of this file.

_env_files=(configs/*.env)
if [[ ! -f "${_env_files[0]}" ]]; then
    echo "No configs/*.env found; run: python configs/create_env_file.py"; exit 1
fi
for _env_file in "${_env_files[@]}"; do
    source "$_env_file" || { echo "Could not source $_env_file"; exit 1; }
done
source setup/.venv/bin/activate || { echo "Core virtual environment not found; run: bash setup/setup.sh"; exit 1; }
# Model weights come from the user's Hugging Face cache; it must already be configured.
[[ -n "${HF_HOME:-}" ]] || { echo "HF_HOME is not set; set it (e.g. in ~/.bashrc) to your Hugging Face cache."; exit 1; }
PROJECT_ROOT=$(pwd) # expects to be run from root, always.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# uv downloads and Python installs live under storage_dir, never under $HOME.
# PWM_UV_CACHE_DIR overrides the cache location (jobs use node-local disk: uv has stalled on the shared filesystem).
export UV_CACHE_DIR="${PWM_UV_CACHE_DIR:-${storage_dir%/}/uv/cache}"
# Not a cache: every built environment's venv runs the interpreter installed here (uv/python/cpython-3.x-*), so
# deleting it breaks them all. Restore with: UV_PYTHON_INSTALL_DIR=<this> uv python install 3.10 3.12
export UV_PYTHON_INSTALL_DIR="${storage_dir%/}/uv/python"
export UV_LINK_MODE=copy
# Built environments (setup/envs/<env>/build.sh) are published here as <env>.tar + <env>/manifest.json ...
export PWM_ENV_DIR="${storage_dir%/}/envs"
# ... after being built in, and unpacked for use to, node-local disk.
export PWM_BUILD_DIR="${PWM_BUILD_DIR:-/tmp/pwm_build}"
export PWM_LOCAL_ENV_DIR="${PWM_LOCAL_ENV_DIR:-/tmp/pwm_envs}"

# args_to_flags <assoc_array_name>
#
# Converts a bash associative array into a flat --key value string suitable
# for passing to a Python click command or another bash script.
# Empty values ("") are emitted as --key none.
#
# Usage (capture-safe — all diagnostics go to stderr):
#   declare -A ARGS=( ["lr"]="0.001" ["dataset"]="" )
#   flags=$(args_to_flags ARGS)
#   python get_strings.py <string_kind> $flags
function args_to_flags() {
    local -n _dict="$1"
    local result=""
    for key in "${!_dict[@]}"; do
        local val="${_dict[$key]}"
        if [[ -z "$val" ]]; then
            val="none"
        fi
        result+="--${key} ${val} "
    done
    echo "${result% }"  # trim trailing space
}

# get_string_from_args <string_kind> <assoc_array_name>
#
# Utility function to get a string from get_strings.py by passing an associative array of args.
# Usage:
#   string=$(get_string_from_args <string_kind> ARGS)
function get_string_from_args() {
    local string_kind="$1"
    shift
    local flags=$(args_to_flags "$1")
    python ${PROJECT_ROOT}/scripts/get_strings.py "$string_kind" $flags
}


# args_to_flags_subset <assoc_array_name> "${KEY_LIST[@]}"
#
# Like args_to_flags, but only emits flags for the specified keys.
# Keys not present in the array are silently skipped.
# Use this when calling a subscript that doesn't accept all of the caller's ARGS.
#
# Usage:
#   subset=$(args_to_flags_subset ARGS REQUESTED_KEYS_ARRAY)
#   bash scripts/a.sh $subset
function args_to_flags_subset() {
    local -n _dict="$1"
    local -n _keys="$2"
    local result=""
    for key in "${_keys[@]}"; do
        if [[ -v _dict["$key"] ]]; then
            local val="${_dict[$key]}"
            if [[ -z "$val" ]]; then val="none"; fi
            result+="--${key} ${val} "
        fi
    done
    echo "${result% }"
}

function populate_dict(){
    local -n _source_dict="$1"
    local -n _target_dict="$2"
    for key in "${!_source_dict[@]}"; do
        _target_dict["$key"]="${_source_dict[$key]}"
    done
}

function populate_array(){
    local -n _source_arr="$1"
    local -n _target_arr="$2"
    _target_arr+=("${_source_arr[@]}")
}

function populate_array_subset(){
    local -n _source_arr="$1"
    local -n _target_arr="$2"
    local -n _subset_keys="$3"
    for key in "${_subset_keys[@]}"; do
        for val in "${_source_arr[@]}"; do
            if [[ "$val" == "$key" ]]; then
                _target_arr+=("$val")
                break
            fi
        done
    done
}

function populate_dict_subset(){
    local -n _source_dict="$1"
    local -n _target_dict="$2"
    local -n _subset_keys="$3"
    for key in "${_subset_keys[@]}"; do
        if [[ -v _source_dict["$key"] ]]; then
            _target_dict["$key"]="${_source_dict[$key]}"
        fi
    done
}


# parse_args <assoc_array_name> <required_array_name> "$@"
#
# The BASH_TEMPLATE.md argument parser as one function: validates that optional
# defaults are non-blank, accepts only --key value pairs for known keys, checks the
# required ones, and prints the active variables. Exits with the usage string on error.
#   declare -A ARGS=( ["port"]="8000" ); REQUIRED_ARGS=("env")
#   parse_args ARGS REQUIRED_ARGS "$@"
function parse_args() {
    local -n _pa_args="$1"
    local -n _pa_req="$2"
    shift 2
    local usage_str="Usage: $0"
    local req opt
    for req in "${_pa_req[@]}"; do usage_str+=" --$req <value>"; done
    for opt in "${!_pa_args[@]}"; do
        if [[ ! " ${_pa_req[*]} " =~ " ${opt} " ]]; then
            if [[ -z "${_pa_args[$opt]}" ]]; then
                echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
            fi
            usage_str+=" [--$opt <value> (default: ${_pa_args[$opt]})]"
        fi
    done
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) echo "$usage_str"; exit 1 ;;
            --*)
                local flag=${1#--}
                if [[ ! " ${_pa_req[*]} ${!_pa_args[*]} " =~ " ${flag} " ]]; then
                    echo "Error: Unknown flag --$flag"; echo "$usage_str"; exit 1
                fi
                _pa_args["$flag"]="$2"
                shift 2
                ;;
            *) echo "Unknown argument: $1"; echo "$usage_str"; exit 1 ;;
        esac
    done
    local failed=false
    for req in "${_pa_req[@]}"; do
        if [[ -z "${_pa_args[$req]}" ]]; then echo "Error: Argument --$req is required."; failed=true; fi
    done
    if [[ "$failed" == true ]]; then echo "$usage_str"; exit 1; fi
    echo "Script: $0 Active variables:"
    for opt in "${!_pa_args[@]}"; do echo "  -$opt = ${_pa_args[$opt]}"; done
}


# use_env <env>
#
# Switches the current shell from the core venv to a simulator/policy/world-model environment built by
# setup/envs/<env>/build.sh. The published tar is unpacked to node-local disk ($PWM_LOCAL_ENV_DIR) unless the copy
# there already has the same manifest, then the venv is activated and setup/envs/<env>/env.sh sourced.
#   use_env robocasa_gr1
function use_env() {
    local env_name="$1"
    local published="${PWM_ENV_DIR}/${env_name}"
    local env_root="${PWM_LOCAL_ENV_DIR%/}/${env_name}"
    [[ -f "${published}/manifest.json" && -f "${published}.tar" ]] || { echo "Environment ${env_name} is not built; run: bash setup/envs/${env_name}/build.sh"; exit 1; }
    # Several processes may activate the same environment at once: one unpacks (under a lock, into a temporary
    # directory renamed into place when complete), the others wait and then find it current.
    mkdir -p "$PWM_LOCAL_ENV_DIR"
    local lock_fd
    exec {lock_fd}>"${PWM_LOCAL_ENV_DIR%/}/.${env_name}.lock"
    flock "$lock_fd"
    if ! cmp -s "${published}/manifest.json" "${env_root}/manifest.json" 2>/dev/null; then
        rm -rf "${env_root}.unpack"; mkdir -p "${env_root}.unpack"
        tar -xf "${published}.tar" -C "${env_root}.unpack" || { echo "Could not unpack ${published}.tar"; flock -u "$lock_fd"; exit 1; }
        rm -rf "$env_root"; mv "${env_root}.unpack" "$env_root"
    fi
    flock -u "$lock_fd"
    exec {lock_fd}>&-
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then deactivate; fi
    export PWM_ACTIVE_ENV="$env_name" PWM_ACTIVE_ENV_ROOT="$env_root"
    source "${env_root}/venv/bin/activate" || { echo "Could not activate ${env_root}/venv"; exit 1; }
    # a missing interpreter (UV_PYTHON_INSTALL_DIR) would otherwise fall through to another python on PATH
    "${env_root}/venv/bin/python" -c "" 2>/dev/null \
        || { echo "Environment ${env_name}: its python does not run (is ${UV_PYTHON_INSTALL_DIR} intact?)"; exit 1; }
    source "${PROJECT_ROOT}/setup/envs/${env_name}/env.sh" || { echo "Could not source setup/envs/${env_name}/env.sh"; exit 1; }
}

# use_core_env: switch back from an environment activated by use_env to the core venv.
function use_core_env() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then deactivate; fi
    unset PWM_ACTIVE_ENV PWM_ACTIVE_ENV_ROOT
    source "${PROJECT_ROOT}/setup/.venv/bin/activate" || { echo "Core virtual environment not found"; exit 1; }
}

################################################################################
# Script defaults: every bash script's optional arguments (<NAME>_DEFAULTS) and required ones
# (<NAME>_ESSENTIALS). Scripts load them with populate_dict / populate_array before parse_args.
################################################################################

# setup/envs/<env>/build.sh
BUILD_ENV_ESSENTIALS=()
declare -A BUILD_ENV_DEFAULTS=(
    ["force"]="false"
)
BUILD_ENV_ARG_KEYS=("${BUILD_ENV_ESSENTIALS[@]}" "${!BUILD_ENV_DEFAULTS[@]}")

# scripts/jobs/verify_robot.sh
VERIFY_ROBOT_ESSENTIALS=("robot")
declare -A VERIFY_ROBOT_DEFAULTS=(
    ["episodes"]="none"   # per task; none = 10 per LIBERO task (protocol), 20 per GR-1 task, 24 per SimplerEnv task
    ["port"]="5555"
    ["n_envs"]="5"        # GR-1: simulator processes per task
)
VERIFY_ROBOT_ARG_KEYS=("${VERIFY_ROBOT_ESSENTIALS[@]}" "${!VERIFY_ROBOT_DEFAULTS[@]}")

# scripts/jobs/build_envs.sh
BUILD_ENVS_ESSENTIALS=()
declare -A BUILD_ENVS_DEFAULTS=(
    ["envs"]="all"        # space-separated environment names, or all
    ["force"]="false"
)
BUILD_ENVS_ARG_KEYS=("${BUILD_ENVS_ESSENTIALS[@]}" "${!BUILD_ENVS_DEFAULTS[@]}")

# scripts/jobs/train_dreamerv3.sh
TRAIN_DREAMERV3_ESSENTIALS=("robot")
declare -A TRAIN_DREAMERV3_DEFAULTS=(
    ["dataset"]="nominal"
    ["run_name"]="none"   # none = dv3-<robot>-<dataset>
    ["ckpt_dir"]="none"   # none = <storage_dir>/checkpoints/<run_name>
    ["overrides"]="none"  # comma-separated key=value over configs/world_models.yaml dreamerv3, e.g. size=large,eval_every=500
)
TRAIN_DREAMERV3_ARG_KEYS=("${TRAIN_DREAMERV3_ESSENTIALS[@]}" "${!TRAIN_DREAMERV3_DEFAULTS[@]}")

# scripts/jobs/eval_dreamerv3.sh
EVAL_DREAMERV3_ESSENTIALS=("robot")
declare -A EVAL_DREAMERV3_DEFAULTS=(
    ["mode"]="predict"        # predict (imagined window) | encode (real-frame features for the probe)
    ["model_dir"]="none"      # none = <model_dir>/dreamerv3/dv3-<robot>-nominal
    ["dataset"]="nominal"
    ["split"]="none"          # none = test (predict) | train (encode)
    ["checkpoint_tag"]="none" # none = the model directory's name
    ["samples"]="1"           # predict: samples per episode (sample 0 = pred.mp4)
    ["max_episodes"]="none"   # encode: none = 2000 on train, all on test/calib; 0 = all
)
EVAL_DREAMERV3_ARG_KEYS=("${EVAL_DREAMERV3_ESSENTIALS[@]}" "${!EVAL_DREAMERV3_DEFAULTS[@]}")

# scripts/jobs/generate.sh
GENERATE_ESSENTIALS=("robot")
declare -A GENERATE_DEFAULTS=(
    ["kind"]="nominal"          # nominal | test | calib
    ["shift"]="none"
    ["level"]="none"
    ["workers"]="8"             # worker processes on this GPU
    ["groups_per_task"]="none"  # test sets: counterfactual groups per task (none = all test starts)
    ["calib_units"]="200"       # calibration pools: policy episodes
    ["port"]="5555"
)
GENERATE_ARG_KEYS=("${GENERATE_ESSENTIALS[@]}" "${!GENERATE_DEFAULTS[@]}")

# scripts/jobs/screen_shifts.sh
SCREEN_SHIFTS_ESSENTIALS=()
declare -A SCREEN_SHIFTS_DEFAULTS=(
    ["out"]="none"          # none = <storage_dir>/data/shift_screen/franka_libero
    ["worker"]="0"
    ["num_workers"]="1"
    ["shifts"]="all"        # comma-separated shift names, or all
    ["selftest_robot"]="none"  # a robot name: run pwm.shifts.selftest for it instead of the screen
    ["candidates_key"]="candidates"  # candidates | extension_candidates (second screening round)
    ["confirm"]="none"      # JSON {shift: [values]}: re-screen those values on the confirmation starts
)
SCREEN_SHIFTS_ARG_KEYS=("${SCREEN_SHIFTS_ESSENTIALS[@]}" "${!SCREEN_SHIFTS_DEFAULTS[@]}")

# scripts/jobs/prepare_dreamdojo.sh
PREPARE_DREAMDOJO_ESSENTIALS=()
declare -A PREPARE_DREAMDOJO_DEFAULTS=()
PREPARE_DREAMDOJO_ARG_KEYS=("${PREPARE_DREAMDOJO_ESSENTIALS[@]}" "${!PREPARE_DREAMDOJO_DEFAULTS[@]}")

# scripts/jobs/train_dreamdojo.sh
TRAIN_DREAMDOJO_ESSENTIALS=("robot")
declare -A TRAIN_DREAMDOJO_DEFAULTS=(
    ["dataset"]="nominal"
    ["run_name"]="none"       # none = dd-<robot>-<dataset>
    ["run_dir"]="none"        # none = <storage_dir>/checkpoints/<run_name>
    ["init"]="none"           # inference-format weights to start from; none = released 2B pretrain
    ["gpus"]="all"            # processes (one per GPU); all = every visible GPU
    ["max_steps"]="0"         # 0 = until the validation-loss stopping rule
    ["val_every_clips"]="0"   # 0 = configs/world_models.yaml (smoke tests override)
    ["save_every"]="0"        # 0 = time-based checkpoints only
    ["port"]="12355"
)
TRAIN_DREAMDOJO_ARG_KEYS=("${TRAIN_DREAMDOJO_ESSENTIALS[@]}" "${!TRAIN_DREAMDOJO_DEFAULTS[@]}")

# scripts/jobs/eval_dreamdojo.sh
EVAL_DREAMDOJO_ESSENTIALS=("robot")
declare -A EVAL_DREAMDOJO_DEFAULTS=(
    ["mode"]="predict"        # predict (needs run_dir) | encode (real-frame VAE latents for probe training)
    ["run_dir"]="none"        # predict: the training run (weights + action_stats.json)
    ["weights"]="best"        # predict: best | latest | a path to inference-format weights
    ["checkpoint_tag"]="none" # predict: none = <run name>-<weights>; encode: none = vae
    ["dataset"]="nominal"
    ["split"]="test"
    ["episodes"]="none"       # optional file of episode ids to restrict to
    ["max_episodes"]="0"      # encode: deterministic sample of at most this many (0 = all)
    ["seed"]="0"
)
EVAL_DREAMDOJO_ARG_KEYS=("${EVAL_DREAMDOJO_ESSENTIALS[@]}" "${!EVAL_DREAMDOJO_DEFAULTS[@]}")

# scripts/jobs/train_vjepa2.sh
TRAIN_VJEPA2_ESSENTIALS=("robot")
declare -A TRAIN_VJEPA2_DEFAULTS=(
    ["dataset"]="nominal"
    ["ckpt_dir"]="none"       # none = <storage_dir>/checkpoints/vjepa2/<robot>/<dataset>
    ["cache_workers"]="16"    # parallel frame-cache decoders before training
)
TRAIN_VJEPA2_ARG_KEYS=("${TRAIN_VJEPA2_ESSENTIALS[@]}" "${!TRAIN_VJEPA2_DEFAULTS[@]}")

# scripts/jobs/eval_vjepa2.sh
EVAL_VJEPA2_ESSENTIALS=("robot" "dataset" "mode")
declare -A EVAL_VJEPA2_DEFAULTS=(
    ["split"]="test"
    ["checkpoint"]="none"     # best.pt of a training run (predict mode)
    ["max_episodes"]="0"      # encode mode: 0 = all, else a fixed random sample
)
EVAL_VJEPA2_ARG_KEYS=("${EVAL_VJEPA2_ESSENTIALS[@]}" "${!EVAL_VJEPA2_DEFAULTS[@]}")

# scripts/jobs/eval_panel.sh
EVAL_PANEL_ESSENTIALS=("model" "robot" "tag")
declare -A EVAL_PANEL_DEFAULTS=(
    ["datasets"]="all"     # comma-separated test sets, or all (nominal-test + every shift-<s>-L<l>)
)
EVAL_PANEL_ARG_KEYS=("${EVAL_PANEL_ESSENTIALS[@]}" "${!EVAL_PANEL_DEFAULTS[@]}")

# scripts/jobs/generate_shift_suite.sh
GENERATE_SHIFT_SUITE_ESSENTIALS=("robot" "shift")
declare -A GENERATE_SHIFT_SUITE_DEFAULTS=(
    ["levels"]="1 2 3 4 5"
    ["groups_per_task"]="1"     # counterfactual groups per task (x 5 noise levels each); see launch_shift_suites.sh
    ["calib_units"]="50"        # calibration-pool episodes per level
    ["workers"]="8"
)
GENERATE_SHIFT_SUITE_ARG_KEYS=("${GENERATE_SHIFT_SUITE_ESSENTIALS[@]}" "${!GENERATE_SHIFT_SUITE_DEFAULTS[@]}")

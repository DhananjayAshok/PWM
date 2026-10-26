# Shared build functions for setup/envs/<env>/build.sh. Sourced, never executed.
#
# An environment is built on node-local disk, in $PWM_BUILD_DIR/<env>/ (shared filesystems are far too slow for
# venvs and simulator assets with tens of thousands of small files):
#   venv/           relocatable uv venv, synced from setup/envs/<env>/pyproject.toml + uv.lock
#   src/<name>/     upstream code at the commit pinned in setup/envs/<env>/sources.txt, with patches applied
#   assets/         downloaded simulator assets (environment specific)
#   manifest.json   what was built from what (definition hash, source commits, patches)
# and published to storage_dir as two files only: $PWM_ENV_DIR/<env>.tar and $PWM_ENV_DIR/<env>/manifest.json.
# use_env (scripts/utils.sh) unpacks the tar to node-local disk. A build is skipped while the published manifest's hash
# matches the definition.

# env_inputs_hash <env_dir>: hash of everything that determines the build (lock, sources, patches, build/env scripts)
function env_inputs_hash() {
    local env_dir="$1"
    (cd "$env_dir" && find . -type f \( -name 'uv.lock' -o -name 'pyproject.toml' -o -name '.python-version' \
        -o -name 'sources.txt' -o -name '*.diff' -o -name 'build.sh' -o -name 'env.sh' \) | sort | xargs sha256sum) \
        | sha256sum | cut -d' ' -f1
}

# env_build_root <env_name>: fresh local directory to build the environment in
function env_build_root() {
    local root="${PWM_BUILD_DIR%/}/$1"
    rm -rf "$root"
    mkdir -p "$root"
    echo "$root"
}

# env_is_current <env_name> <env_dir>: 0 if the built environment matches the current inputs
function env_is_current() {
    local manifest="${PWM_ENV_DIR}/$1/manifest.json"
    [[ -f "$manifest" && -f "${PWM_ENV_DIR}/$1.tar" ]] || return 1
    grep -q "\"inputs_hash\": \"$(env_inputs_hash "$2")\"" "$manifest"
}

# fetch_sources <env_dir> <dest>: clone every "name url commit submodules" line of sources.txt at its commit, init only
# the listed submodules (comma-separated paths, or none) and apply setup/envs/<env>/patches/<name>/*.diff in sorted order.
function fetch_sources() {
    local env_dir="$1" dest="$2"
    mkdir -p "$dest"
    [[ -f "$env_dir/sources.txt" ]] || return 0
    while read -r name url commit submodules; do
        [[ -z "$name" || "$name" == \#* ]] && continue
        echo "[source] $name @ $commit (submodules: ${submodules:-none})"
        rm -rf "$dest/$name"
        git clone --quiet --filter=blob:none "$url" "$dest/$name"
        git -C "$dest/$name" checkout --quiet "$commit"
        if [[ -n "$submodules" && "$submodules" != "none" ]]; then
            IFS=',' read -ra subs <<< "$submodules"
            git -C "$dest/$name" submodule update --quiet --init --recursive -- "${subs[@]}"
        fi
        if [[ -d "$env_dir/patches/$name" ]]; then
            # absolute paths: git -C resolves relative ones inside the cloned repository
            for p in $(ls "$(realpath "$env_dir/patches/$name")"/*.diff 2>/dev/null | sort); do
                echo "[patch] $name <- $(basename "$p")"
                git -C "$dest/$name" apply "$p" || { echo "Patch $p does not apply"; return 1; }
            done
        fi
    done < "$env_dir/sources.txt"
}

# sync_venv <env_dir> <venv_dir>: create a relocatable venv with the pinned Python and sync the locked dependencies.
# The interpreter is uv-managed (under storage_dir, so jobs that mount it find it), never the machine's system Python.
function sync_venv() {
    local env_dir="$1" venv_dir="$2"
    rm -rf "$venv_dir"
    uv venv --relocatable --managed-python --python "$(cat "$env_dir/.python-version")" "$venv_dir"
    UV_PROJECT_ENVIRONMENT="$venv_dir" uv sync --frozen --project "$env_dir"
}

# finish_env <env_name> <env_dir> <build_root>: write the manifest, publish tar + manifest to $PWM_ENV_DIR, and remove
# the local build
function finish_env() {
    local env_name="$1" env_dir="$2" root="$3" published="${PWM_ENV_DIR}/$1"
    python - "$root" "$env_name" "$env_dir" "$(env_inputs_hash "$env_dir")" <<'PY'
import json, os, subprocess, sys, datetime
root, env_name, env_dir, inputs_hash = sys.argv[1:]
sources = {}
for name in sorted(os.listdir(os.path.join(root, "src"))) if os.path.isdir(os.path.join(root, "src")) else []:
    head = subprocess.run(["git", "-C", os.path.join(root, "src", name), "rev-parse", "HEAD"], capture_output=True, text=True)
    sources[name] = head.stdout.strip()
patches = sorted(os.path.relpath(os.path.join(d, f), env_dir) for d, _, fs in os.walk(os.path.join(env_dir, "patches")) for f in fs if f.endswith(".diff"))
manifest = {"env": env_name, "inputs_hash": inputs_hash, "sources": sources, "patches": patches,
            "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
with open(os.path.join(root, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print(json.dumps(manifest, indent=1))
PY
    mkdir -p "$published"
    (cd "$root" && tar -cf "${published}.tar.tmp" .) && mv "${published}.tar.tmp" "${published}.tar"
    cp "$root/manifest.json" "$published/manifest.json.tmp" && mv "$published/manifest.json.tmp" "$published/manifest.json"
    rm -rf "$root"
    echo "[done] $env_name -> ${published}.tar ($(du -sh "${published}.tar" | cut -f1))"
}

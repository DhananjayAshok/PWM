#!/usr/bin/env bash
# Builds the libero environment into $PWM_ENV_DIR/libero (see setup/envs/README.md). Idempotent; --force rebuilds.
#   bash setup/envs/libero/build.sh [--force <true|false>]
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
source setup/envs/common.sh || { echo "Could not source setup/envs/common.sh"; exit 1; }
declare -A ARGS
populate_dict BUILD_ENV_DEFAULTS ARGS
REQUIRED_ARGS=()
populate_array BUILD_ENV_ESSENTIALS REQUIRED_ARGS
parse_args ARGS REQUIRED_ARGS "$@"
set -euo pipefail

ENV_NAME=libero
ENV_SRC="setup/envs/${ENV_NAME}"
if [[ "${ARGS["force"]}" != "true" ]] && env_is_current "$ENV_NAME" "$ENV_SRC"; then
    echo "[skip] ${ENV_NAME} is up to date (${PWM_ENV_DIR}/${ENV_NAME}.tar)"; exit 0
fi
ROOT=$(env_build_root "$ENV_NAME")
mkdir -p "$ROOT/assets" "$ROOT/libero_config"
sync_venv "$ENV_SRC" "$ROOT/venv"

# LIBERO scene assets (hf-libero otherwise downloads them lazily into ~/.cache at first use).
# env.sh writes LIBERO's config for this location (it must exist before libero.libero is imported).
PWM_ACTIVE_ENV_ROOT="$ROOT" source "$ENV_SRC/env.sh"
"$ROOT/venv/bin/python" -c "
from libero.libero.utils.download_utils import download_assets_from_huggingface
import libero.libero, os
download_assets_from_huggingface(download_dir=os.path.join(os.path.dirname(libero.libero.__file__), 'assets'))"

# pi0.5 tokenizer: same ids as the gated google/paligemma-3b-pt-224 tokenizer.
TOK="$ROOT/assets/paligemma_tokenizer"
mkdir -p "$TOK"
curl -fsSL -o "$TOK/tokenizer.model" https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
cat > "$TOK/tokenizer_config.json" <<'JSON'
{"tokenizer_class": "GemmaTokenizer", "add_bos_token": true, "add_eos_token": false, "bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>", "pad_token": "<pad>", "padding_side": "right"}
JSON

finish_env "$ENV_NAME" "$ENV_SRC" "$ROOT"

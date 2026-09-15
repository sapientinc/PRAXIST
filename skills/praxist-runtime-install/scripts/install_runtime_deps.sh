#!/usr/bin/env bash
set -euo pipefail

PRAXIST_RUNTIME_CODEX_INDEX_URL="${PRAXIST_RUNTIME_CODEX_INDEX_URL:-https://pypi.org/simple}"
PRAXIST_RUNTIME_CODEX_RELAY_PACKAGE="${PRAXIST_RUNTIME_CODEX_RELAY_PACKAGE:-codex-relay==0.5.5}"
PRAXIST_RUNTIME_CODEX_SDK_PACKAGE="${PRAXIST_RUNTIME_CODEX_SDK_PACKAGE:-openai-codex==0.147.0}"

usage() {
  cat <<'EOF'
Install Praxist runtime dependencies only.

Usage:
  install_runtime_deps.sh [--repo DIR | --pip-package] [options]

Options:
  --repo DIR          Install from a Praxist source checkout.
  --pip-package      Install the praxist package from the configured pip index.
  --target DIR       Virtualenv path for pip installs. Default: .venv
  --python BIN       Python executable. Default: python3.11, then python3, then python
  --method auto      Source install method: auto, uv, or pip. Default: auto
  --with-storage     Include the optional storage extra.
  --skip-install     Only write requested Praxist config/credentials; do not install deps.
  --provider NAME    Persist PRAXIST_LLM_PROVIDER and map NAME to the provider key env var.
  --api-key-stdin    Read the provider API key from stdin and persist it.
  --api-key-env VAR  Read the provider API key from environment variable VAR and persist it.
  --agent-system VAL Persist PRAXIST_AGENT_SYSTEM.
  --model VAL        Persist PRAXIST_MODEL.
  --config-file PATH User-level Praxist env file. Default: ${XDG_CONFIG_HOME:-$HOME/.config}/praxist/env
  --dry-run          Print commands without executing them.
  -h, --help         Show this help.

Default runtime extras are agents,codex. Source checkouts also install dependency-groups.dev.
Pip package installs omit test/dev/docs dependencies.
Codex SDK and codex-relay use https://pypi.org/simple by default; override
PRAXIST_RUNTIME_CODEX_INDEX_URL when an approved mirror carries both packages.
EOF
}

repo_dir=""
target_dir=".venv"
python_bin=""
method="auto"
pip_package=0
with_storage=0
dry_run=0
skip_install=0
provider=""
api_key_stdin=0
api_key_env=""
agent_system=""
model=""
config_file="${XDG_CONFIG_HOME:-${HOME}/.config}/praxist/env"
credential_status="not requested"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      if [[ $# -lt 2 ]]; then
        echo "error: --repo requires a path" >&2
        exit 2
      fi
      repo_dir="$2"
      shift 2
      ;;
    --pip-package)
      pip_package=1
      shift
      ;;
    --target)
      if [[ $# -lt 2 ]]; then
        echo "error: --target requires a path" >&2
        exit 2
      fi
      target_dir="$2"
      shift 2
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "error: --python requires an executable" >&2
        exit 2
      fi
      python_bin="$2"
      shift 2
      ;;
    --method)
      if [[ $# -lt 2 ]]; then
        echo "error: --method requires auto, uv, or pip" >&2
        exit 2
      fi
      method="$2"
      shift 2
      ;;
    --with-storage)
      with_storage=1
      shift
      ;;
    --skip-install)
      skip_install=1
      shift
      ;;
    --provider)
      if [[ $# -lt 2 ]]; then
        echo "error: --provider requires a provider name" >&2
        exit 2
      fi
      provider="$2"
      shift 2
      ;;
    --api-key-stdin)
      api_key_stdin=1
      shift
      ;;
    --api-key-env)
      if [[ $# -lt 2 ]]; then
        echo "error: --api-key-env requires an environment variable name" >&2
        exit 2
      fi
      api_key_env="$2"
      shift 2
      ;;
    --agent-system)
      if [[ $# -lt 2 ]]; then
        echo "error: --agent-system requires a value" >&2
        exit 2
      fi
      agent_system="$2"
      shift 2
      ;;
    --model)
      if [[ $# -lt 2 ]]; then
        echo "error: --model requires a value" >&2
        exit 2
      fi
      model="$2"
      shift 2
      ;;
    --config-file)
      if [[ $# -lt 2 ]]; then
        echo "error: --config-file requires a path" >&2
        exit 2
      fi
      config_file="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${method}" != "auto" && "${method}" != "uv" && "${method}" != "pip" ]]; then
  echo "error: --method must be auto, uv, or pip" >&2
  exit 2
fi

if [[ "${api_key_stdin}" -eq 1 && -n "${api_key_env}" ]]; then
  echo "error: use only one of --api-key-stdin or --api-key-env" >&2
  exit 2
fi

if [[ ( "${api_key_stdin}" -eq 1 || -n "${api_key_env}" ) && -z "${provider}" ]]; then
  echo "error: --api-key-stdin/--api-key-env requires --provider so the env var is unambiguous" >&2
  exit 2
fi

if [[ "${skip_install}" -eq 1 && -z "${provider}" && -z "${agent_system}" && -z "${model}" ]]; then
  echo "error: --skip-install requires --provider, --agent-system, or --model" >&2
  exit 2
fi

choose_python() {
  if [[ -n "${python_bin}" ]]; then
    printf '%s\n' "${python_bin}"
    return
  fi
  for candidate in python3.11 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  echo "error: no Python executable found" >&2
  exit 1
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${dry_run}" -eq 0 ]]; then
    "$@"
  fi
}

run_in_dir() {
  local cwd="$1"
  shift
  printf '+ cd %q &&' "${cwd}"
  printf ' %q' "$@"
  printf '\n'
  if [[ "${dry_run}" -eq 0 ]]; then
    (cd "${cwd}" && "$@")
  fi
}

install_codex_packages() {
  local py_bin="$1"
  local install_method="$2"
  if [[ "${install_method}" == "uv" ]]; then
    run_cmd uv pip install \
      --python "${py_bin}" \
      --index-url "${PRAXIST_RUNTIME_CODEX_INDEX_URL}" \
      "${PRAXIST_RUNTIME_CODEX_RELAY_PACKAGE}"
    run_cmd uv pip install \
      --python "${py_bin}" \
      --index-url "${PRAXIST_RUNTIME_CODEX_INDEX_URL}" \
      "${PRAXIST_RUNTIME_CODEX_SDK_PACKAGE}"
    return
  fi
  run_cmd "${py_bin}" -m pip install \
    --index-url "${PRAXIST_RUNTIME_CODEX_INDEX_URL}" \
    "${PRAXIST_RUNTIME_CODEX_RELAY_PACKAGE}"
  run_cmd "${py_bin}" -m pip install \
    --index-url "${PRAXIST_RUNTIME_CODEX_INDEX_URL}" \
    "${PRAXIST_RUNTIME_CODEX_SDK_PACKAGE}"
}

is_source_checkout() {
  local dir="$1"
  [[ -f "${dir}/pyproject.toml" ]] || return 1
  [[ -d "${dir}/praxist" ]] || return 1
  grep -q 'name = "praxist"' "${dir}/pyproject.toml"
}

provider_key_var() {
  local raw="${1#model_provider:}"
  raw="${raw,,}"
  case "${raw}" in
    anthropic|anthropic_messages)
      printf '%s\n' "ANTHROPIC_API_KEY"
      ;;
    openrouter)
      printf '%s\n' "OPENROUTER_API_KEY"
      ;;
    orcarouter)
      printf '%s\n' "ORCAROUTER_API_KEY"
      ;;
    deepseek|deepseek_alias)
      printf '%s\n' "DEEPSEEK_API_KEY"
      ;;
    openai|openai_compatible)
      printf '%s\n' "OPENAI_API_KEY"
      ;;
    kimi|moonshot)
      printf '%s\n' "MOONSHOT_API_KEY"
      ;;
    qwen|dashscope)
      printf '%s\n' "DASHSCOPE_API_KEY"
      ;;
    google)
      printf '%s\n' "GOOGLE_API_KEY"
      ;;
    mistral)
      printf '%s\n' "MISTRAL_API_KEY"
      ;;
    groq)
      printf '%s\n' "GROQ_API_KEY"
      ;;
    xai)
      printf '%s\n' "XAI_API_KEY"
      ;;
    brave)
      printf '%s\n' "BRAVE_API_KEY"
      ;;
    *)
      return 1
      ;;
  esac
}

provider_short_name() {
  local raw="${1#model_provider:}"
  raw="${raw,,}"
  case "${raw}" in
    anthropic_messages) printf '%s\n' "anthropic" ;;
    deepseek_alias) printf '%s\n' "deepseek" ;;
    openai_compatible) printf '%s\n' "openai" ;;
    kimi) printf '%s\n' "moonshot" ;;
    dashscope) printf '%s\n' "qwen" ;;
    *) printf '%s\n' "${raw}" ;;
  esac
}

shell_quote() {
  printf '%q' "$1"
}

write_env_var() {
  local file="$1"
  local var="$2"
  local value="$3"
  local dir tmp next

  if [[ "${value}" == *$'\n'* || "${value}" == *$'\r'* ]]; then
    echo "error: refusing to write multiline value for ${var}" >&2
    exit 2
  fi

  dir="$(dirname -- "${file}")"
  if [[ "${dry_run}" -eq 1 ]]; then
    echo "+ write ${var} to ${file} (redacted)"
    return
  fi

  mkdir -p "${dir}"
  chmod 700 "${dir}" 2>/dev/null || true
  tmp="$(mktemp "${dir}/.env.tmp.XXXXXX")"
  next="$(mktemp "${dir}/.env.next.XXXXXX")"
  if [[ -f "${file}" ]]; then
    awk -v var="${var}" '$0 ~ "^[[:space:]]*(export[[:space:]]+)?" var "=" {next} {print}' "${file}" > "${tmp}"
  fi
  {
    cat "${tmp}"
    printf 'export %s=%s\n' "${var}" "$(shell_quote "${value}")"
  } > "${next}"
  mv "${next}" "${file}"
  rm -f "${tmp}"
  chmod 600 "${file}"
}

configure_credentials() {
  local wrote=()
  local provider_key=""
  local provider_short=""
  local api_key=""

  if [[ -n "${provider}" ]]; then
    if ! provider_key="$(provider_key_var "${provider}")"; then
      echo "error: unsupported provider for credential config: ${provider}" >&2
      exit 2
    fi
    provider_short="$(provider_short_name "${provider}")"
    write_env_var "${config_file}" "PRAXIST_LLM_PROVIDER" "${provider_short}"
    wrote+=("PRAXIST_LLM_PROVIDER")

    if [[ "${api_key_stdin}" -eq 1 ]]; then
      IFS= read -r api_key
    elif [[ -n "${api_key_env}" ]]; then
      api_key="${!api_key_env:-}"
      if [[ -z "${api_key}" ]]; then
        echo "error: ${api_key_env} is empty or unset" >&2
        exit 2
      fi
    fi

    if [[ -n "${api_key}" ]]; then
      write_env_var "${config_file}" "${provider_key}" "${api_key}"
      wrote+=("${provider_key}")
    fi
  fi

  if [[ -n "${agent_system}" ]]; then
    write_env_var "${config_file}" "PRAXIST_AGENT_SYSTEM" "${agent_system}"
    wrote+=("PRAXIST_AGENT_SYSTEM")
  fi

  if [[ -n "${model}" ]]; then
    write_env_var "${config_file}" "PRAXIST_MODEL" "${model}"
    wrote+=("PRAXIST_MODEL")
  fi

  if [[ "${#wrote[@]}" -gt 0 ]]; then
    credential_status="wrote ${wrote[*]} to ${config_file}"
  fi
}

read_dev_dependencies() {
  local repo="$1"
  "${python_bin}" - "${repo}/pyproject.toml" <<'PY'
import sys
import tomllib
from pathlib import Path

pyproject = Path(sys.argv[1])
data = tomllib.loads(pyproject.read_text())
for dep in data.get("dependency-groups", {}).get("dev", []):
    print(dep)
PY
}

discover_repo() {
  local candidates=(
    "${PWD}"
    "${PWD}/Praxist"
    "$(dirname -- "${PWD}")/Praxist"
  )
  local dir
  for dir in "${candidates[@]}"; do
    if is_source_checkout "${dir}"; then
      cd "${dir}" && pwd
      return
    fi
  done
  return 1
}

if [[ -n "${provider}" || -n "${agent_system}" || -n "${model}" ]]; then
  configure_credentials
fi

if [[ "${skip_install}" -eq 1 ]]; then
  echo
  echo "Praxist config update complete."
  echo "config file: ${config_file}"
  echo "credentials/config: ${credential_status}"
  echo "activate: set -a; . ${config_file}; set +a"
  exit 0
fi

python_bin="$(choose_python)"

if [[ "${dry_run}" -eq 0 ]]; then
  "${python_bin}" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python >=3.11 is required, got {sys.version.split()[0]}")
print(f"Python OK: {sys.version.split()[0]}")
PY
else
  echo "+ ${python_bin} -c 'assert Python >= 3.11'"
fi

extras=(agents codex)
if [[ "${with_storage}" -eq 1 ]]; then
  extras+=(storage)
fi
extra_csv="$(IFS=,; echo "${extras[*]}")"

if [[ -n "${repo_dir}" && "${pip_package}" -eq 1 ]]; then
  echo "error: use either --repo or --pip-package, not both" >&2
  exit 2
fi

if [[ -z "${repo_dir}" && "${pip_package}" -eq 0 ]]; then
  if discovered="$(discover_repo)"; then
    repo_dir="${discovered}"
  else
    pip_package=1
  fi
fi

if [[ -n "${repo_dir}" ]]; then
  repo_dir="$(cd "${repo_dir}" && pwd)"
  if ! is_source_checkout "${repo_dir}"; then
    echo "error: not a Praxist source checkout: ${repo_dir}" >&2
    exit 1
  fi

  if [[ "${method}" == "uv" || ( "${method}" == "auto" && $(command -v uv >/dev/null 2>&1; echo $?) -eq 0 ) ]]; then
    if ! command -v uv >/dev/null 2>&1; then
      echo "error: --method uv requested, but uv is not on PATH" >&2
      exit 1
    fi
    cmd=(uv sync --group dev)
    for extra in "${extras[@]}"; do
      if [[ "${extra}" == "codex" ]]; then
        continue
      fi
      cmd+=(--extra "${extra}")
    done
    run_in_dir "${repo_dir}" "${cmd[@]}"
    praxist_bin="${repo_dir}/.venv/bin/praxist"
    py_bin="${repo_dir}/.venv/bin/python"
    install_codex_packages "${py_bin}" uv
    dev_dependency_status="installed from dependency-groups.dev"
  else
    if [[ "${target_dir}" != /* ]]; then
      target_dir="${repo_dir}/${target_dir}"
    fi
    run_cmd "${python_bin}" -m venv "${target_dir}"
    run_cmd "${target_dir}/bin/python" -m pip install --upgrade pip
    install_codex_packages "${target_dir}/bin/python" pip
    run_cmd "${target_dir}/bin/python" -m pip install -e "${repo_dir}[${extra_csv}]"
    dev_deps=()
    while IFS= read -r dependency; do
      dev_deps+=("${dependency}")
    done < <(read_dev_dependencies "${repo_dir}")
    if [[ "${#dev_deps[@]}" -gt 0 ]]; then
      run_cmd "${target_dir}/bin/python" -m pip install "${dev_deps[@]}"
    fi
    praxist_bin="${target_dir}/bin/praxist"
    py_bin="${target_dir}/bin/python"
    dev_dependency_status="installed from dependency-groups.dev"
  fi
  install_mode="source checkout"
else
  if [[ "${target_dir}" != /* ]]; then
    target_dir="${PWD}/${target_dir}"
  fi
  run_cmd "${python_bin}" -m venv "${target_dir}"
  run_cmd "${target_dir}/bin/python" -m pip install --upgrade pip
  install_codex_packages "${target_dir}/bin/python" pip
  run_cmd "${target_dir}/bin/python" -m pip install "praxist[${extra_csv}]"
  praxist_bin="${target_dir}/bin/praxist"
  py_bin="${target_dir}/bin/python"
  install_mode="pip package"
  dev_dependency_status="omitted for pip package install"
fi

if [[ "${dry_run}" -eq 0 ]]; then
  run_cmd "${py_bin}" -c 'import praxist; print(praxist.__file__)'
  run_cmd "${py_bin}" -c 'import praxist.product_usage; print(praxist.product_usage.__file__)'
  run_cmd "${py_bin}" -c 'import openai_codex, mcp; from codex_cli_bin import bundled_codex_path; print("Codex SDK, bundled binary, and MCP imports OK:", bundled_codex_path())'
  run_cmd "${praxist_bin}" --help
fi

relay_bin="$(dirname "${py_bin}")/codex-relay"
if [[ "${dry_run}" -eq 0 && ! -x "${relay_bin}" ]]; then
  echo "error: codex-relay was not installed into the Praxist environment" >&2
  exit 1
fi

echo
echo "Praxist runtime dependency install complete."
echo "mode: ${install_mode}"
echo "extras: ${extra_csv}"
echo "product usage: built into Praxist (consent required)"
echo "test/dev deps: ${dev_dependency_status}"
echo "credentials/config: ${credential_status}"
echo "python: ${py_bin}"
echo "praxist: ${praxist_bin}"
echo "codex sdk: openai_codex (Python package)"
echo "codex relay: ${relay_bin}"
if [[ "${dry_run}" -eq 0 ]]; then
  sdk_codex_bin="$("${py_bin}" -c 'from codex_cli_bin import bundled_codex_path; print(bundled_codex_path())')"
  echo "codex sdk binary: ${sdk_codex_bin}"
fi

if command -v codex >/dev/null 2>&1; then
  echo "operator codex CLI: $(command -v codex)"
else
  echo "operator codex CLI: missing (optional; SDK-bundled binary remains available)"
fi

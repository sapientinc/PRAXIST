"""Run-scoped Responses-to-Chat-Completions relay for Codex SDK providers."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_PROVIDER_UPSTREAMS = {
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "orcarouter": "https://api.orcarouter.ai/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "mistral": "https://api.mistral.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "xai": "https://api.x.ai/v1",
}

_PROVIDER_KEY_VARS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "orcarouter": "ORCAROUTER_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
}


@dataclass
class RelayHandle:
    """One relay process owned by a Codex SDK runtime instance."""

    provider: str
    port: int
    process: subprocess.Popen[bytes]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def provider_name(provider_ref: str) -> str:
    """Normalize a model-provider plugin ref to the relay's short name."""

    name = provider_ref.rsplit(":", 1)[-1].strip().lower()
    if name.endswith("_alias"):
        name = name[: -len("_alias")]
    if name == "openai_compatible":
        return "openai"
    return name


def needs_relay(provider: str) -> bool:
    """Return whether the provider requires the private compatibility relay."""

    return provider not in {"", "openai"}


def provider_key_var(provider: str) -> str:
    """Return the credential environment variable for a provider."""

    return _PROVIDER_KEY_VARS.get(provider, "OPENAI_API_KEY")


def start_relay(
    *,
    provider: str,
    api_key: str,
    state_dir: Path,
    upstream_session_id: str | None = None,
    upstream_extra_params: Mapping[str, object] | None = None,
    drop_upstream_params: Sequence[str] = (),
) -> RelayHandle:
    """Start a private relay on an ephemeral localhost port."""

    relay_binary = _relay_binary()
    if relay_binary is None:
        raise RuntimeError(
            "codex-relay is required for this model provider; install praxist[codex]"
        )
    upstream = _PROVIDER_UPSTREAMS.get(provider)
    if upstream is None:
        raise RuntimeError(f"Codex SDK relay does not support provider {provider!r}")
    if not api_key:
        raise RuntimeError(f"{provider_key_var(provider)} is required for provider {provider}")

    state_dir.mkdir(parents=True, exist_ok=True)
    port = _available_port()
    env = {
        key: value
        for key in ("HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "VIRTUAL_ENV")
        if (value := os.environ.get(key))
    }
    env.update(
        {
            provider_key_var(provider): api_key,
            "OPENAI_API_KEY": api_key,
            "CODEX_RELAY_API_KEY": api_key,
        }
    )
    log_path = state_dir / f"relay-{provider}-{port}.log"
    command = [relay_binary, "--port", str(port), "--upstream", upstream]
    extra_params = dict(upstream_extra_params or {})
    if provider == "openrouter" and upstream_session_id:
        extra_params.setdefault("session_id", upstream_session_id)
    if extra_params:
        command.extend(
            [
                "--upstream-extra-params",
                json.dumps(extra_params, separators=(",", ":"), sort_keys=True),
            ]
        )
    dropped = sorted({str(name) for name in drop_upstream_params if str(name)})
    if dropped:
        command.extend(["--drop-upstream-params", json.dumps(dropped, separators=(",", ":"))])
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=env,
            start_new_session=True,
        )
    if not _wait_for_listener(port, process):
        with contextlib.suppress(Exception):
            process.terminate()
            process.wait(timeout=2)
        raise RuntimeError(f"codex-relay failed to listen; inspect {log_path}")
    return RelayHandle(provider=provider, port=port, process=process)


def _relay_binary() -> str | None:
    """Locate the relay installed beside Praxist before consulting ``PATH``."""

    sibling = Path(sys.executable).with_name("codex-relay")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which("codex-relay")


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_listener(
    port: int,
    process: subprocess.Popen[bytes],
    *,
    timeout: float = 10.0,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


__all__ = [
    "RelayHandle",
    "needs_relay",
    "provider_key_var",
    "provider_name",
    "start_relay",
]

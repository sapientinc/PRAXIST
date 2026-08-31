"""Discover saved ChatGPT authentication for the official Codex SDK runtime."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from praxist.core.credentials import CredentialRef

OPENAI_COMPATIBLE_PROVIDER_REF = "model_provider:openai_compatible"
CHATGPT_CREDENTIAL_KEY_PREFIX = "openai_compatible:codex_sdk:chatgpt"
SUBSCRIPTION_ENV_KEYS = (
    "CODEX_ACCESS_TOKEN",
    "CODEX_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)
PROBE_TIMEOUT_SECONDS = 5
_CHATGPT_LOGIN_STATUS_LINES = frozenset({"logged in using chatgpt"})
_PROBE_ENV_KEYS = frozenset(
    {
        "ALL_PROXY",
        "CODEX_HOME",
        "CURL_CA_BUNDLE",
        "DBUS_SESSION_BUS_ADDRESS",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NO_PROXY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "XDG_RUNTIME_DIR",
    }
)


@dataclass
class StagedChatgptHome:
    """Codex home preserving auth identity without sharing file-backed state."""

    path: Path
    credential_store: str
    credential_key_id: str
    remove_on_close: bool = True

    def close(self) -> None:
        """Remove disposable authentication state owned by this instance."""

        if self.remove_on_close:
            shutil.rmtree(self.path, ignore_errors=True)


def discover_chatgpt_credential(
    model_provider_ref: str,
    *,
    codex_bin: str | Path | None = None,
) -> CredentialRef | None:
    """Return a redacted credential reference for a saved ChatGPT login."""

    if model_provider_ref != OPENAI_COMPATIBLE_PROVIDER_REF:
        return None

    verify_chatgpt_login(codex_bin)

    return CredentialRef(
        scope="model_provider",
        provider="openai_compatible",
        target_ref=OPENAI_COMPATIBLE_PROVIDER_REF,
        key_id=chatgpt_credential_key_id(operator_codex_home()),
        source="runtime_session",
    )


def is_chatgpt_subscription_credential(credential: CredentialRef | None) -> bool:
    """Return whether ``credential`` identifies native Codex ChatGPT auth."""

    return bool(
        credential is not None
        and credential.scope == "model_provider"
        and credential.provider == "openai_compatible"
        and credential.target_ref in (None, OPENAI_COMPATIBLE_PROVIDER_REF)
        and credential.key_id.startswith(f"{CHATGPT_CREDENTIAL_KEY_PREFIX}:")
    )


def resolve_codex_binary(codex_bin: str | Path | None = None) -> str:
    """Resolve the same SDK-pinned Codex binary used by runtime execution."""

    explicit = str(codex_bin or os.environ.get("PRAXIST_CODEX_BIN") or "").strip()
    if explicit and explicit != "codex":
        return explicit
    try:
        module = importlib.import_module("codex_cli_bin")
        bundled_codex_path = getattr(module, "bundled_codex_path", None)
        if not callable(bundled_codex_path):
            raise ImportError("codex_cli_bin.bundled_codex_path is unavailable")
    except ImportError as exc:
        raise OSError("the SDK-pinned Codex runtime is not installed") from exc
    return str(bundled_codex_path())


def operator_codex_home(env: Mapping[str, str] | None = None) -> Path:
    """Return the operator-owned Codex home containing saved authentication."""

    source = os.environ if env is None else env
    configured = str(source.get("CODEX_HOME") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    home = str(source.get("HOME") or "").strip()
    if not home and env is not None:
        raise ValueError("CODEX_HOME or HOME is required to locate saved ChatGPT authentication")
    root = Path(home).expanduser() if home else Path.home()
    return (root / ".codex").resolve()


def chatgpt_credential_key_id(codex_home: Path) -> str:
    """Return a stable redacted identity for one operator auth source."""

    resolved_home = codex_home.resolve()
    snapshot = _read_chatgpt_file_auth(resolved_home)
    source = "file" if snapshot is not None else "keyring"
    account_id = snapshot[1] if snapshot is not None else None
    return _credential_key_id(resolved_home, source, account_id)


def stage_chatgpt_home(codex_home: Path) -> StagedChatgptHome:
    """Resolve a safe runtime home without changing the saved-auth identity."""

    source_home = codex_home.resolve()
    snapshot = _read_chatgpt_file_auth(source_home)
    source = "file" if snapshot is not None else "keyring"
    account_id = snapshot[1] if snapshot is not None else None
    credential_key_id = _credential_key_id(source_home, source, account_id)
    if snapshot is None:
        # Codex derives its keyring lookup key from canonical CODEX_HOME.
        return StagedChatgptHome(
            path=source_home,
            credential_store=source,
            credential_key_id=credential_key_id,
            remove_on_close=False,
        )

    path = Path(tempfile.mkdtemp(prefix="praxist-codex-auth-"))
    path.chmod(0o700)
    staged = StagedChatgptHome(
        path=path,
        credential_store=source,
        credential_key_id=credential_key_id,
    )
    try:
        if snapshot is not None:
            destination = path / "auth.json"
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(snapshot[0])
        return staged
    except BaseException:
        staged.close()
        raise


def verify_chatgpt_login(codex_bin: str | Path | None = None) -> None:
    """Require the exact ChatGPT login marker from one Codex binary."""

    try:
        binary = resolve_codex_binary(codex_bin)
        login_status = subprocess.run(
            [binary, "login", "status"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
            env=_subscription_probe_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Timed out after 5 seconds while checking Codex ChatGPT login. "
            "Rerun `praxist setup --profile codex-native` in a local terminal."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "Unable to execute the Codex runtime while checking ChatGPT login. "
            "Install praxist[codex] or set PRAXIST_CODEX_BIN to its path."
        ) from exc

    if login_status.returncode != 0 or not _has_chatgpt_login_status(login_status):
        raise RuntimeError(
            "The SDK-pinned Codex runtime is not logged in with ChatGPT. "
            "Run `praxist setup --profile codex-native` in a local terminal."
        )


def ensure_chatgpt_login(*, allow_interactive: bool) -> bool:
    """Ensure explicit Codex-native mode has a saved ChatGPT login.

    Returns ``True`` when this call launched the SDK-pinned Codex login flow and
    ``False`` when a valid login already existed. Provider secrets are removed
    from the child environment so an API-key login cannot silently satisfy the
    subscription-only profile.
    """

    codex_bin_override = os.environ.pop("PRAXIST_CODEX_BIN", None)
    try:
        try:
            binary = resolve_codex_binary()
        except OSError as exc:
            raise RuntimeError(
                "Unable to launch the SDK-pinned Codex login flow. Install "
                "praxist[codex], then rerun Codex-native setup."
            ) from exc
        try:
            verify_chatgpt_login(binary)
            return False
        except RuntimeError as status_error:
            if not allow_interactive:
                raise RuntimeError(
                    "Codex-native mode needs a saved ChatGPT login. Rerun "
                    "`praxist setup --profile codex-native` in a local interactive terminal."
                ) from status_error

        try:
            result = subprocess.run(
                [binary, "login"],
                check=False,
                env=_subscription_probe_env(),
            )
        except OSError as exc:
            raise RuntimeError(
                "Unable to launch the SDK-pinned Codex login flow. Install "
                "praxist[codex], then rerun Codex-native setup."
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                "The SDK-pinned Codex login flow did not complete; "
                "Codex-native setup was cancelled."
            )
        verify_chatgpt_login(binary)
        return True
    finally:
        if codex_bin_override is not None:
            os.environ["PRAXIST_CODEX_BIN"] = codex_bin_override


def _read_chatgpt_file_auth(codex_home: Path) -> tuple[bytes, str | None] | None:
    """Read one regular auth-file snapshot without following its final target."""

    try:
        target = (codex_home / "auth.json").resolve(strict=True)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return None
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                raw = stream.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or str(payload.get("auth_mode", "")).casefold() != "chatgpt":
        return None
    tokens = payload.get("tokens")
    account_id = tokens.get("account_id") if isinstance(tokens, dict) else None
    return raw, str(account_id) if account_id else None


def _credential_key_id(codex_home: Path, source: str, account_id: str | None) -> str:
    identity = f"{codex_home}\0{source}\0{account_id or ''}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{CHATGPT_CREDENTIAL_KEY_PREFIX}:{digest}"


def _subscription_probe_env() -> dict[str, str]:
    """Return only non-secret host context needed by the local status probe."""

    return {
        key: value
        for key, value in os.environ.items()
        if key in _PROBE_ENV_KEYS and value not in (None, "")
    }


def _has_chatgpt_login_status(result: subprocess.CompletedProcess[str]) -> bool:
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    return any(
        line.strip().casefold() in _CHATGPT_LOGIN_STATUS_LINES for line in output.splitlines()
    )

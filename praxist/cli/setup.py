"""``praxist setup`` — conservative host setup orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from praxist.cli._env import (
    AGENT_SYSTEM_TO_RUNTIME_REF,
    AGENT_SYSTEM_VALUES,
    CODEX_NATIVE_DEFAULT_MODEL,
    agent_system_for_runtime_ref,
)
from praxist.cli._setup_common import (
    SETUP_PROFILE_ENV_VAR,
    provider_key_var,
    provider_plugin_ref,
    provider_short_name,
    read_env_file,
    selected_config_file,
    write_env_file,
)
from praxist.cli._terminal_ui import (
    Choice,
    TerminalInteractionCancelled,
    TerminalInteractionError,
    interactive_terminal_available,
    select_choice,
)
from praxist.cli.configure_llm import (
    ConfigureLLMCancelled,
    ConfigureLLMError,
    configure_llm,
)
from praxist.cli.doctor import build_report, print_report
from praxist.cli.examples import (
    BUNDLED_EXAMPLES,
    ExampleInstallError,
    materialize_example,
)
from praxist.cli.install_skills import (
    InstallSkillsError,
    SkillConflictError,
    install_codex_skills,
)
from praxist.cli.install_skills import (
    install_skills as install_skills_for_host,
)
from praxist.cli.product_usage import (
    ProductUsageUnavailableError,
    prompt_for_consent_if_unset,
    read_product_usage_status,
)
from praxist.cli.user_agreement import (
    FAIR_SOURCE_LICENSE_URL,
    USER_AGREEMENT_URL,
    prompt_for_acceptance_if_needed,
)
from praxist.user_agreement import (
    FAIR_SOURCE_LICENSE_VERSION,
    USER_AGREEMENT_VERSION,
    current_acceptance,
)


@dataclass(frozen=True)
class SetupProfile:
    """One coherent operator-selectable runtime and provider combination."""

    profile_id: str
    label: str
    detail: str
    provider: str
    agent_system: str
    model: str
    requires_api_key: bool
    authentication: str
    authorization_detail: str


SETUP_PROFILES: tuple[SetupProfile, ...] = (
    SetupProfile(
        "codex-native",
        "Codex-native mode",
        "saved Codex login; best for short or exploratory runs",
        "openai",
        "codex_sdk",
        CODEX_NATIVE_DEFAULT_MODEL,
        False,
        "saved_chatgpt_login",
        "uses the existing saved ChatGPT/Codex login; no provider API key or new code",
    ),
    SetupProfile(
        "deepseek-api",
        "DeepSeek API (recommended)",
        "cost-efficient long research through claude_sdk",
        "deepseek",
        "claude_sdk",
        "deepseek-v4-pro[1m]",
        True,
        "provider_api_key",
        "requires a DeepSeek API key entered only through the local masked prompt",
    ),
    SetupProfile(
        "openrouter-api",
        "OpenRouter API",
        "OpenRouter model routing through claude_sdk",
        "openrouter",
        "claude_sdk",
        "anthropic/claude-opus-4.7",
        True,
        "provider_api_key",
        "requires an OpenRouter API key entered only through the local masked prompt",
    ),
    SetupProfile(
        "orcarouter-api",
        "OrcaRouter API",
        "OrcaRouter model routing through claude_sdk",
        "orcarouter",
        "claude_sdk",
        "orcarouter/auto",
        True,
        "provider_api_key",
        "requires an OrcaRouter API key entered only through the local masked prompt",
    ),
    SetupProfile(
        "anthropic-api",
        "Anthropic API",
        "Anthropic Messages through claude_sdk",
        "anthropic",
        "claude_sdk",
        "claude-opus-4-7",
        True,
        "provider_api_key",
        "requires an Anthropic API key entered only through the local masked prompt",
    ),
)


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``praxist setup`` subcommand."""
    parser = subparsers.add_parser(
        "setup",
        help="Configure this host for Praxist operation.",
        description=(
            "Pip-first Praxist host setup. Run this after installing the package and runtime "
            "extras. It writes only Praxist user-level configuration and Praxist-managed "
            "agent skill registrations; it does not install global agent CLIs or "
            "task-specific dependencies."
        ),
    )
    parser.add_argument(
        "--agent-system",
        choices=AGENT_SYSTEM_VALUES,
        default=None,
        help="Agent runtime selection to persist.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Built-in provider name to configure.",
    )
    parser.add_argument("--model", default=None, help="Provider model name to persist.")
    secret = parser.add_mutually_exclusive_group()
    secret.add_argument(
        "--api-key-stdin",
        action="store_true",
        help="Read the provider API key from stdin; a local TTY shows one * per character.",
    )
    secret.add_argument(
        "--api-key-env",
        default=None,
        help="Read the provider API key from this environment variable.",
    )
    secret.add_argument(
        "--no-api-key",
        action="store_true",
        help="Configure a supported no-key authentication route.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Review the License and User Agreement, choose optional privacy, and select a coherent "
            "runtime profile in a local TTY wizard."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=tuple(profile.profile_id for profile in SETUP_PROFILES),
        default=None,
        help="Apply one complete profile; a missing API key is requested in the local terminal.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List supported setup profiles as JSON and exit without changes.",
    )
    parser.add_argument(
        "--agent-managed",
        "--codex-managed",
        dest="agent_managed",
        action="store_true",
        help=(
            "Report the read-only agent-managed first-use decision state and next required "
            "action as JSON. --codex-managed remains a compatibility alias."
        ),
    )
    parser.add_argument(
        "--install-skills",
        choices=("codex", "claude", "none"),
        default=None,
        help="Install bundled skills for an agent host (interactive default: codex).",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Config file to update (default: $PRAXIST_CONFIG_FILE or the user config).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit setup and readiness results as JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report changes without writing files.",
    )
    parser.add_argument(
        "--skip-doctor",
        action="store_true",
        help="Skip the final readiness report.",
    )
    parser.set_defaults(func=cmd_setup)


def cmd_setup(args: argparse.Namespace) -> int:
    """Run conservative setup operations and report readiness."""
    if args.agent_managed:
        if any(
            (
                args.list_profiles,
                args.interactive,
                args.profile,
                args.provider,
                args.model,
                args.agent_system,
                args.api_key_stdin,
                args.api_key_env,
                args.no_api_key,
                args.install_skills,
                args.dry_run,
                args.skip_doctor,
            )
        ):
            print(
                "praxist setup: --agent-managed is a read-only status command and cannot "
                "be combined with setup operations",
                file=sys.stderr,
            )
            return 2
        print(json.dumps(_agent_managed_status(selected_config_file(args.config_file)), indent=2))
        return 0

    if args.list_profiles:
        print(json.dumps([asdict(profile) for profile in SETUP_PROFILES], indent=2))
        return 0

    if args.interactive and not interactive_terminal_available():
        print("praxist setup: --interactive requires a local interactive terminal", file=sys.stderr)
        return 2
    if args.interactive and args.profile:
        print(
            "praxist setup: --interactive and --profile are alternative setup modes",
            file=sys.stderr,
        )
        return 2
    if (args.interactive or args.profile) and any(
        (
            args.provider,
            args.model,
            args.agent_system,
            args.api_key_stdin,
            args.api_key_env,
            args.no_api_key,
        )
    ):
        print(
            "praxist setup: --interactive/--profile cannot be combined with explicit "
            "runtime, provider, model, or credential options",
            file=sys.stderr,
        )
        return 2

    config_file = selected_config_file(args.config_file)
    selected_profile: SetupProfile | None = None
    if args.interactive:
        consent_output = sys.stderr if args.json_output else sys.stdout
        if not args.dry_run:
            try:
                accepted = prompt_for_acceptance_if_needed(output_stream=consent_output)
            except (TerminalInteractionCancelled, TerminalInteractionError) as exc:
                print(f"praxist setup: {exc}", file=sys.stderr)
                return 130
            if not accepted:
                print(
                    "praxist setup: License and User Agreement were not accepted", file=sys.stderr
                )
                return 130
            if not prompt_for_consent_if_unset(output_stream=consent_output):
                print("praxist setup: interactive setup cancelled", file=sys.stderr)
                return 130
        try:
            selected_profile = _select_setup_profile(config_file)
        except (TerminalInteractionCancelled, TerminalInteractionError) as exc:
            print(f"praxist setup: {exc}", file=sys.stderr)
            return 130
    elif args.profile:
        selected_profile = _profile_by_id(args.profile)

    operations: list[dict[str, object]] = []
    if selected_profile and selected_profile.profile_id == "codex-native":
        if args.dry_run:
            operations.append(
                {
                    "operation": "ensure_codex_chatgpt_login",
                    "result": {"status": "not_checked", "dry_run": True},
                }
            )
        else:
            from praxist.plugins.agent_runtimes.codex_sdk._auth import ensure_chatgpt_login

            try:
                login_started = ensure_chatgpt_login(
                    allow_interactive=interactive_terminal_available()
                )
            except RuntimeError as exc:
                print(f"praxist setup: {exc}", file=sys.stderr)
                return 1
            operations.append(
                {
                    "operation": "ensure_codex_chatgpt_login",
                    "result": {
                        "status": "authenticated",
                        "login_started": login_started,
                    },
                }
            )

    if selected_profile is not None:
        needs_key = selected_profile.requires_api_key and not _provider_key_available(
            selected_profile.provider,
            config_file,
        )
        if (
            needs_key
            and not args.interactive
            and not args.dry_run
            and not interactive_terminal_available()
        ):
            print(
                "praxist setup: this API profile needs a local terminal for masked key input; "
                "run `praxist setup --interactive` or use explicit provider automation flags",
                file=sys.stderr,
            )
            return 2
        args.provider = selected_profile.provider
        args.model = selected_profile.model
        args.agent_system = selected_profile.agent_system
        args.api_key_stdin = needs_key
        args.no_api_key = not needs_key

    if args.provider:
        try:
            operations.append(
                {
                    "operation": "configure_llm",
                    "result": configure_llm(
                        provider=args.provider,
                        model=args.model,
                        agent_system=args.agent_system,
                        api_key_stdin=args.api_key_stdin,
                        api_key_env=args.api_key_env,
                        no_api_key=args.no_api_key or not (args.api_key_stdin or args.api_key_env),
                        config_file=config_file,
                        project_config_file=None,
                        dry_run=args.dry_run,
                        setup_profile=(
                            selected_profile.profile_id if selected_profile is not None else None
                        ),
                    ),
                }
            )
        except ConfigureLLMCancelled as exc:
            print(f"praxist setup: {exc}", file=sys.stderr)
            return 130
        except ConfigureLLMError as exc:
            print(f"praxist setup: {exc}", file=sys.stderr)
            return 1
    elif args.model or args.api_key_stdin or args.api_key_env or args.no_api_key:
        print("praxist setup: --model/--api-key-* require --provider", file=sys.stderr)
        return 2
    elif args.agent_system:
        if not args.dry_run:
            try:
                write_env_file(
                    config_file,
                    {
                        "PRAXIST_AGENT_SYSTEM": args.agent_system,
                        "PRAXIST_AGENT_RUNTIME_REF": AGENT_SYSTEM_TO_RUNTIME_REF[args.agent_system],
                    },
                    remove_keys={SETUP_PROFILE_ENV_VAR},
                )
            except OSError as exc:
                print(f"praxist setup: could not write configuration: {exc}", file=sys.stderr)
                return 1
        operations.append(
            {
                "operation": "configure_agent_system",
                "result": {
                    "agent_system": args.agent_system,
                    "config_file": str(config_file),
                    "dry_run": str(args.dry_run).lower(),
                },
            }
        )

    install_skills = args.install_skills or ("codex" if args.interactive else "none")
    if install_skills != "none":
        try:
            skill_result = _install_setup_skills(
                target=install_skills,
                target_dir=None,
                mode="copy",
                replace=True,
                dry_run=args.dry_run,
            )
        except SkillConflictError as exc:
            if not args.interactive:
                print(f"praxist setup: {exc}", file=sys.stderr)
                return 1
            print("Praxist found operator-owned skills with bundled names:", file=sys.stderr)
            for path in exc.paths:
                print(f"  {path}", file=sys.stderr)
            try:
                conflict_action = select_choice(
                    "Choose how to handle these skill conflicts",
                    (
                        Choice(
                            "keep",
                            "Keep existing skills",
                            "install non-conflicting Praxist skills only",
                        ),
                        Choice(
                            "backup-replace",
                            "Back up and replace",
                            "preserve each existing path beside the installed skill",
                        ),
                        Choice(
                            "cancel",
                            "Cancel skill setup",
                            "leave every skill unchanged; keep completed profile configuration",
                        ),
                    ),
                    default=0,
                    input_stream=sys.stdin,
                    output_stream=sys.stderr,
                )
            except (TerminalInteractionCancelled, TerminalInteractionError) as choice_exc:
                print(f"praxist setup: {choice_exc}", file=sys.stderr)
                return 130
            if conflict_action == "cancel":
                print("praxist setup: skill installation cancelled", file=sys.stderr)
                return 130
            if conflict_action == "keep":
                try:
                    skill_result = _install_setup_skills(
                        target=install_skills,
                        target_dir=None,
                        mode="copy",
                        replace=True,
                        dry_run=args.dry_run,
                        skip_unmanaged=True,
                    )
                except InstallSkillsError as retry_exc:
                    print(f"praxist setup: {retry_exc}", file=sys.stderr)
                    return 1
            else:
                try:
                    skill_result = _install_setup_skills(
                        target=install_skills,
                        target_dir=None,
                        mode="copy",
                        replace=True,
                        dry_run=args.dry_run,
                        force_unmanaged=True,
                    )
                except InstallSkillsError as retry_exc:
                    print(f"praxist setup: {retry_exc}", file=sys.stderr)
                    return 1
        except InstallSkillsError as exc:
            print(f"praxist setup: {exc}", file=sys.stderr)
            return 1
        operations.append(
            {
                "operation": f"install_{install_skills}_skills",
                "result": skill_result,
            }
        )
    for example_name in BUNDLED_EXAMPLES:
        try:
            example_result = materialize_example(
                example_name,
                dry_run=args.dry_run,
            )
        except ExampleInstallError as exc:
            print(f"praxist setup: {exc}", file=sys.stderr)
            return 1
        operations.append(
            {
                "operation": f"install_{example_name}_example",
                "result": asdict(example_result),
            }
        )
    report = (
        {"ok": True, "skipped": True, "checks": [], "next_actions": []}
        if args.skip_doctor
        else build_report(
            task_path=None,
            target=install_skills if install_skills != "none" else "auto",
            agent_system=args.agent_system,
            model_provider_ref=(provider_plugin_ref(args.provider) if args.provider else None),
            model=args.model,
            codex_native=bool(selected_profile and selected_profile.profile_id == "codex-native"),
            config_file=config_file,
        )
    )
    payload = {"operations": operations, "doctor": report}
    if args.json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        example_operations = {f"install_{name}_example" for name in BUNDLED_EXAMPLES}
        for operation in operations:
            print(f"{operation['operation']}: ok", file=sys.stderr)
            if operation["operation"] in example_operations:
                result = operation["result"]
                if isinstance(result, dict):
                    print(f"example project: {result['destination']}", file=sys.stderr)
        if report["ok"]:
            suffix = (
                "readiness check skipped"
                if report.get("skipped")
                else "run `praxist doctor` to check again"
            )
            print(f"setup complete; {suffix}", file=sys.stderr)
        else:
            print("setup incomplete: readiness checks failed", file=sys.stderr)
            print_report(report)
    return 0 if report["ok"] else 1


def _install_setup_skills(
    *,
    target: str,
    target_dir: Path | None,
    mode: str,
    replace: bool,
    dry_run: bool,
    force_unmanaged: bool = False,
    skip_unmanaged: bool = False,
) -> dict[str, object]:
    """Preserve the established Codex seam while adding a second host."""

    if target == "codex":
        return install_codex_skills(
            target_dir=target_dir,
            mode=mode,
            replace=replace,
            dry_run=dry_run,
            force_unmanaged=force_unmanaged,
            skip_unmanaged=skip_unmanaged,
        )
    return install_skills_for_host(
        target=target,
        target_dir=target_dir,
        mode=mode,
        replace=replace,
        dry_run=dry_run,
        force_unmanaged=force_unmanaged,
        skip_unmanaged=skip_unmanaged,
    )


def _profile_by_id(profile_id: str) -> SetupProfile:
    for profile in SETUP_PROFILES:
        if profile.profile_id == profile_id:
            return profile
    raise ValueError(f"unknown setup profile: {profile_id}")


def _select_setup_profile(config_file: Path) -> SetupProfile:
    profile_id = select_choice(
        "Choose a Praxist runtime profile",
        tuple(
            Choice(profile.profile_id, profile.label, profile.detail) for profile in SETUP_PROFILES
        ),
        default=_configured_profile_default(config_file),
        input_stream=sys.stdin,
        output_stream=sys.stderr,
    )
    return _profile_by_id(profile_id)


def _configured_profile_default(config_file: Path) -> int | None:
    _, configured = read_env_file(config_file)
    if not configured:
        return 0
    matched = _profile_matching_config(configured, require_model=False)
    if matched is None:
        return None
    return SETUP_PROFILES.index(matched)


def _profile_matching_config(
    configured: dict[str, str], *, require_model: bool
) -> SetupProfile | None:
    runtime_ref = configured.get("PRAXIST_AGENT_RUNTIME_REF", "") or configured.get(
        "RUNTIME_REF", ""
    )
    agent_system = agent_system_for_runtime_ref(runtime_ref) or configured.get(
        "PRAXIST_AGENT_SYSTEM", ""
    )
    provider_ref = configured.get("PRAXIST_MODEL_PROVIDER_REF", "") or configured.get(
        "MODEL_PROVIDER_REF", ""
    )
    provider = (
        provider_short_name(provider_ref)
        if provider_ref
        else configured.get("PRAXIST_LLM_PROVIDER", "")
    )
    model = configured.get("PRAXIST_MODEL", "")
    for profile in SETUP_PROFILES:
        if (
            profile.agent_system == agent_system
            and profile.provider == provider
            and ((not require_model and not model) or profile.model == model)
        ):
            return profile
    return None


def _agent_managed_status(config_file: Path) -> dict[str, object]:
    """Derive the Agent OOBE decision state from canonical user records."""

    _, configured = read_env_file(config_file)
    configured_profile = _profile_matching_config(configured, require_model=False)
    recorded_profile_id = configured.get(SETUP_PROFILE_ENV_VAR, "").strip()
    recorded_profile = next(
        (profile for profile in SETUP_PROFILES if profile.profile_id == recorded_profile_id),
        None,
    )
    confirmed_profile = (
        recorded_profile
        if recorded_profile is not None
        and _profile_matching_config(configured, require_model=True) == recorded_profile
        else None
    )
    if confirmed_profile is not None:
        profile_state = "confirmed"
    elif recorded_profile_id:
        profile_state = "changed_since_selection"
    elif configured_profile is not None:
        profile_state = "configured_but_not_selected"
    else:
        profile_state = "not_selected"

    agreement = current_acceptance()
    try:
        usage = read_product_usage_status()
    except ProductUsageUnavailableError:
        usage = {"collection_available": False, "status": "unavailable"}
    collection_available = bool(usage["collection_available"])
    privacy_decision_required = collection_available and usage["status"] == "unset"

    if agreement is None:
        next_action = "review_user_agreement"
    elif privacy_decision_required:
        next_action = "choose_product_usage"
    elif confirmed_profile is None:
        next_action = "choose_profile"
    else:
        next_action = "run_doctor_then_finish_setup"

    authorization = None
    if confirmed_profile is not None:
        authorization = {
            "mode": confirmed_profile.authentication,
            "api_key_required": confirmed_profile.requires_api_key,
            "detail": confirmed_profile.authorization_detail,
        }

    return {
        "installed": True,
        "setup_decisions_complete": (
            agreement is not None
            and not privacy_decision_required
            and confirmed_profile is not None
        ),
        "next_required_action": next_action,
        "user_agreement": {
            "accepted": agreement is not None,
            "version": USER_AGREEMENT_VERSION,
            "review_url": USER_AGREEMENT_URL,
            "license_version": FAIR_SOURCE_LICENSE_VERSION,
            "license_url": FAIR_SOURCE_LICENSE_URL,
        },
        "product_usage": {
            **usage,
            "decision_required": privacy_decision_required,
            "detail": (
                "collection is unavailable; no product-usage authorization is required"
                if not collection_available
                else "sharing remains optional and requires an explicit choice when unset"
            ),
        },
        "profile": {
            "selected": confirmed_profile is not None,
            "state": profile_state,
            "profile_id": confirmed_profile.profile_id if confirmed_profile else None,
            "configured_profile_id": (
                configured_profile.profile_id if configured_profile is not None else None
            ),
            "authorization": authorization,
        },
        "profiles": [asdict(profile) for profile in SETUP_PROFILES],
        "config_file": str(config_file),
    }


# Compatibility for callers that imported the original private helper.
_codex_managed_status = _agent_managed_status


def _provider_key_available(provider: str, config_file: Path) -> bool:
    key_var = provider_key_var(provider)
    _, configured = read_env_file(config_file)
    return bool(os.environ.get(key_var) or configured.get(key_var))

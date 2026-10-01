"""Frozen, multi-metric evaluator with one first-contact success standard.

The evaluator owns the validation split, frozen plant, first-contact metrics,
contract checks, evidence-stage labels, and canonical Praxist summary.  A
candidate owns only ``controller.py``, ``controller_config.json``, and
``variant.json`` inside its variant directory.

Landing rollouts stop at interpolated first landing-leg contact.  The plant's
post-contact spring/damper response is therefore never allowed to convert an
unsafe impact into evaluator success.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from evaluations.controller_ood.operational_metrics import (
    INITIAL_FUEL_KG,
    INITIAL_MASS_KG,
    MASS_EMPTY_KG,
    NO_CONTACT_SINK_PENALTY_MPS,
    SUCCESS_THRESHOLDS,
    first_contact_arrays,
    summarize,
    wilson_interval,
)
from evaluations.controller_ood.operational_rollout import (
    AUDIT_COLUMNS,
    build_operational_rollout,
)


TASK_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TASK_ROOT.parent
PROJECT_SRC = PROJECT_ROOT / "src"

# These values attest the immutable scientific boundary independently of the
# peer-authored candidate and of task metadata files.
FROZEN_HASHES = {
    "src/plant_adapter.py": "b6a68c67bfb87ebc708747fe708fc8ae80a00242c2225e1fe3ada1600bf8cd62",
    "vendor/frozen_c05_plant/config.yaml": "80201171ae6002fb016991ca630b440d61f67291e3e714fd8df398632de3bc59",
    "vendor/frozen_c05_plant/contact_law.py": "0d746ba03e3423317af77a5122fdb5f39d75e2c6f04c1325dd5f45437e14f9ee",
    "vendor/frozen_c05_plant/fuel_schedule.py": "8032873f77831484f4d750716309e40c7a3801ab24d24dc62abca9c84d829595",
    "vendor/frozen_c05_plant/gear_frac_candidate.yaml": "b270d3af8279f3a819fdd39cf419a4cda935593e626611e9acf0fe4019831f86",
    "vendor/frozen_c05_plant/grid_fin_aero.py": "d2bc0370e904fabf71d32a0c21487497705bb11965a7a564f3a9ab9b668c6078",
    "vendor/frozen_c05_plant/guidance_divert.py": "8a3821200c8395e886cc31b5a8b3aa80e0c672cde074883c9e1973267d660f31",
    "vendor/frozen_c05_plant/harness/integrator_flags.py": "209720ec3b44005fc765b7bcd087b3e63fbd86360b67730a193e967b3e770529",
    "vendor/frozen_c05_plant/plant_aero.py": "a8c92998d8ca5aa69da0bd6af45289b35c31e019fb7d4a982108316df8a1f4f6",
    "vendor/frozen_c05_plant/ppo_rocket_6dof_finned_jax.py": "e9ff8976c66773e9d10612a84b6636cfab69c13c6fb10f83c21661084d17d5a5",
    "vendor/frozen_c05_plant/provenance_labels.py": "427722dd8ca0e008964ba32f680ebba66c8786bb05c382da104562102b9235e5",
    "vendor/frozen_c05_plant/residual_fuel_mask.py": "31f73eb40f65b7aa260b889568ec7a423fe87614a7a0af6cc965b11138eab714",
    "vendor/frozen_c05_plant/residual_mask.py": "98aacdc530c98a2086c85942b9058ce4f6ef9bbf892ae9a458ef0a5d1350d936",
    "vendor/frozen_c05_plant/reward_shaping.py": "43f5675fbc197adb7f5a75d2a74fa239ccb78971173c3c3b303c29d04c9a1490",
    "vendor/frozen_c05_plant/terminal_reward.py": "668d318bf87bd3f5d53924cd1820590158511dccb2541f9fcd77a50679918fcd",
}
SOURCE_BANKS = {
    "nominal_unseen": (
        PROJECT_ROOT / "data/source_banks/nominal_unseen_40960.npz",
        "674d119f8f0cd36c2553f0cc9134ec23886fd00d1a34396d515d957132311a3d",
    ),
    "near_ood": (
        PROJECT_ROOT / "data/source_banks/near_ood_easy_velocity_40960.npz",
        "d35803608dd3dedbbd4db76d9b39aebe37c2eedbdba4ad9ec0dcd94a690d7730",
    ),
    "hard_ood": (
        PROJECT_ROOT / "data/source_banks/hard_ood_fast_outer_annulus_40960.npz",
        "55d3e0054ef5d225b78f6558a4e15a3a908bedb84daabb7cc66b8e6579fd297c",
    ),
}
DEVELOPMENT_PATH = PROJECT_ROOT / "data/development_ood_2048.npz"
DEVELOPMENT_SHA256 = "00539ee4e538fab8a65e82ae737289e5b42aa9dfae70da9da1cec5e7c8871f94"
COMPLETE_LANDING_UNITS = 12_288
ROLL_UNITS = 1_024
COMPLETE_UNITS = COMPLETE_LANDING_UNITS + ROLL_UNITS
VALIDATION_SEED = 20260821
NOMINAL_SEED = 20260822
VALIDATION_SLICE = slice(9_216, 13_312)
ALLOWED_CHANGED_MODULES = {
    "energy_manager",
    "trajectory_guidance",
    "attitude_controller_yz",
    "allocator_yz",
    "fin_effectiveness_model",
    "state_disturbance_estimator",
    "roll_rcs_controller",
    "constraint_governor",
    "terminal_landing_manager",
    "robust_design_validation",
}
REQUIRED_DIMENSIONS = {
    "mechanism_family",
    "intervention_surface",
    "intent",
    "semantic_family",
    "parent_lineage",
    "novelty_axis",
}
FORBIDDEN_IMPORT_ROOTS = {
    "torch",
    "tensorflow",
    "keras",
    "flax",
    "haiku",
    "optax",
    "sklearn",
    "stable_baselines3",
    "ray",
    "requests",
    "urllib3",
    "socket",
    "subprocess",
}
FORBIDDEN_CALL_NAMES = {
    "eval",
    "exec",
    "compile",
    "system",
    "popen",
    "urlopen",
}
FORBIDDEN_PATH_TOKENS = {
    "formal_ood_16384",
    "terminal_results.npz",
    "full_three_bank_evaluation",
    "initialization_resource_observation",
}
VARIANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
CANONICAL_PEER_ID_RE = re.compile(r"^gen(?P<generation>\d+)_peer(?P<index>\d+)$")
FROZEN_AUDIT_LIMITS = {
    "dt": 0.1,
    "gimbal_normal_rad": 0.075,
    "grid_normal_rad": 0.25,
    "throttle_normal_min": 0.02,
    "throttle_normal_max": 0.98,
    "rcs_roll_normal_cap_nm": 30_000.0,
    "rcs_roll_emergency_cap_nm": 60_000.0,
}
PROTOCOL_NAME = "rocket_booster_recovery_first_contact_7000kg_private_validation_v2"
PROTOCOL_VERSION = 2


class EvaluationError(RuntimeError):
    """A candidate or frozen-protocol contract failed."""


def _producer_identity_from_env() -> dict[str, Any]:
    """Return a canonical scheduler producer identity when one is available.

    Scheduler ownership of the explicit ``--output-dir`` remains authoritative.
    These summary fields provide a stable, independently parseable fallback for
    materialization and make attribution regressions visible in the artifact.
    Standalone/manual evaluations intentionally omit producer identity.
    """

    peer_values = {
        value
        for key in ("PRAXIST_PEER_ID", "PEER_ID")
        if (value := os.environ.get(key, "").strip())
    }
    if len(peer_values) != 1:
        return {}
    peer_id = next(iter(peer_values))
    match = CANONICAL_PEER_ID_RE.fullmatch(peer_id)
    if match is None:
        return {}
    generation_id = int(match.group("generation"))
    generation_values: set[int] = set()
    for key in ("PRAXIST_LOGICAL_GENERATION_ID", "GENERATION_ID"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        try:
            generation_values.add(int(raw))
        except ValueError:
            return {}
    if generation_values and generation_values != {generation_id}:
        return {}
    return {
        "peer_id": peer_id,
        "generation_id": generation_id,
        "source_generation_id": generation_id,
    }


class RollDiagnostic(NamedTuple):
    terminal_state: jax.Array
    terminal_previous_omega: jax.Array
    done: jax.Array
    active_steps: jax.Array
    last_unsettled_step: jax.Array
    peak_abs_roll_rate: jax.Array
    peak_pitch_yaw_rate: jax.Array
    rcs_switches: jax.Array
    rcs_total_variation_nm: jax.Array
    max_forbidden_action_abs: jax.Array
    nonfinite_seen: jax.Array


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationError("summary contains a non-finite value")
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(payload)
    path.write_text(
        json.dumps(safe, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _attest_frozen_assets() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in FROZEN_HASHES.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise EvaluationError(f"frozen asset missing: {relative}")
        actual = _sha256(path)
        observed[relative] = actual
        if actual != expected:
            raise EvaluationError(
                f"frozen asset hash mismatch: {relative}; expected {expected}, got {actual}"
            )
    for name, (path, expected) in SOURCE_BANKS.items():
        if not path.is_file():
            raise EvaluationError(f"source bank missing: {name}")
        actual = _sha256(path)
        observed[str(path.relative_to(PROJECT_ROOT))] = actual
        if actual != expected:
            raise EvaluationError(
                f"source bank hash mismatch: {name}; expected {expected}, got {actual}"
            )
    return observed


def _scan_candidate_tree(variant_dir: Path) -> dict[str, Any]:
    python_files = sorted(variant_dir.rglob("*.py"))
    if not python_files:
        raise EvaluationError("variant contains no Python source")
    if len(python_files) > 32:
        raise EvaluationError("variant contains more than 32 Python files")
    total_bytes = sum(path.stat().st_size for path in python_files)
    if total_bytes > 2_000_000:
        raise EvaluationError("variant Python source exceeds 2 MB")
    imports: set[str] = set()
    violations: list[str] = []
    for path in python_files:
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in FORBIDDEN_PATH_TOKENS):
            violations.append(f"{path.name}: reference to evaluator-only/result asset")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise EvaluationError(f"candidate syntax error in {path.name}: {exc}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id.lower()
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr.lower()
                if name in FORBIDDEN_CALL_NAMES:
                    violations.append(f"{path.name}: forbidden call {name}()")
    bad_imports = sorted(imports & FORBIDDEN_IMPORT_ROOTS)
    if bad_imports:
        violations.append("forbidden imports: " + ", ".join(bad_imports))
    if violations:
        raise EvaluationError("candidate static contract failed: " + "; ".join(violations))
    return {
        "python_files": [str(path.relative_to(variant_dir)) for path in python_files],
        "python_source_bytes": total_bytes,
        "import_roots": sorted(imports),
        "neural_or_rl_imports": False,
    }


def _load_variant(variant_dir: Path) -> tuple[ModuleType, dict[str, Any], dict[str, Any], dict[str, Any]]:
    variant_dir = variant_dir.resolve()
    controller_path = variant_dir / "controller.py"
    config_path = variant_dir / "controller_config.json"
    manifest_path = variant_dir / "variant.json"
    for path in (controller_path, config_path, manifest_path):
        if not path.is_file():
            raise EvaluationError(f"variant is missing {path.name}")
    scan = _scan_candidate_tree(variant_dir)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid variant.json: {exc}") from exc
    variant_id = str(manifest.get("variant_id") or "").strip()
    if not VARIANT_ID_RE.fullmatch(variant_id):
        raise EvaluationError("variant_id is missing or unsafe")
    if manifest.get("method_class") != "deterministic_classical_control":
        raise EvaluationError("method_class must be deterministic_classical_control")
    changed = manifest.get("changed_modules")
    if not isinstance(changed, list) or not all(isinstance(x, str) for x in changed):
        raise EvaluationError("changed_modules must be a list of module names")
    unknown = sorted(set(changed) - ALLOWED_CHANGED_MODULES)
    if unknown:
        raise EvaluationError("changed_modules outside allowed surface: " + ", ".join(unknown))
    dimensions = manifest.get("design_dimensions")
    if not isinstance(dimensions, dict) or not REQUIRED_DIMENSIONS.issubset(dimensions):
        missing = sorted(REQUIRED_DIMENSIONS - set(dimensions or {}))
        raise EvaluationError("variant design_dimensions missing: " + ", ".join(missing))
    if any(not str(dimensions[key]).strip() for key in REQUIRED_DIMENSIONS):
        raise EvaluationError("variant design_dimensions values must be non-empty")

    module_name = f"rocket_booster_recovery_candidate_{hashlib.sha256(str(controller_path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, controller_path)
    if spec is None or spec.loader is None:
        raise EvaluationError("cannot create candidate module spec")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(variant_dir))
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvaluationError(f"candidate import failed: {exc}") from exc
    finally:
        sys.path[:] = old_path
    for name in ("load_config", "init_memory", "control_step", "DIAGNOSTIC_COLUMNS"):
        if not hasattr(module, name):
            raise EvaluationError(f"candidate controller lacks required API: {name}")
    try:
        effective_config = module.load_config(config_path)
    except Exception as exc:
        raise EvaluationError(f"candidate configuration failed to resolve: {exc}") from exc
    if not isinstance(effective_config, dict):
        raise EvaluationError("load_config must return a resolved dictionary")
    # JSON serialization is both a finite-value check and the canonical schema
    # for effective-configuration provenance.
    effective_config = _json_safe(effective_config)
    return module, effective_config, manifest, {
        **scan,
        "controller_sha256": _sha256(controller_path),
        "config_sha256": _sha256(config_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def _load_project_modules():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.plant_adapter import load_frozen_plant, provenance

    return load_frozen_plant, provenance


def _set_protocol_initial_mass(state: np.ndarray) -> np.ndarray:
    """Return a copy with the immutable v2 total mass of 29,200 kg."""

    resolved = np.asarray(state, dtype=np.float32).copy()
    resolved[:, 13] = np.float32(INITIAL_MASS_KG)
    return resolved


def _private_validation_states() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    states: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    rows: list[np.ndarray] = []

    nominal_path = SOURCE_BANKS["nominal_unseen"][0]
    with np.load(nominal_path, allow_pickle=False) as bank:
        nominal = np.asarray(bank["state"], dtype=np.float32)
    nominal_order = np.random.default_rng(NOMINAL_SEED).permutation(len(nominal))
    nominal_idx = nominal_order[:4_096]
    states.append(nominal[nominal_idx])
    labels.append(np.zeros(4_096, dtype=np.int8))
    rows.append(nominal_idx.astype(np.int32))

    # Reproduce the original sequential RNG stream, then take only rows after
    # both the 1,024-row dev prefix and 8,192-row historical formal slice.
    rng = np.random.default_rng(VALIDATION_SEED)
    for source_id, name in enumerate(("near_ood", "hard_ood"), start=1):
        path = SOURCE_BANKS[name][0]
        with np.load(path, allow_pickle=False) as bank:
            bank_state = np.asarray(bank["state"], dtype=np.float32)
        order = rng.permutation(len(bank_state))
        selected = order[VALIDATION_SLICE]
        states.append(bank_state[selected])
        labels.append(np.full(4_096, source_id, dtype=np.int8))
        rows.append(selected.astype(np.int32))
    return (
        _set_protocol_initial_mass(np.concatenate(states)),
        np.concatenate(labels),
        np.concatenate(rows),
        ["nominal_unseen", "near_ood", "hard_ood"],
    )


def _development_states(limit: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if _sha256(DEVELOPMENT_PATH) != DEVELOPMENT_SHA256:
        raise EvaluationError("development bank hash mismatch")
    with np.load(DEVELOPMENT_PATH, allow_pickle=False) as bank:
        state = np.asarray(bank["state"], dtype=np.float32)
        source = np.asarray(bank["source_id"], dtype=np.int8)
        row = np.asarray(bank["source_row"], dtype=np.int32)
        names = np.asarray(bank["source_names"]).astype(str).tolist()
    if limit is not None:
        state, source, row = state[:limit], source[:limit], row[:limit]
    return _set_protocol_initial_mass(state), source, row, names


def _quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(lhs, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _roll_disturbance_states() -> np.ndarray:
    landing, _, _, _ = _private_validation_states()
    base = landing[:128].copy()
    patterns = np.asarray(
        [
            (-math.pi, 0.80),
            (-3.0 * math.pi / 4.0, 0.60),
            (-math.pi / 2.0, 0.40),
            (-math.pi / 4.0, 0.20),
            (math.pi / 4.0, -0.20),
            (math.pi / 2.0, -0.40),
            (3.0 * math.pi / 4.0, -0.60),
            (math.pi, -0.80),
        ],
        dtype=np.float64,
    )
    out: list[np.ndarray] = []
    for phi, roll_rate in patterns:
        block = base.copy()
        q_roll = np.zeros((len(block), 4), dtype=np.float64)
        q_roll[:, 0] = math.cos(phi / 2.0)
        q_roll[:, 1] = math.sin(phi / 2.0)
        q = _quaternion_multiply(block[:, 6:10].astype(np.float64), q_roll)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        block[:, 6:10] = q.astype(np.float32)
        block[:, 10] = np.float32(roll_rate)
        block[:, 11:13] = 0.0
        out.append(block)
    return np.concatenate(out)


def _run_landing(
    states: np.ndarray,
    *,
    batch_size: int,
    module: ModuleType,
    controller_cfg: dict[str, Any],
    plant: Any,
    plant_cfg: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[float],
]:
    # The candidate receives its own fully resolved configuration, while the
    # evaluator's saturation/cap thresholds remain fixed. This prevents a
    # candidate from improving an audit metric merely by relaxing the threshold
    # used to count hits.
    audit_cfg = dict(controller_cfg)
    audit_cfg.update(FROZEN_AUDIT_LIMITS)
    rollout = build_operational_rollout(
        batch_size=batch_size,
        module=module,
        controller_cfg=controller_cfg,
        audit_cfg=audit_cfg,
        plant=plant,
        plant_cfg=plant_cfg,
    )
    outputs: list[tuple[np.ndarray, ...]] = []
    timings: list[float] = []
    for start in range(0, len(states), batch_size):
        valid = min(batch_size, len(states) - start)
        block = states[start : start + valid]
        if valid < batch_size:
            block = np.concatenate(
                (block, np.repeat(block[-1:], batch_size - valid, axis=0)), axis=0
            )
        tic = time.perf_counter()
        result = jax.device_get(rollout(jnp.asarray(block)))
        jax.block_until_ready(result.terminal_state)
        timings.append(time.perf_counter() - tic)
        outputs.append(tuple(np.asarray(item)[:valid] for item in result))
        print(
            f"landing block {start // batch_size + 1}/"
            f"{(len(states) + batch_size - 1) // batch_size}: rows={valid} "
            f"elapsed={timings[-1]:.3f}s",
            flush=True,
        )
    terminal = np.concatenate([item[0] for item in outputs])
    previous = np.concatenate([item[1] for item in outputs])
    max_action = np.concatenate([item[3] for item in outputs])
    audit = np.concatenate([item[4] for item in outputs])
    contact_detected = np.concatenate([item[6] for item in outputs])
    contact_step = np.concatenate([item[7] for item in outputs])
    contact_leg_sink = np.concatenate([item[8] for item in outputs])
    return (
        terminal,
        previous,
        max_action,
        audit,
        contact_detected,
        contact_step,
        contact_leg_sink,
        timings,
    )


def _select_tree(proposed: Any, previous: Any, active: jax.Array) -> Any:
    return jax.tree.map(
        lambda x, y: jnp.where(
            active.reshape((active.shape[0],) + (1,) * (x.ndim - 1)), x, y
        ),
        proposed,
        previous,
    )


def _build_roll_rollout(
    *,
    batch_size: int,
    module: ModuleType,
    controller_cfg: dict[str, Any],
    plant: Any,
    plant_cfg: Any,
):
    init_batch = jax.vmap(lambda state: module.init_memory(state, controller_cfg))
    control_batch = jax.vmap(
        lambda state, memory: module.control_step(state, memory, controller_cfg)
    )
    key = jax.random.PRNGKey(0)
    plant_batch = jax.vmap(lambda state, action: plant.step_one(key, state, action, plant_cfg))
    dt = float(controller_cfg["dt"])

    @jax.jit
    def rollout(initial_state: jax.Array) -> RollDiagnostic:
        memory = init_batch(initial_state)
        n = initial_state.shape[0]
        done = jnp.zeros(n, dtype=bool)
        terminal_previous = initial_state[:, 10:13]
        active_steps = jnp.zeros(n, dtype=jnp.int32)
        last_unsettled = jnp.zeros(n, dtype=jnp.int32)
        peak_roll = jnp.abs(initial_state[:, 10])
        peak_pitch_yaw = jnp.linalg.norm(initial_state[:, 11:13], axis=1)
        previous_rcs = jnp.zeros(n, dtype=initial_state.dtype)
        switches = jnp.zeros(n, dtype=jnp.int32)
        total_variation = jnp.zeros(n, dtype=initial_state.dtype)
        max_forbidden = jnp.zeros(n, dtype=initial_state.dtype)
        nonfinite = jnp.zeros(n, dtype=bool)
        carry = (
            initial_state,
            memory,
            done,
            terminal_previous,
            active_steps,
            last_unsettled,
            peak_roll,
            peak_pitch_yaw,
            previous_rcs,
            switches,
            total_variation,
            max_forbidden,
            nonfinite,
        )

        def body(carry: tuple[Any, ...], _: Any):
            (
                state,
                memory,
                done,
                terminal_previous,
                active_steps,
                last_unsettled,
                peak_roll,
                peak_pitch_yaw,
                previous_rcs,
                switches,
                total_variation,
                max_forbidden,
                nonfinite,
            ) = carry
            active = ~done
            action, proposed_memory, diagnostic = control_batch(state, memory)
            _, proposed_state, _, proposed_done, _ = plant_batch(state, action)
            finite = jnp.all(jnp.isfinite(proposed_state), axis=1)
            newly_done = active & ((proposed_done > 0.5) | (~finite))
            terminal_previous = jnp.where(
                newly_done[:, None], state[:, 10:13], terminal_previous
            )
            next_state = jnp.where(active[:, None], proposed_state, state)
            next_done = done | newly_done
            next_memory = _select_tree(proposed_memory, memory, active)
            step_number = active_steps + active.astype(jnp.int32)
            active_steps = step_number
            outside = active & (jnp.abs(next_state[:, 10]) >= 0.02)
            last_unsettled = jnp.where(outside, step_number, last_unsettled)
            peak_roll = jnp.maximum(
                peak_roll, jnp.abs(next_state[:, 10]) * active.astype(state.dtype)
            )
            peak_pitch_yaw = jnp.maximum(
                peak_pitch_yaw,
                jnp.linalg.norm(next_state[:, 11:13], axis=1) * active.astype(state.dtype),
            )
            rcs = diagnostic[:, 16]
            sign_changed = (
                active
                & (jnp.abs(previous_rcs) > 1.0)
                & (jnp.abs(rcs) > 1.0)
                & (jnp.sign(previous_rcs) != jnp.sign(rcs))
            )
            switches = switches + sign_changed.astype(jnp.int32)
            total_variation = total_variation + jnp.abs(rcs - previous_rcs) * active
            previous_rcs = jnp.where(active, rcs, previous_rcs)
            forbidden = jnp.max(jnp.abs(action[:, jnp.asarray([4, 5, 8])]), axis=1)
            max_forbidden = jnp.maximum(max_forbidden, forbidden * active)
            nonfinite = nonfinite | (active & (~finite))
            return (
                next_state,
                next_memory,
                next_done,
                terminal_previous,
                active_steps,
                last_unsettled,
                peak_roll,
                peak_pitch_yaw,
                previous_rcs,
                switches,
                total_variation,
                max_forbidden,
                nonfinite,
            ), None

        final, _ = jax.lax.scan(body, carry, xs=None, length=int(plant_cfg.max_steps))
        return RollDiagnostic(
            terminal_state=final[0],
            terminal_previous_omega=final[3],
            done=final[2],
            active_steps=final[4],
            last_unsettled_step=final[5],
            peak_abs_roll_rate=final[6],
            peak_pitch_yaw_rate=final[7],
            rcs_switches=final[9],
            rcs_total_variation_nm=final[10],
            max_forbidden_action_abs=final[11],
            nonfinite_seen=final[12],
        )

    return rollout, dt


def _run_roll(
    states: np.ndarray,
    *,
    batch_size: int,
    module: ModuleType,
    controller_cfg: dict[str, Any],
    plant: Any,
    plant_cfg: Any,
) -> tuple[dict[str, float], dict[str, Any], list[float]]:
    rollout, dt = _build_roll_rollout(
        batch_size=batch_size,
        module=module,
        controller_cfg=controller_cfg,
        plant=plant,
        plant_cfg=plant_cfg,
    )
    blocks: list[RollDiagnostic] = []
    timings: list[float] = []
    for start in range(0, len(states), batch_size):
        valid = min(batch_size, len(states) - start)
        block = states[start : start + valid]
        if valid < batch_size:
            block = np.concatenate(
                (block, np.repeat(block[-1:], batch_size - valid, axis=0)), axis=0
            )
        tic = time.perf_counter()
        result = jax.device_get(rollout(jnp.asarray(block)))
        jax.block_until_ready(result.terminal_state)
        timings.append(time.perf_counter() - tic)
        blocks.append(RollDiagnostic(*(np.asarray(item)[:valid] for item in result)))
        print(
            f"roll block {start // batch_size + 1}/"
            f"{(len(states) + batch_size - 1) // batch_size}: rows={valid} "
            f"elapsed={timings[-1]:.3f}s",
            flush=True,
        )
    joined = RollDiagnostic(
        *(np.concatenate([getattr(block, field) for block in blocks]) for field in RollDiagnostic._fields)
    )
    final_roll = np.abs(joined.terminal_state[:, 10])
    previous_roll = np.abs(joined.terminal_previous_omega[:, 0])
    stable = (
        (final_roll < 0.02)
        & (previous_roll < 0.02)
        & (~joined.nonfinite_seen.astype(bool))
    )
    settling = joined.last_unsettled_step.astype(np.float64) * dt
    metrics = {
        "roll_stable_rate": float(np.mean(stable)),
        "roll_settling_time_p95_s": float(np.quantile(settling, 0.95)),
        "roll_peak_rate_p95_radps": float(np.quantile(joined.peak_abs_roll_rate, 0.95)),
        "roll_pitch_yaw_coupling_p95_radps": float(
            np.quantile(joined.peak_pitch_yaw_rate, 0.95)
        ),
        "roll_rcs_switches_mean": float(np.mean(joined.rcs_switches)),
        "roll_rcs_total_variation_mean_nm": float(np.mean(joined.rcs_total_variation_nm)),
        "roll_forbidden_action_max_abs": float(np.max(joined.max_forbidden_action_abs)),
        "roll_nonfinite_rate": float(np.mean(joined.nonfinite_seen.astype(np.float64))),
    }
    detail = {
        "trajectories": len(states),
        "stable_count": int(np.count_nonzero(stable)),
        "initial_roll_rates_radps": sorted(set(float(x) for x in states[:, 10])),
        "settling_definition": "time of last sample outside |omega_x|<0.02 rad/s",
    }
    return metrics, detail, timings


def _rate(arrays: dict[str, np.ndarray], name: str, mask: np.ndarray | None = None) -> float:
    values = np.asarray(arrays[name], dtype=bool)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    return float(np.mean(values)) if len(values) else 0.0


def _landing_metrics(
    *,
    initial: np.ndarray,
    source: np.ndarray,
    source_names: list[str],
    endpoint: np.ndarray,
    first_contact_detected: np.ndarray,
    first_contact_step: np.ndarray,
    first_contact_leg_sink_speed_mps: np.ndarray,
    audit: np.ndarray,
) -> tuple[dict[str, float | bool], dict[str, Any]]:
    arrays = first_contact_arrays(
        endpoint,
        first_contact_detected,
        first_contact_leg_sink_speed_mps,
    )
    idx = {name: i for i, name in enumerate(AUDIT_COLUMNS)}
    active_steps = np.maximum(audit[:, idx["active_steps"]], 1.0)
    active_total = float(np.sum(active_steps))
    detected = np.asarray(arrays["first_contact_detected"], dtype=bool)
    success = np.asarray(arrays["landing_success_pass"], dtype=bool)
    penalized_com_sink = np.where(
        detected,
        np.asarray(arrays["com_sink_speed_mps"]),
        NO_CONTACT_SINK_PENALTY_MPS,
    )
    penalized_leg_sink = np.where(
        detected,
        np.asarray(arrays["contact_leg_sink_speed_mps"]),
        NO_CONTACT_SINK_PENALTY_MPS,
    )
    success_count = int(np.count_nonzero(success))
    success_wilson_low, success_wilson_high = wilson_interval(
        success_count, len(success)
    )
    contact_tilt_deg = np.rad2deg(np.asarray(arrays["tilt_rad"]))
    metrics: dict[str, float | bool] = {
        "landing_success_rate": _rate(arrays, "landing_success_pass"),
        "landing_success_wilson_95_low": float(success_wilson_low),
        "landing_success_wilson_95_high": float(success_wilson_high),
        "first_contact_rate": _rate(arrays, "first_contact_detected"),
        "fuel_gate_pass_rate": _rate(arrays, "fuel_gate_pass"),
        "vertical_first_contact_gate_pass_rate": _rate(
            arrays, "vertical_gate_pass"
        ),
        "fuel_reserve_mean_fraction": float(np.mean(arrays["fuel_fraction"])),
        "fuel_reserve_p05_fraction": float(np.quantile(arrays["fuel_fraction"], 0.05)),
        "fuel_depletion_rate": float(np.mean(np.asarray(arrays["fuel_fraction"]) <= 0.0)),
        "fuel_gate_shortfall_rate": float(
            1.0 - _rate(arrays, "fuel_gate_pass")
        ),
        "fuel_reserve_margin_above_2pct_p05_fraction": float(
            np.quantile(
                np.maximum(np.asarray(arrays["fuel_fraction"]) - 0.02, 0.0),
                0.05,
            )
        ),
        "first_contact_sink_speed_mean_mps": float(np.mean(penalized_com_sink)),
        "first_contact_sink_speed_p50_mps": float(
            np.quantile(penalized_com_sink, 0.50)
        ),
        "first_contact_sink_speed_p95_mps": float(
            np.quantile(penalized_com_sink, 0.95)
        ),
        "first_contact_sink_speed_p99_mps": float(
            np.quantile(penalized_com_sink, 0.99)
        ),
        "first_contact_sink_speed_max_mps": float(np.max(penalized_com_sink)),
        "first_contact_leg_sink_speed_mean_mps": float(
            np.mean(penalized_leg_sink)
        ),
        "first_contact_leg_sink_speed_p95_mps": float(
            np.quantile(penalized_leg_sink, 0.95)
        ),
        "first_contact_total_speed_p95_mps": float(
            np.quantile(
                np.where(
                    detected,
                    np.asarray(arrays["total_speed_mps"]),
                    NO_CONTACT_SINK_PENALTY_MPS,
                ),
                0.95,
            )
        ),
        "first_contact_lateral_speed_p95_mps": float(
            np.quantile(arrays["lateral_speed_mps"], 0.95)
        ),
        "first_contact_lateral_error_p95_m": float(
            np.quantile(arrays["lateral_error_m"], 0.95)
        ),
        "first_contact_tilt_p95_deg": float(np.quantile(contact_tilt_deg, 0.95)),
        "first_contact_abs_roll_rate_p95_radps": float(
            np.quantile(np.abs(arrays["roll_rate_radps"]), 0.95)
        ),
        "first_contact_pitch_yaw_rate_p95_radps": float(
            np.quantile(arrays["pitch_yaw_rate_radps"], 0.95)
        ),
        "first_contact_time_p95_s": float(
            np.quantile(
                np.where(detected, np.asarray(first_contact_step) * 0.1, 90.0),
                0.95,
            )
        ),
        "unsafe_first_contact_rate": float(
            np.mean(detected & (~np.asarray(arrays["vertical_gate_pass"], dtype=bool)))
        ),
        # Structural invariants: no state after first contact is scored.
        "gear_damping_credit_rate": 0.0,
        "post_contact_scored_steps": 0.0,
        "grid_saturation_rate": float(
            np.sum(audit[:, idx["grid_saturation_steps"]]) / active_total
        ),
        "gimbal_saturation_rate": float(
            np.sum(audit[:, idx["gimbal_saturation_steps"]]) / active_total
        ),
        "throttle_saturation_rate": float(
            np.sum(audit[:, idx["throttle_saturation_steps"]]) / active_total
        ),
        "rcs_roll_cap_rate": float(np.sum(audit[:, idx["rcs_roll_cap_steps"]]) / active_total),
        "forbidden_action_max_abs": float(
            np.max(audit[:, idx["max_forbidden_action_abs"]])
        ),
        "plant_lateral_rcs_max_nm": float(
            np.max(audit[:, idx["max_plant_lateral_rcs_nm"]])
        ),
        "nonfinite_trajectory_rate": float(
            np.mean(audit[:, idx["nonfinite_seen"]] > 0.5)
        ),
        "gimbal_total_variation_mean_rad": float(
            np.mean(audit[:, idx["gimbal_total_variation_rad"]])
        ),
        "grid_total_variation_mean_rad": float(
            np.mean(audit[:, idx["grid_total_variation_rad"]])
        ),
        "throttle_total_variation_mean": float(
            np.mean(audit[:, idx["throttle_total_variation"]])
        ),
    }
    metrics["landing_success_min_fuel_reserve_fraction"] = (
        float(np.min(np.asarray(arrays["fuel_fraction"])[success]))
        if np.any(success)
        else 0.0
    )
    by_source: dict[str, Any] = {}
    for source_id, name in enumerate(source_names):
        mask = source == source_id
        source_summary = summarize(arrays, mask)
        by_source[name] = source_summary
        metrics[f"{name}_landing_success_rate"] = _rate(
            arrays, "landing_success_pass", mask
        )
        metrics[f"{name}_first_contact_rate"] = _rate(
            arrays, "first_contact_detected", mask
        )
        source_sink = penalized_com_sink[mask]
        metrics[f"{name}_first_contact_sink_speed_p95_mps"] = (
            float(np.quantile(source_sink, 0.95)) if len(source_sink) else 0.0
        )
    metrics["hard_ood_landing_success_rate"] = float(
        metrics.get("hard_ood_landing_success_rate", 0.0)
    )
    metrics["hard_ood_first_contact_sink_speed_p95_mps"] = float(
        metrics.get("hard_ood_first_contact_sink_speed_p95_mps", 0.0)
    )

    radius = np.linalg.norm(initial[:, 1:3], axis=1)
    radius_edges = np.asarray([0.0, 450.0, 900.0, 1_200.0, 1_450.0, 1_651.0])
    radius_detail: dict[str, Any] = {}
    landing_success_rates: list[float] = []
    for low, high in zip(radius_edges[:-1], radius_edges[1:], strict=True):
        mask = (radius >= low) & (radius < high)
        label = f"r{int(low)}_{int(high)}m"
        if not np.any(mask):
            continue
        landing_rate = _rate(arrays, "landing_success_pass", mask)
        landing_success_rates.append(landing_rate)
        radius_detail[label] = {
            "trajectories": int(np.count_nonzero(mask)),
            "landing_success_rate": landing_rate,
            "first_contact_rate": _rate(arrays, "first_contact_detected", mask),
        }
        metrics[f"{label}_landing_success_rate"] = landing_rate
    metrics["worst_radius_bin_landing_success_rate"] = (
        min(landing_success_rates) if landing_success_rates else 0.0
    )
    detail = {
        "overall": summarize(arrays),
        "by_source": by_source,
        "by_initial_radius": radius_detail,
        "success_definition": {
            "name": "landing_success",
            "only_success_standard": True,
            "endpoint": "interpolated first landing-leg contact before spring/damper response",
            "thresholds": SUCCESS_THRESHOLDS,
        },
        "anti_damping_protocol": (
            "The landing rollout terminates for scoring at first leg contact. "
            "Post-contact spring/damper states contribute zero scored steps and zero success credit."
        ),
        "contact_observability_limit": (
            "First leg contact is observable from frozen gear geometry. Post-contact dwell, "
            "bounce height, leg loads, slip, and overturn remain unscored and must not be claimed."
        ),
    }
    return metrics, detail


def _mode_contract(mode: str) -> dict[str, Any]:
    if mode == "complete":
        return {
            "landing_units": COMPLETE_LANDING_UNITS,
            "roll_units": ROLL_UNITS,
            "effort_ratio": 1.0,
            "coverage_ratio": 1.0,
            "landing": True,
            "roll": True,
        }
    if mode == "development":
        return {
            "landing_units": 2_048,
            "roll_units": 0,
            "effort_ratio": 2_048 / COMPLETE_UNITS,
            "coverage_ratio": 0.0,
            "landing": True,
            "roll": False,
        }
    if mode == "canary":
        return {
            "landing_units": 1,
            "roll_units": 0,
            "effort_ratio": 1 / COMPLETE_UNITS,
            "coverage_ratio": 0.0,
            "landing": True,
            "roll": False,
        }
    if mode == "roll_diagnostic":
        return {
            "landing_units": 0,
            "roll_units": ROLL_UNITS,
            "effort_ratio": ROLL_UNITS / COMPLETE_UNITS,
            "coverage_ratio": ROLL_UNITS / COMPLETE_UNITS,
            "landing": False,
            "roll": True,
        }
    raise EvaluationError(f"unsupported evaluator mode: {mode}")


def _default_metrics() -> dict[str, float | bool]:
    return {
        "landing_success_rate": 0.0,
        "landing_success_wilson_95_low": 0.0,
        "landing_success_wilson_95_high": 0.0,
        "first_contact_rate": 0.0,
        "fuel_gate_pass_rate": 0.0,
        "vertical_first_contact_gate_pass_rate": 0.0,
        "hard_ood_landing_success_rate": 0.0,
        "hard_ood_first_contact_sink_speed_p95_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "worst_radius_bin_landing_success_rate": 0.0,
        "fuel_reserve_mean_fraction": 0.0,
        "fuel_reserve_p05_fraction": 0.0,
        "fuel_depletion_rate": 0.0,
        "fuel_gate_shortfall_rate": 0.0,
        "fuel_reserve_margin_above_2pct_p05_fraction": 0.0,
        "landing_success_min_fuel_reserve_fraction": 0.0,
        "first_contact_sink_speed_mean_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_sink_speed_p50_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_sink_speed_p95_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_sink_speed_p99_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_sink_speed_max_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_leg_sink_speed_mean_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_leg_sink_speed_p95_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_total_speed_p95_mps": NO_CONTACT_SINK_PENALTY_MPS,
        "first_contact_lateral_speed_p95_mps": 0.0,
        "first_contact_lateral_error_p95_m": 0.0,
        "first_contact_tilt_p95_deg": 0.0,
        "first_contact_abs_roll_rate_p95_radps": 0.0,
        "first_contact_pitch_yaw_rate_p95_radps": 0.0,
        "first_contact_time_p95_s": 90.0,
        "unsafe_first_contact_rate": 0.0,
        "gear_damping_credit_rate": 0.0,
        "post_contact_scored_steps": 0.0,
        "grid_saturation_rate": 0.0,
        "gimbal_saturation_rate": 0.0,
        "throttle_saturation_rate": 0.0,
        "rcs_roll_cap_rate": 0.0,
        "forbidden_action_max_abs": 0.0,
        "plant_lateral_rcs_max_nm": 0.0,
        "nonfinite_trajectory_rate": 0.0,
        "gimbal_total_variation_mean_rad": 0.0,
        "grid_total_variation_mean_rad": 0.0,
        "throttle_total_variation_mean": 0.0,
        "roll_stable_rate": 0.0,
        "roll_settling_time_p95_s": 0.0,
        "roll_peak_rate_p95_radps": 0.0,
        "roll_pitch_yaw_coupling_p95_radps": 0.0,
        "roll_rcs_switches_mean": 0.0,
        "roll_rcs_total_variation_mean_nm": 0.0,
        "roll_forbidden_action_max_abs": 0.0,
        "roll_nonfinite_rate": 0.0,
    }


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    started = time.perf_counter()
    mode_spec = _mode_contract(args.mode)
    frozen_hashes = _attest_frozen_assets()
    module, controller_cfg, manifest, candidate_audit = _load_variant(args.variant_dir)
    if abs(float(controller_cfg.get("dt", -1.0)) - 0.1) > 1e-12:
        raise EvaluationError("controller dt must match the frozen 0.1 s integrator step")
    if abs(float(controller_cfg.get("mass_empty_kg", -1.0)) - MASS_EMPTY_KG) > 1e-9:
        raise EvaluationError("controller mass_empty_kg must remain 22200 kg")
    if abs(float(controller_cfg.get("initial_fuel_kg", -1.0)) - INITIAL_FUEL_KG) > 1e-9:
        raise EvaluationError("controller initial_fuel_kg must remain 7000 kg")
    load_frozen_plant, provenance = _load_project_modules()
    if tuple(module.DIAGNOSTIC_COLUMNS) != tuple(
        __import__("src.rocket_booster_recovery_controller", fromlist=["DIAGNOSTIC_COLUMNS"]).DIAGNOSTIC_COLUMNS
    ):
        raise EvaluationError("DIAGNOSTIC_COLUMNS layout differs from the frozen interface")
    plant, plant_cfg = load_frozen_plant("rk4", 1)
    # Task-owner protocol update: vendor dynamics/contact equations are frozen,
    # while the initial propellant capacity is fixed at 7,000 kg.
    plant_cfg = plant_cfg._replace(mass_full=INITIAL_MASS_KG, fuel_scale=1.0)
    metrics = _default_metrics()
    details: dict[str, Any] = {}
    block_timings: list[float] = []
    completed_units = 0

    if mode_spec["landing"]:
        if args.mode == "complete":
            initial, source, rows, source_names = _private_validation_states()
        else:
            initial, source, rows, source_names = _development_states(
                1 if args.mode == "canary" else None
            )
        (
            endpoint,
            previous,
            max_action,
            audit,
            contact_detected,
            contact_step,
            contact_leg_sink,
            timings,
        ) = _run_landing(
            initial,
            batch_size=args.batch_size,
            module=module,
            controller_cfg=controller_cfg,
            plant=plant,
            plant_cfg=plant_cfg,
        )
        landing_metrics, landing_detail = _landing_metrics(
            initial=initial,
            source=source,
            source_names=source_names,
            endpoint=endpoint,
            first_contact_detected=contact_detected,
            first_contact_step=contact_step,
            first_contact_leg_sink_speed_mps=contact_leg_sink,
            audit=audit,
        )
        metrics.update(landing_metrics)
        details["landing"] = landing_detail
        details["landing"]["source_rows_sha256"] = _canonical_digest(rows.tolist())
        details["landing"]["max_abs_normalized_action_by_channel"] = np.max(
            max_action, axis=0
        ).tolist()
        details["landing"]["endpoint_previous_omega_available"] = bool(
            np.all(np.isfinite(previous))
        )
        block_timings.extend(timings)
        completed_units += len(initial)

    if mode_spec["roll"]:
        roll_states = _roll_disturbance_states()
        roll_metrics, roll_detail, timings = _run_roll(
            roll_states,
            batch_size=args.batch_size,
            module=module,
            controller_cfg=controller_cfg,
            plant=plant,
            plant_cfg=plant_cfg,
        )
        metrics.update(roll_metrics)
        details["roll"] = roll_detail
        block_timings.extend(timings)
        completed_units += len(roll_states)

    contract_lock_passed = bool(
        metrics["forbidden_action_max_abs"] == 0.0
        and metrics["plant_lateral_rcs_max_nm"] == 0.0
        and metrics["roll_forbidden_action_max_abs"] == 0.0
        and metrics["nonfinite_trajectory_rate"] == 0.0
        and metrics["roll_nonfinite_rate"] == 0.0
        and metrics["gear_damping_credit_rate"] == 0.0
        and metrics["post_contact_scored_steps"] == 0.0
    )
    expected_units = int(mode_spec["landing_units"] + mode_spec["roll_units"])
    complete_budget_reached = completed_units == expected_units
    protocol_integrity_passed = bool(contract_lock_passed and complete_budget_reached)
    scored_complete = bool(args.mode == "complete" and protocol_integrity_passed)
    landing_success_nonzero = bool(metrics["landing_success_rate"] > 0.0)
    hard_ood_nonzero = bool(metrics["hard_ood_landing_success_rate"] > 0.0)
    radius_gate_nonzero = bool(
        metrics["worst_radius_bin_landing_success_rate"] > 0.0
    )
    # Keep evidence maturity/close eligibility separate from the task-specific
    # confirmed-frontier performance gate. Praxist treats an explicit
    # promotion_eligible=false result as signal-only, so coupling this generic
    # durability flag to non-zero hard-OOD performance prevents otherwise clean
    # complete evaluations from satisfying the mature-peer close quorum.
    confirmed_performance_gate_passed = bool(
        scored_complete
        and landing_success_nonzero
        and hard_ood_nonzero
        and radius_gate_nonzero
    )
    promotion_eligible = bool(scored_complete)
    if scored_complete:
        source_lane = "performance"
    elif args.mode == "development" and protocol_integrity_passed:
        source_lane = "task_candidate"
    else:
        source_lane = "diagnostic"

    elapsed = time.perf_counter() - started
    metrics.update(
        {
            "effort_ratio": float(mode_spec["effort_ratio"]),
            "coverage_ratio": float(mode_spec["coverage_ratio"]),
            "evaluation_units_completed": int(completed_units),
            "evaluation_units_required": int(COMPLETE_UNITS),
            "mode_units_expected": expected_units,
            "scored_complete": scored_complete,
            "complete_eval": scored_complete,
            "protocol_integrity_passed": protocol_integrity_passed,
            "protocol_integrity_failed": not protocol_integrity_passed,
            "contract_lock_passed": contract_lock_passed,
            "landing_success_nonzero": landing_success_nonzero,
            "hard_ood_success_nonzero": hard_ood_nonzero,
            "radius_strata_gate_nonzero": radius_gate_nonzero,
            "confirmed_performance_gate_passed": confirmed_performance_gate_passed,
            "promotion_eligible": promotion_eligible,
            "parent_authorized": scored_complete,
            "close_eligible": scored_complete,
            "partial": args.mode != "complete",
            "is_smoke_eval": args.mode == "canary",
            "validation_only": args.mode == "roll_diagnostic",
            "scout_only": args.mode == "canary",
            "suspect_protocol": False,
            "suspect_leakage": False,
            "late_after_generation_boundary": False,
            "evaluator_wall_seconds": elapsed,
        }
    )
    effective_digest = _canonical_digest(controller_cfg)
    assigned = os.environ.get("PRAXIST_ASSIGNED_GPU_UUIDS", "").strip()
    producer_identity = _producer_identity_from_env()
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "variant_id": manifest["variant_id"],
        "variant_name": manifest.get("display_name", manifest["variant_id"]),
        **producer_identity,
        "success_definition": {
            "name": "landing_success",
            "only_success_standard": True,
            "endpoint": "interpolated_first_landing_leg_contact",
            "post_contact_damping_credit": False,
            "thresholds": SUCCESS_THRESHOLDS,
        },
        "metrics": metrics,
        "frontier_lane": source_lane,
        "promotion_lane": source_lane,
        "evidence_stage": args.mode,
        "eval_stage": args.mode,
        "tier": args.mode,
        "result_status": "scored_complete" if scored_complete else args.mode,
        "scored_complete": scored_complete,
        "complete_eval": scored_complete,
        "confirmed_performance_gate_passed": confirmed_performance_gate_passed,
        "promotion_eligible": promotion_eligible,
        "effort_ratio": float(mode_spec["effort_ratio"]),
        "coverage_ratio": float(mode_spec["coverage_ratio"]),
        "actual_evaluation_units": int(completed_units),
        "reference_evaluation_units": int(COMPLETE_UNITS),
        "evaluation_units_completed": int(completed_units),
        "evaluation_units_required": int(COMPLETE_UNITS),
        "effective_config": controller_cfg,
        "effective_config_complete": True,
        "effective_config_digest": effective_digest,
        "effective_config_schema": "candidate.load_config(controller_config.json):json:v1",
        "design_dimensions": manifest["design_dimensions"],
        "changed_modules": manifest["changed_modules"],
        "method_class": manifest["method_class"],
        "extra": {
            "frontier_lane": source_lane,
            "promotion_lane": source_lane,
            "evidence_stage": args.mode,
            "evaluation_summary": str((args.out_dir / "evaluation_summary.json").resolve()),
            "design_dimensions": manifest["design_dimensions"],
            "effective_config_digest": effective_digest,
            **producer_identity,
        },
        "protocol_integrity": {
            "passed": protocol_integrity_passed,
            "contract_lock_passed": contract_lock_passed,
            "frozen_assets_attested": True,
            "outcome_dependent_selection": False,
            "forbidden_learning_methods_detected": False,
            "candidate_static_scan": candidate_audit,
            "frozen_audit_limits": FROZEN_AUDIT_LIMITS,
            "single_success_standard": True,
            "post_contact_damping_credit_forbidden": True,
        },
        "dataset": {
            "selection_seed_near_hard": VALIDATION_SEED,
            "selection_seed_nominal": NOMINAL_SEED,
            "complete_landing_units": COMPLETE_LANDING_UNITS,
            "roll_units": ROLL_UNITS,
            "complete_units": COMPLETE_UNITS,
            "mode_units": expected_units,
            "source_bank_hashes": {
                name: expected for name, (_, expected) in SOURCE_BANKS.items()
            },
            "historical_formal_overlap": False,
            "initial_mass_kg": INITIAL_MASS_KG,
            "initial_fuel_kg": INITIAL_FUEL_KG,
            "mass_empty_kg": MASS_EMPTY_KG,
        },
        "plant": {
            "integrator": "rk4",
            "substeps": 1,
            "dt": float(plant_cfg.dt),
            "max_steps": int(plant_cfg.max_steps),
            "protocol_initial_mass_override_kg": INITIAL_MASS_KG,
            "protocol_fuel_scale": 1.0,
            "landing_scoring_endpoint": "interpolated_first_leg_contact",
            "post_contact_scored_steps": 0,
            "provenance": provenance(),
            "attested_hashes": frozen_hashes,
        },
        "execution": {
            "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": elapsed,
            "block_seconds": block_timings,
            "batch_size": args.batch_size,
            "python": platform.python_version(),
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "scheduler_managed": bool(assigned),
            "assigned_gpu_uuids": [item for item in assigned.split(",") if item],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
        },
        "details": details,
    }
    summary_path = args.out_dir / "evaluation_summary.json"
    _write_json(summary_path, summary)
    _write_json(
        args.out_dir / "resolved_effective_config.json",
        {
            "variant_id": manifest["variant_id"],
            "effective_config": controller_cfg,
            "effective_config_complete": True,
            "effective_config_digest": effective_digest,
        },
    )
    return summary, summary_path


def _failure_summary(args: argparse.Namespace, exc: BaseException) -> tuple[dict[str, Any], Path]:
    variant_id = args.variant_dir.name
    try:
        payload = json.loads((args.variant_dir / "variant.json").read_text(encoding="utf-8"))
        candidate = str(payload.get("variant_id") or "").strip()
        if VARIANT_ID_RE.fullmatch(candidate):
            variant_id = candidate
    except Exception:
        pass
    metrics = _default_metrics()
    metrics.update(
        {
            "effort_ratio": 0.0,
            "coverage_ratio": 0.0,
            "evaluation_units_completed": 0,
            "evaluation_units_required": COMPLETE_UNITS,
            "scored_complete": False,
            "complete_eval": False,
            "protocol_integrity_passed": False,
            "protocol_integrity_failed": True,
            "contract_lock_passed": False,
            "promotion_eligible": False,
            "parent_authorized": False,
            "close_eligible": False,
            "partial": True,
            "is_smoke_eval": args.mode == "canary",
            "validation_only": args.mode == "roll_diagnostic",
            "scout_only": args.mode == "canary",
            "suspect_protocol": True,
            "suspect_leakage": False,
            "late_after_generation_boundary": False,
            "evaluator_wall_seconds": 0.0,
        }
    )
    producer_identity = _producer_identity_from_env()
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "variant_id": variant_id,
        **producer_identity,
        "metrics": metrics,
        "frontier_lane": "diagnostic",
        "promotion_lane": "diagnostic",
        "evidence_stage": args.mode,
        "eval_stage": args.mode,
        "tier": args.mode,
        "result_status": "protocol_failed",
        "scored_complete": False,
        "complete_eval": False,
        "promotion_eligible": False,
        "effort_ratio": 0.0,
        "coverage_ratio": 0.0,
        "actual_evaluation_units": 0,
        "reference_evaluation_units": COMPLETE_UNITS,
        "evaluation_units_completed": 0,
        "evaluation_units_required": COMPLETE_UNITS,
        "effective_config": {},
        "effective_config_complete": False,
        "design_dimensions": {},
        "extra": {
            "frontier_lane": "diagnostic",
            "promotion_lane": "diagnostic",
            "evidence_stage": args.mode,
            "evaluation_summary": str((args.out_dir / "evaluation_summary.json").resolve()),
            **producer_identity,
        },
        "protocol_integrity": {
            "passed": False,
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc)[:2_000],
        },
    }
    path = args.out_dir / "evaluation_summary.json"
    _write_json(path, summary)
    (args.out_dir / "failure_traceback.log").write_text(
        "".join(traceback.format_exception(exc)), encoding="utf-8"
    )
    return summary, path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a deterministic classical Rocket Booster Recovery controller variant."
    )
    parser.add_argument(
        "--variant-dir",
        type=Path,
        required=True,
        help="Directory containing controller.py, controller_config.json, and variant.json.",
    )
    parser.add_argument(
        "--mode",
        choices=("canary", "development", "roll_diagnostic", "complete"),
        default="development",
    )
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        dest="out_dir",
        type=Path,
        required=True,
        help=(
            "Unique result directory for this peer/variant/protocol. Scheduled runs "
            "must use the canonical --output-dir spelling; --out-dir is retained only "
            "for backward-compatible manual invocations."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args(argv)
    args.variant_dir = args.variant_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if args.batch_size < 1 or args.batch_size > 4096:
        parser.error("--batch-size must be between 1 and 4096")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary, path = evaluate(args)
    except Exception as exc:
        summary, path = _failure_summary(args, exc)
        print(
            json.dumps(
                {
                    "summary": str(path),
                    "variant_id": summary["variant_id"],
                    "status": "protocol_failed",
                    "error": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        raise SystemExit(2) from exc
    print(
        json.dumps(
            {
                "summary": str(path),
                "variant_id": summary["variant_id"],
                "stage": summary["evidence_stage"],
                "scored_complete": summary["scored_complete"],
                "landing_success_rate": summary["metrics"]["landing_success_rate"],
                "first_contact_sink_speed_p95_mps": summary["metrics"][
                    "first_contact_sink_speed_p95_mps"
                ],
                "elapsed_seconds": summary["execution"]["elapsed_seconds"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

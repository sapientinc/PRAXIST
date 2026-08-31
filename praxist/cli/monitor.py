"""``praxist --monitor`` - live read-only operator dashboard for Praxist runs.

The command is intentionally a thin consumer of existing status surfaces:
``praxist status`` rows, ``orchestrator_status.json``, peer-memory health, recent
logs, and lightweight host-load probes. It never mutates run artifacts or
research-loop state.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from praxist.cli import status
from praxist.cli.registry import STATE_RUNNING
from praxist.plugins.workflow_stages.research_loop.backend.orchestrator_status import (
    read_effective_orchestrator_status,
)

DEFAULT_INTERVAL_SECONDS = 0.2
DEFAULT_PLAIN_INTERVAL_SECONDS = 1.0
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
SAMPLER_SHUTDOWN_TIMEOUT_SECONDS = 6.5
MIN_INTERVAL_SECONDS = 0.05
DEFAULT_LOG_LINES = 18
DEFAULT_PEER_LIMIT = 24
MAX_JSON_BYTES = 1_000_000
MAX_LOG_BYTES = 256_000
ANSI_STYLE_RE = re.compile(r"\x1b\[[0-9;]*m")
BRAND_RGB = (60, 85, 255)
LIVE_RGB = (61, 214, 140)
WARNING_RGB = (245, 184, 65)
ERROR_RGB = (255, 91, 111)
UNKNOWN_RGB = (148, 163, 184)

# Low-density terminal derivatives of docs/assets/brand/praxist-mark.svg. The
# SVG remains canonical; these masks retain its footprint with a lighter stroke.
_BRAND_MARK_PIXELS = (
    "01111111000000",
    "01000001110000",
    "01000000011000",
    "01100000001000",
    "00011100101000",
    "00010110001000",
    "00011011011000",
    "00001101110000",
    "00000001001000",
    "00000001001000",
    "00000001111000",
    "00000000110000",
)
_COMPACT_BRAND_MARK_PIXELS = (
    "11111000",
    "01000100",
    "01100010",
    "00011110",
    "00001000",
    "00001100",
)


@dataclass(frozen=True)
class MonitorTarget:
    """Operator-selected monitor target."""

    run_id: str | None = None
    run_dir: str | None = None
    task_path: str | None = None
    latest: bool = False


@dataclass(frozen=True)
class HardwareSnapshot:
    """Small host-load view for monitor rendering."""

    loadavg: str = "-"
    memory: str = "-"
    gpus: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MonitorSnapshot:
    """One complete monitor frame."""

    rows: list[status.StatusRow]
    selected: status.StatusRow | None
    target: MonitorTarget
    generated_at: str
    orchestrator_status: dict[str, Any] = field(default_factory=dict)
    phase: str = "unknown"
    recent_logs: list[str] = field(default_factory=list)
    hardware: HardwareSnapshot = field(default_factory=HardwareSnapshot)
    warnings: list[str] = field(default_factory=list)


class TextMonitorRenderer:
    """Simple text renderer kept separate from collection for future TUI backends."""

    def __init__(self, *, peer_limit: int = DEFAULT_PEER_LIMIT):
        self.peer_limit = max(1, int(peer_limit))

    def render(self, snapshot: MonitorSnapshot) -> str:
        lines: list[str] = []
        lines.append("Praxist Monitor")
        lines.append("=" * 80)
        lines.append(f"Updated: {snapshot.generated_at}")
        lines.append("Mode: read-only; Ctrl-C exits this display without affecting the Praxist run")
        lines.append("")
        lines.extend(self._render_runs(snapshot))
        if snapshot.selected is not None:
            lines.append("")
            lines.extend(self._render_selected(snapshot))
            lines.append("")
            lines.extend(self._render_peers(snapshot.selected))
            lines.append("")
            lines.extend(self._render_logs(snapshot.recent_logs))
        lines.append("")
        lines.extend(self._render_hardware(snapshot.hardware))
        if snapshot.warnings or snapshot.hardware.warnings:
            lines.append("")
            lines.append("Warnings")
            for warning in [*snapshot.warnings, *snapshot.hardware.warnings]:
                lines.append(f"- {warning}")
        return "\n".join(_terminal_safe_text(line) for line in lines).rstrip() + "\n"

    def _render_runs(self, snapshot: MonitorSnapshot) -> list[str]:
        lines = ["Runs"]
        if not snapshot.rows:
            lines.append("- No Praxist run rows detected.")
            return lines
        header = f"{'SEL':3} {'PID':>7} {'SRC':8} {'STATE':8} {'AGE':>10} {'GEN':>5} {'PEERS':>11}  RUN_ID"
        lines.append(header)
        lines.append("-" * min(100, len(header) + 30))
        for row in snapshot.rows:
            selected = "*" if snapshot.selected is row else " "
            gen = str(row.generation) if row.generation is not None else "-"
            peers = _format_peer_health(row.peer_health_summary)
            run_id = row.run_id or "-"
            lines.append(
                f"{selected:3} {row.pid:7d} {row.source[:8]:8} {row.state[:8]:8} "
                f"{(row.etime or '-')[:10]:>10} {gen:>5} {peers:>11}  {run_id}"
            )
        return lines

    def _render_selected(self, snapshot: MonitorSnapshot) -> list[str]:
        row = snapshot.selected
        if row is None:
            return [
                "Selected Run",
                "- No single selected run; pass --run-id/--run-dir or --latest.",
            ]
        orch = snapshot.orchestrator_status
        lines = ["Selected Run"]
        lines.append(f"- run_id: {row.run_id or '-'}")
        lines.append(f"- run_dir: {row.run_dir or '-'}")
        lines.append(f"- task_path: {row.task_path or '-'}")
        lines.append(f"- provider/model: {row.model_provider_ref or '-'} / {row.model or '-'}")
        lines.append(
            "- generation: "
            f"{_display(orch.get('current_generation'), row.generation)} / "
            f"completed {_display(orch.get('generations_completed'), '-')}"
        )
        lines.append(
            "- cohort/findings/frontier/gems: "
            f"{_display(orch.get('cohort_size'), '-')} peers / "
            f"{_display(orch.get('findings_total'), row.findings_total)} findings / "
            f"{_display(orch.get('frontier_candidates'), '-')} frontier / "
            f"{_display(orch.get('gems_count'), '-')} gems"
        )
        lines.append(f"- phase: {snapshot.phase}")
        lines.extend(f"- {line}" for line in _scoreless_limit_lines(orch))
        lines.append(f"- updated_at: {_display(orch.get('updated_at'), row.updated_at)}")
        blocker = str(orch.get("gen_promotion_blocker") or "").strip()
        if blocker:
            lines.append(f"- promotion_blocker: {blocker}")
        mature = _result_summary(orch.get("best_mature_result"))
        if mature:
            lines.append(f"- best_mature_result: {mature}")
        signal = _result_summary(orch.get("best_validation_signal"), include_reason=True)
        if signal:
            lines.append(f"- best_validation_signal: {signal}")
        scheduler_blockers = _scheduler_blocker_summary(orch.get("resource_scheduler"))
        if scheduler_blockers:
            lines.append(f"- scheduler_wait: {scheduler_blockers}")
        peer_mix = orch.get("last_peer_mix")
        if isinstance(peer_mix, dict) and peer_mix:
            ratio = peer_mix.get("mature_constructive_ratio")
            target = peer_mix.get("target_constructive_ratio")
            lines.append(
                "- last_peer_mix: "
                f"constructive={_format_float(ratio)} target={_format_float(target)}"
            )
        stop_audit = orch.get("last_stop_audit")
        if isinstance(stop_audit, dict) and stop_audit:
            reason = stop_audit.get("trigger_reason") or stop_audit.get("signal_file")
            if stop_audit.get("mode") == "scoreless":
                lines.append(f"- last_stop: {reason or '-'} maturity=not_scored")
            else:
                mature = stop_audit.get("mature_result_peers")
                required = stop_audit.get("required_mature_result_peers")
                lines.append(f"- last_stop: {reason or '-'} mature={mature or 0}/{required or 0}")
        return lines

    def _render_peers(self, row: status.StatusRow) -> list[str]:
        lines = ["Peers"]
        peers = row.peers or []
        if not peers:
            lines.append("- No peer health rows available yet.")
            return lines
        header = (
            f"{'PEER':18} {'HEALTH':7} {'STATE':18} {'VARIANT':24} {'BEST':>12} {'BASE':10} UPDATED"
        )
        lines.append(header)
        lines.append("-" * min(120, len(header)))
        for peer in peers[: self.peer_limit]:
            peer_id = str(peer.get("peer_id") or "-")
            health = str(peer.get("health") or "-")
            state_text = _truncate(str(peer.get("research_state") or "-"), 18)
            variant = _truncate(str(peer.get("active_variant") or "-"), 24)
            best = _format_float(peer.get("best_metric_value"))
            base = str(peer.get("baseline_status") or "-")[:10]
            updated = str(peer.get("last_updated_utc") or "-")
            reason = str(peer.get("health_reason") or "").strip()
            lines.append(
                f"{peer_id[:18]:18} {health[:7]:7} {state_text:18} {variant:24} "
                f"{best:>12} {base:10} {updated}"
            )
            if reason:
                lines.append(f"  health_reason[{peer_id}]: {reason}")
        if len(peers) > self.peer_limit:
            lines.append(f"- {len(peers) - self.peer_limit} more peers hidden by --peer-limit.")
        return lines

    def _render_logs(self, logs: list[str]) -> list[str]:
        lines = ["Recent Logs"]
        if not logs:
            lines.append("- No recent run logs found.")
            return lines
        lines.extend(logs)
        return lines

    def _render_hardware(self, hw: HardwareSnapshot) -> list[str]:
        lines = ["Hardware"]
        lines.append(f"- loadavg: {hw.loadavg}")
        lines.append(f"- memory: {hw.memory}")
        if hw.gpus:
            lines.append("- gpu:")
            for gpu in hw.gpus:
                lines.append(f"  - {gpu}")
        else:
            lines.append("- gpu: unavailable")
        return lines


class TuiMonitorRenderer:
    """Fullscreen terminal renderer for the live operator dashboard.

    This intentionally stays dependency-free. It behaves like a Ratatui-style
    frame renderer: collect a snapshot, render a complete frame sized to the
    current terminal, then let the display loop swap the frame in place.
    """

    def __init__(
        self,
        *,
        peer_limit: int = DEFAULT_PEER_LIMIT,
        color: bool | None = None,
        unicode_glyphs: bool = True,
    ):
        self.peer_limit = max(1, int(peer_limit))
        self.color = ("NO_COLOR" not in os.environ) if color is None else bool(color)
        self.unicode_glyphs = bool(unicode_glyphs)

    def render(
        self,
        snapshot: MonitorSnapshot,
        *,
        width: int,
        height: int,
        frame_index: int = 0,
    ) -> str:
        width = max(1, int(width))
        height = max(1, int(height))
        if width < 20 or height < 6:
            return self._tiny_layout(snapshot, width=width, height=height, frame_index=frame_index)
        lines: list[str] = []
        lines.extend(self._header(snapshot, width=width, height=height, frame_index=frame_index))
        if width < 50 or height < 14:
            lines.extend(self._micro_layout(snapshot, width=width, height=height - len(lines) - 1))
        elif width >= 118 and height >= 28:
            lines.extend(
                self._wide_layout(
                    snapshot,
                    width=width,
                    height=height - len(lines) - 1,
                    frame_index=frame_index,
                )
            )
        else:
            lines.extend(
                self._compact_layout(
                    snapshot,
                    width=width,
                    height=height - len(lines) - 1,
                    frame_index=frame_index,
                )
            )
        lines.append(self._footer(width))
        lines = lines[:height]
        while len(lines) < height:
            lines.append("")
        return "\n".join(_fit_ansi(line, width) for line in lines)

    def _tiny_layout(
        self,
        snapshot: MonitorSnapshot,
        *,
        width: int,
        height: int,
        frame_index: int,
    ) -> str:
        mark = _truecolor("#", BRAND_RGB, self.color)
        pulse = _pulse_glyph(frame_index, unicode_glyphs=False)
        run_id = snapshot.selected.run_id if snapshot.selected else "no-run"
        lines = [mark + _fit_plain(" PRAXIST", max(0, width - 1))]
        lines.extend(
            [
                _fit_plain(f"LIVE {pulse}", width),
                _fit_plain(str(run_id or "-"), width),
                _fit_plain(str(snapshot.phase), width),
            ]
        )
        lines = lines[:height]
        while len(lines) < height:
            lines.append("")
        return "\n".join(_fit_ansi(line, width) for line in lines)

    def _header(
        self,
        snapshot: MonitorSnapshot,
        *,
        width: int,
        height: int,
        frame_index: int,
    ) -> list[str]:
        run_id = _terminal_safe_text(
            str(snapshot.selected.run_id or "-") if snapshot.selected else "no-run-selected"
        )
        phase = _terminal_safe_text(str(snapshot.phase))
        generated_at = _terminal_safe_text(str(snapshot.generated_at))
        pulse = _pulse_glyph(frame_index, unicode_glyphs=self.unicode_glyphs)
        live = _truecolor(f"LIVE {pulse}", LIVE_RGB, self.color)
        if width >= 90 and height >= 30:
            mark = _render_brand_mark(
                _BRAND_MARK_PIXELS,
                color=self.color,
                unicode_glyphs=self.unicode_glyphs,
            )
            details = [
                _join_left_right("PRAXIST  RESEARCH MONITOR", generated_at, width - 16),
                live,
                "Operator visibility without research-process control",
                f"run   {run_id or '-'}",
                f"phase {phase}",
                "Ctrl-C exits only this monitor",
            ]
            details[0] = _style(details[0], "1;37", self.color)
            details[2] = _style(details[2], "2;37", self.color)
            return [_fit_ansi(line, width) for line in _hstack(mark, details, gap=2)]
        if width >= 52 and height >= 16:
            mark = _render_brand_mark(
                _COMPACT_BRAND_MARK_PIXELS,
                color=self.color,
                unicode_glyphs=self.unicode_glyphs,
            )
            detail_width = max(1, width - 10)
            details = [
                _style(_fit_plain("PRAXIST  RESEARCH MONITOR", detail_width), "1;37", self.color),
                _fit_plain(
                    f"{_strip_ansi(live)} | {run_id or '-'}",
                    detail_width,
                ),
                _style(
                    _fit_plain(f"{phase} | sampled {generated_at}", detail_width),
                    "2;37",
                    self.color,
                ),
            ]
            details[1] = details[1].replace("LIVE " + pulse, live, 1)
            return [_fit_ansi(line, width) for line in _hstack(mark, details, gap=2)]
        mark_glyph = "◆" if self.unicode_glyphs else "#"
        mark = _truecolor(mark_glyph, BRAND_RGB, self.color)
        line = _join_left_right("Praxist Monitor", generated_at, max(1, width - 2))
        subtitle = f"{_strip_ansi(live)} | {run_id or '-'} | {phase}"
        return [
            _fit_ansi(mark + " " + _style(line, "1;36", self.color), width),
            _fit_ansi(subtitle.replace("LIVE " + pulse, live, 1), width),
        ]

    def _micro_layout(self, snapshot: MonitorSnapshot, *, width: int, height: int) -> list[str]:
        if height <= 0:
            return []
        row = snapshot.selected
        orch = snapshot.orchestrator_status
        run_id = row.run_id if row else "-"
        generation = _display(orch.get("current_generation"), row.generation if row else "-")
        findings = _display(orch.get("findings_total"), row.findings_total if row else "-")
        peers = _format_peer_health(row.peer_health_summary if row else None)
        lines = [
            f"run: {run_id}",
            f"phase: {snapshot.phase}",
            f"gen: {generation}  findings: {findings}",
            f"peers: {peers}  load: {snapshot.hardware.loadavg}",
        ]
        warnings = [*snapshot.warnings, *snapshot.hardware.warnings]
        if warnings:
            lines.append(f"warn: {warnings[0]}")
        return [_fit_plain(line, width) for line in lines[:height]]

    def _wide_layout(
        self,
        snapshot: MonitorSnapshot,
        *,
        width: int,
        height: int,
        frame_index: int,
    ) -> list[str]:
        if height <= 0:
            return []
        log_h = self._stream_log_height(height)
        dashboard_h = max(0, height - log_h)
        gap = 2
        left_w = max(44, width // 2 - 1)
        right_w = max(44, width - left_w - gap)
        aux_h = min(6, max(4, dashboard_h // 4))
        top_h = min(8, max(6, (dashboard_h - aux_h) // 2))
        peer_h = max(3, dashboard_h - top_h - aux_h)
        lines: list[str] = []
        if dashboard_h >= 9:
            lines.extend(
                _hstack(
                    self._box(
                        "Runs",
                        self._runs_lines(snapshot),
                        width=left_w,
                        height=top_h,
                        color="36",
                    ),
                    self._box(
                        "Selected Run",
                        self._selected_lines(snapshot),
                        width=right_w,
                        height=top_h,
                        color="35",
                    ),
                    gap=gap,
                )
            )
            lines.extend(
                self._box(
                    "Peers (bounded)",
                    self._peer_lines(snapshot.selected, frame_index=frame_index),
                    width=width,
                    height=peer_h,
                    color="32",
                    ansi_content=True,
                )
            )
            logs_w = max(60, int(width * 0.64))
            hw_w = max(38, width - logs_w - gap)
            lines.extend(
                _hstack(
                    self._box(
                        "Recent Logs",
                        self._summary_log_lines(snapshot),
                        width=logs_w,
                        height=aux_h,
                        color="33",
                    ),
                    self._box(
                        "Hardware / Warnings",
                        self._hardware_lines(snapshot),
                        width=hw_w,
                        height=aux_h,
                        color="34",
                    ),
                    gap=gap,
                )
            )
        else:
            lines.extend(self._micro_layout(snapshot, width=width, height=dashboard_h))
        lines.extend(
            self._stream_log_area(
                snapshot,
                width=width,
                height=max(0, height - len(lines)),
            )
        )
        return lines[:height]

    def _compact_layout(
        self,
        snapshot: MonitorSnapshot,
        *,
        width: int,
        height: int,
        frame_index: int,
    ) -> list[str]:
        if height <= 0:
            return []
        log_h = self._stream_log_height(height)
        dashboard_h = max(0, height - log_h)
        section_specs = [
            ("Runs", self._runs_lines(snapshot), 4, "36"),
            ("Selected Run", self._selected_lines(snapshot), 5, "35"),
            (
                "Peers (bounded)",
                self._peer_lines(snapshot.selected, frame_index=frame_index),
                4,
                "32",
            ),
            ("Recent Logs", self._summary_log_lines(snapshot), 3, "33"),
            ("Hardware / Warnings", self._hardware_lines(snapshot), 3, "34"),
        ]
        lines: list[str] = []
        remaining = dashboard_h
        for idx, (title, content, preferred, color) in enumerate(section_specs):
            if remaining <= 0:
                break
            rest_min = max(0, len(section_specs) - idx - 1) * 3
            if idx == len(section_specs) - 1:
                section_h = remaining
            else:
                section_h = max(3, min(preferred, remaining - rest_min))
            if section_h < 3:
                break
            lines.extend(
                self._box(
                    title,
                    content,
                    width=width,
                    height=section_h,
                    color=color,
                    ansi_content=title == "Peers (bounded)",
                )
            )
            remaining -= section_h
        lines.extend(
            self._stream_log_area(
                snapshot,
                width=width,
                height=max(0, height - len(lines)),
            )
        )
        return lines[:height]

    def _runs_lines(self, snapshot: MonitorSnapshot) -> list[str]:
        if not snapshot.rows:
            return ["No Praxist run rows detected."]
        lines = [
            _columns(
                [
                    ("", 1, "left"),
                    ("PID", 7, "right"),
                    ("STATE", 8, "left"),
                    ("GEN", 5, "right"),
                    ("PEERS", 11, "right"),
                    ("RUN_ID", 28, "left"),
                ]
            )
        ]
        for row in snapshot.rows[:8]:
            selected = "*" if snapshot.selected is row else " "
            gen = str(row.generation) if row.generation is not None else "-"
            peers = _format_peer_health(row.peer_health_summary)
            run_id = row.run_id or "-"
            lines.append(
                _columns(
                    [
                        (selected, 1, "left"),
                        (str(row.pid), 7, "right"),
                        (row.state, 8, "left"),
                        (gen, 5, "right"),
                        (peers, 11, "right"),
                        (run_id, 28, "left"),
                    ]
                )
            )
        if len(snapshot.rows) > 8:
            lines.append(f"{len(snapshot.rows) - 8} more runs hidden.")
        return lines

    def _selected_lines(self, snapshot: MonitorSnapshot) -> list[str]:
        row = snapshot.selected
        if row is None:
            return ["No selected run.", "Pass --latest, --run-id, or --run-dir."]
        orch = snapshot.orchestrator_status
        generation = (
            f"{_display(orch.get('current_generation'), row.generation)} "
            f"(completed {_display(orch.get('generations_completed'), '-')})"
        )
        lines = [
            f"run_id: {row.run_id or '-'} | generation: {generation}",
            "cohort/findings/frontier/gems: "
            f"{_display(orch.get('cohort_size'), '-')} / "
            f"{_display(orch.get('findings_total'), row.findings_total)} / "
            f"{_display(orch.get('frontier_candidates'), '-')} / "
            f"{_display(orch.get('gems_count'), '-')}",
        ]
        lines.extend(_scoreless_limit_lines(orch))
        mature = _result_summary(orch.get("best_mature_result"))
        if mature:
            lines.append(f"best_mature: {mature}")
        signal = _result_summary(orch.get("best_validation_signal"), include_reason=True)
        if signal:
            lines.append(f"validation_signal: {signal}")
        lines.extend(
            [
                f"task: {row.task_path or '-'}",
                f"model: {row.model_provider_ref or '-'} / {row.model or '-'}",
                f"run_dir: {row.run_dir or '-'}",
                f"hardware: load {snapshot.hardware.loadavg} | mem {snapshot.hardware.memory}",
                f"updated: {_display(orch.get('updated_at'), row.updated_at)}",
            ]
        )
        if snapshot.hardware.gpus:
            lines.append(f"gpu: {_truncate(' | '.join(snapshot.hardware.gpus), 96)}")
        warnings = [*snapshot.warnings, *snapshot.hardware.warnings]
        if warnings:
            lines.append(f"warning: {_truncate(warnings[0], 96)}")
        blocker = str(orch.get("gen_promotion_blocker") or "").strip()
        if blocker:
            lines.append(f"promotion_blocker: {blocker}")
        scheduler_blockers = _scheduler_blocker_summary(orch.get("resource_scheduler"))
        if scheduler_blockers:
            lines.append(f"scheduler_wait: {scheduler_blockers}")
        peer_mix = orch.get("last_peer_mix")
        if isinstance(peer_mix, dict) and peer_mix:
            ratio = peer_mix.get("mature_constructive_ratio")
            target = peer_mix.get("target_constructive_ratio")
            lines.append(
                f"last_peer_mix: constructive={_format_float(ratio)} target={_format_float(target)}"
            )
        stop_audit = orch.get("last_stop_audit")
        if isinstance(stop_audit, dict) and stop_audit:
            reason = stop_audit.get("trigger_reason") or stop_audit.get("signal_file")
            if stop_audit.get("mode") == "scoreless":
                lines.append(f"last_stop: {reason or '-'} maturity=not_scored")
            else:
                mature = stop_audit.get("mature_result_peers")
                required = stop_audit.get("required_mature_result_peers")
                lines.append(f"last_stop: {reason or '-'} mature={mature or 0}/{required or 0}")
        return lines

    def _peer_lines(
        self,
        row: status.StatusRow | None,
        *,
        frame_index: int = 0,
    ) -> list[str]:
        if row is None:
            return ["No selected run."]
        peers = row.peers or []
        if not peers:
            return ["No peer health rows available yet."]
        lines = [
            _columns(
                [
                    ("PEER", 18, "left"),
                    ("HEALTH / REASON", 20, "left"),
                    ("STATE", 18, "left"),
                    ("VARIANT", 24, "left"),
                    ("BEST", 10, "right"),
                ]
            )
        ]
        for peer in peers[: self.peer_limit]:
            peer_id = str(peer.get("peer_id") or "-")
            health = str(peer.get("health") or "-")
            reason = str(peer.get("health_reason") or "").strip()
            health_display = health if not reason else f"{health}: {reason}"
            pulse = _pulse_glyph(frame_index, unicode_glyphs=self.unicode_glyphs)
            health_cell = _fit_generated_ansi(
                _truecolor(pulse, _peer_health_rgb(health), self.color)
                + " "
                + _fit_plain(health_display, 18),
                20,
            )
            state_text = _truncate(str(peer.get("research_state") or "-"), 18)
            variant = _truncate(str(peer.get("active_variant") or "-"), 24)
            best = _format_float(peer.get("best_metric_value"))
            lines.append(
                _columns([(peer_id, 18, "left")])
                + " "
                + health_cell
                + " "
                + _columns(
                    [
                        (state_text, 18, "left"),
                        (variant, 24, "left"),
                        (best, 10, "right"),
                    ]
                )
            )
        if len(peers) > self.peer_limit:
            lines.append(f"{len(peers) - self.peer_limit} more peers hidden by --peer-limit.")
        return lines

    def _hardware_lines(self, snapshot: MonitorSnapshot) -> list[str]:
        hw = snapshot.hardware
        lines = [f"loadavg: {hw.loadavg}", f"memory: {hw.memory}"]
        if hw.gpus:
            lines.extend(hw.gpus)
        else:
            lines.append("gpu: unavailable")
        warnings = [*snapshot.warnings, *hw.warnings]
        if warnings:
            lines.append("")
            lines.append("warnings:")
            lines.extend(f"- {warning}" for warning in warnings)
        return lines

    def _summary_log_lines(self, snapshot: MonitorSnapshot) -> list[str]:
        return snapshot.recent_logs[-4:] if snapshot.recent_logs else ["No recent run logs found."]

    def _stream_log_height(self, height: int) -> int:
        if height <= 0:
            return 0
        if height < 12:
            return max(0, min(height, max(3, height // 3)))
        return max(5, min(height - 9, (height * 2) // 5))

    def _stream_log_area(
        self,
        snapshot: MonitorSnapshot,
        *,
        width: int,
        height: int,
    ) -> list[str]:
        if height <= 0:
            return []
        title = " Live log stream "
        if _cell_width(title) < width:
            title += "-" * max(0, width - _cell_width(title))
        lines = [_style(_fit_plain(title, width), "1;33", self.color)]
        if height == 1:
            return lines
        body_height = height - 1
        raw_logs = snapshot.recent_logs or ["No recent run logs found."]
        body: list[str] = []
        for raw in raw_logs:
            body.extend(_wrap_plain(_strip_ansi(raw), max(1, width - 1)))
        body = body[-body_height:]
        while len(body) < body_height:
            body.insert(0, "")
        lines.extend(_fit_plain(" " + line, width) if line else "" for line in body)
        return lines[:height]

    def stream_scroll_region(self, *, width: int, height: int) -> tuple[int, int]:
        width = max(1, int(width))
        height = max(1, int(height))
        if width < 20 or height < 6:
            return height, height
        content_height = height - self._header_height(width=width, height=height) - 1
        if width < 50 or height < 14:
            return max(1, height - 1), max(1, height - 1)
        log_h = self._stream_log_height(content_height)
        if log_h <= 1:
            return max(1, height - 1), max(1, height - 1)
        start = max(1, height - log_h)
        end = max(start, height - 1)
        return start, end

    @staticmethod
    def _header_height(*, width: int, height: int) -> int:
        if width >= 90 and height >= 30:
            return 6
        if width >= 52 and height >= 16:
            return 3
        return 2

    def _box(
        self,
        title: str,
        content: list[str],
        *,
        width: int,
        height: int,
        color: str,
        ansi_content: bool = False,
    ) -> list[str]:
        width = max(12, width)
        height = max(3, height)
        inner = width - 4
        title_text = f" {title} "
        top = "+" + "-" * (width - 2) + "+"
        if len(title_text) < width - 2:
            top = "+" + title_text + "-" * (width - 2 - len(title_text)) + "+"
        lines = [_style(top, color, self.color)]
        for raw in content[: height - 2]:
            line = (
                _fit_generated_ansi(raw, inner)
                if ansi_content
                else _fit_plain(_strip_ansi(raw), inner)
            )
            lines.append(
                _style("|", color, self.color)
                + " "
                + _pad_cells(line, inner)
                + " "
                + _style("|", color, self.color)
            )
        while len(lines) < height - 1:
            lines.append(
                _style("|", color, self.color)
                + " "
                + " " * inner
                + " "
                + _style("|", color, self.color)
            )
        lines.append(_style("+" + "-" * (width - 2) + "+", color, self.color))
        return lines

    def _footer(self, width: int) -> str:
        if width < 64:
            text = " Ctrl-C exit | --plain "
        else:
            text = " Ctrl-C exits monitor only | run remains active | --plain "
        return _style(_fit_plain(text, width), "2;37", self.color)


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the compatibility ``praxist monitor`` subcommand.

    The public entry point is ``praxist --monitor``.  The top-level dispatcher
    normalizes that spelling to this parser so both forms share one contract.
    """
    parser = subparsers.add_parser(
        "monitor",
        prog="praxist --monitor",
        help="Watch Praxist run state in a live read-only terminal dashboard.",
        description=(
            "Render a live read-only dashboard from praxist status, orchestrator "
            "snapshots, peer memory health, recent logs, and lightweight host load. "
            "The dashboard runs directly in the current terminal and never controls "
            "the Praxist research process."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", dest="run_id", default=None, help="Monitor one run id.")
    parser.add_argument("--run-dir", dest="run_dir", default=None, help="Monitor one run dir.")
    parser.add_argument(
        "--task-path",
        dest="task_path",
        default=None,
        help="Prefer active rows for this task path.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Select the latest active run row when more than one exists.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help=(
            f"Frame interval in seconds (default: {DEFAULT_INTERVAL_SECONDS:g} for the "
            f"fullscreen TUI, {DEFAULT_PLAIN_INTERVAL_SECONDS:g} for plain text)."
        ),
    )
    parser.add_argument("--once", action="store_true", help="Render one frame and exit.")
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep refreshing even when stdout is not an interactive terminal.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Append frames instead of clearing the terminal between refreshes.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Use the legacy plain-text monitor instead of the fullscreen TUI.",
    )
    parser.add_argument(
        "--log-lines",
        type=int,
        default=DEFAULT_LOG_LINES,
        help=f"Recent log lines to show for the selected run (default: {DEFAULT_LOG_LINES}).",
    )
    parser.add_argument(
        "--peer-limit",
        type=int,
        default=DEFAULT_PEER_LIMIT,
        help=f"Maximum peer rows to render (default: {DEFAULT_PEER_LIMIT}).",
    )
    parser.set_defaults(func=cmd_monitor)


def cmd_monitor(args: argparse.Namespace) -> int:
    """Handler for the public ``praxist --monitor`` entry point."""
    target = MonitorTarget(
        run_id=args.run_id,
        run_dir=str(Path(args.run_dir).expanduser()) if args.run_dir else None,
        task_path=str(Path(args.task_path).expanduser()) if args.task_path else None,
        latest=bool(args.latest),
    )
    plain = bool(args.plain)
    log_lines = max(0, int(args.log_lines))
    peer_limit = max(1, int(args.peer_limit))
    resolved_target = resolve_monitor_target(target)
    if resolved_target is None:
        return 2
    interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
    once = bool(args.once or (not interactive and not args.follow))
    if not interactive and not args.once and not args.follow:
        sys.stderr.write(
            "praxist --monitor: non-interactive stdout detected; rendering once "
            "(pass --follow to stream)\n"
        )
    clear = interactive and not bool(args.no_clear or plain)
    fullscreen = _should_use_tui(once=once, clear=clear, plain=plain)
    interval = float(
        args.interval
        if args.interval is not None
        else (DEFAULT_INTERVAL_SECONDS if fullscreen else DEFAULT_PLAIN_INTERVAL_SECONDS)
    )
    if not math.isfinite(interval) or interval <= 0:
        sys.stderr.write("praxist --monitor: --interval must be a finite positive number\n")
        return 2
    interval = max(MIN_INTERVAL_SECONDS, interval)
    return run_foreground_monitor(
        target=resolved_target,
        interval=interval,
        once=once,
        clear=fullscreen,
        plain=plain,
        log_lines=log_lines,
        peer_limit=peer_limit,
    )


def resolve_monitor_target(target: MonitorTarget) -> MonitorTarget | None:
    """Resolve target ambiguity once before launching the display."""

    rows = _status_rows_for_target(target, include_peer_health=False)
    selected, warnings = select_status_row(rows, target)
    if selected is None and warnings:
        for warning in warnings:
            sys.stderr.write(f"praxist --monitor: {_terminal_safe_text(warning)}\n")
        sys.stderr.write(_candidate_rows_message(rows))
        return None
    if selected is None:
        return target
    if target.run_id or target.run_dir:
        return target
    if selected.run_id:
        return MonitorTarget(run_id=selected.run_id)
    if selected.run_dir:
        return MonitorTarget(run_dir=selected.run_dir)
    sys.stderr.write(
        "praxist --monitor: selected row has no stable run_id or run_dir; pass an explicit "
        "--run-id or --run-dir\n"
    )
    sys.stderr.write(_candidate_rows_message(rows))
    return None


class _MonitorSnapshotSampler:
    """Refresh immutable monitor data independently from the terminal frame rate."""

    def __init__(
        self,
        *,
        initial: MonitorSnapshot,
        collector: Callable[[], MonitorSnapshot],
        interval: float,
    ) -> None:
        self._latest = initial
        self._collector = collector
        self._interval = max(DEFAULT_SAMPLE_INTERVAL_SECONDS, float(interval))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="praxist-monitor-sampler",
            daemon=True,
        )
        self._thread.start()

    def latest(self) -> MonitorSnapshot:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=SAMPLER_SHUTDOWN_TIMEOUT_SECONDS)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                snapshot = self._collector()
            except Exception as exc:
                with self._lock:
                    retained = [
                        warning
                        for warning in self._latest.warnings
                        if not warning.startswith("monitor sampling failed:")
                    ]
                    self._latest = replace(
                        self._latest,
                        warnings=[
                            *retained,
                            f"monitor sampling failed: {type(exc).__name__}: {exc}",
                        ],
                    )
                continue
            with self._lock:
                self._latest = snapshot


def run_foreground_monitor(
    *,
    target: MonitorTarget,
    interval: float,
    once: bool,
    clear: bool,
    plain: bool,
    log_lines: int,
    peer_limit: int,
) -> int:
    """Render monitor frames in the current terminal."""

    use_tui = _should_use_tui(once=once, clear=clear, plain=plain)
    if not use_tui:
        clear = False
    tui_renderer = (
        TuiMonitorRenderer(
            peer_limit=peer_limit,
            unicode_glyphs=_stream_supports_tui_glyphs(sys.stdout),
        )
        if use_tui
        else None
    )
    text_renderer = TextMonitorRenderer(peer_limit=peer_limit)
    exit_signal: int | None = None
    previous_handlers: dict[int, Any] = {}
    terminal_active = False
    sampler: _MonitorSnapshotSampler | None = None
    previous_size: os.terminal_size | None = None
    frame_index = 0

    def _enter_tui() -> None:
        nonlocal terminal_active
        if not use_tui or terminal_active:
            return
        terminal_active = True
        sys.stdout.write("\033[?1049h\033[?25l\033[?7l\033[r\033[H\033[2J")
        sys.stdout.flush()

    def _leave_tui() -> None:
        nonlocal terminal_active
        if not terminal_active:
            return
        try:
            sys.stdout.write("\033[r\033[?7h\033[?25h\033[?1049l")
            sys.stdout.flush()
        finally:
            terminal_active = False

    def _request_clean_exit(signum: int, _frame: object) -> None:
        nonlocal exit_signal
        exit_signal = signum
        raise KeyboardInterrupt

    def _suspend_monitor(signum: int, _frame: object) -> None:
        _leave_tui()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        signal.signal(signum, _suspend_monitor)
        _enter_tui()

    try:
        if use_tui:
            try:
                exit_signals = (
                    getattr(signal, name)
                    for name in ("SIGHUP", "SIGTERM", "SIGQUIT")
                    if hasattr(signal, name)
                )
                for signum in exit_signals:
                    previous_handler = signal.getsignal(signum)
                    signal.signal(signum, _request_clean_exit)
                    previous_handlers[signum] = previous_handler
                if hasattr(signal, "SIGTSTP"):
                    signum = signal.SIGTSTP
                    previous_handler = signal.getsignal(signum)
                    signal.signal(signum, _suspend_monitor)
                    previous_handlers[signum] = previous_handler
            except ValueError:
                # Embedded callers may render from a non-main thread, where
                # Python does not permit signal-handler changes.
                for signum, handler in previous_handlers.items():
                    with contextlib.suppress(ValueError):
                        signal.signal(signum, handler)
                previous_handlers.clear()
                use_tui = False
                tui_renderer = None
                clear = False
                interval = max(DEFAULT_PLAIN_INTERVAL_SECONDS, interval)
            _enter_tui()
        snapshot = collect_monitor_snapshot(
            target=target,
            log_lines=log_lines,
            scan_peer_result_artifacts=once,
        )
        if not once:
            sampler = _MonitorSnapshotSampler(
                initial=snapshot,
                collector=lambda: collect_monitor_snapshot(target=target, log_lines=log_lines),
                interval=max(DEFAULT_SAMPLE_INTERVAL_SECONDS, interval),
            )
            sampler.start()
        next_frame_at = time.monotonic()
        while True:
            size: os.terminal_size | None = None
            if sampler is not None:
                snapshot = sampler.latest()
            if tui_renderer is not None:
                size = shutil.get_terminal_size(fallback=(120, 36))
                frame = tui_renderer.render(
                    snapshot,
                    width=size.columns,
                    height=size.lines,
                    frame_index=frame_index,
                )
                sys.stdout.write("\033[r\033[?7l\033[H")
                if size != previous_size:
                    sys.stdout.write("\033[2J")
                previous_size = size
            else:
                frame = text_renderer.render(snapshot)
            if clear and not use_tui:
                sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(_encode_for_stream(frame, sys.stdout))
            if tui_renderer is not None:
                assert size is not None
                start, end = tui_renderer.stream_scroll_region(
                    width=size.columns,
                    height=size.lines,
                )
                sys.stdout.write(f"\033[{start};{end}r\033[{end};1H")
            sys.stdout.flush()
            if once:
                return 0
            frame_index += 1
            frame_interval = max(MIN_INTERVAL_SECONDS, interval)
            next_frame_at += frame_interval
            now = time.monotonic()
            if next_frame_at <= now:
                next_frame_at = now
                continue
            time.sleep(next_frame_at - now)
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        if not use_tui:
            sys.stderr.write("praxist --monitor: display interrupted\n")
        return 128 + (exit_signal or signal.SIGINT)
    finally:
        _leave_tui()
        try:
            if sampler is not None:
                sampler.close()
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def collect_monitor_snapshot(
    *,
    target: MonitorTarget,
    log_lines: int,
    scan_peer_result_artifacts: bool = False,
) -> MonitorSnapshot:
    """Collect one read-only monitor frame."""

    rows = _status_rows_for_target(
        target,
        scan_peer_result_artifacts=scan_peer_result_artifacts,
    )
    selected, warnings = select_status_row(rows, target)
    orch: dict[str, Any] = {}
    logs: list[str] = []
    phase = "no selected run"
    if selected is not None and selected.run_dir:
        run_dir = Path(selected.run_dir)
        if not run_dir.is_dir():
            warnings.append(
                f"selected run directory is missing: {run_dir}; registry state cannot be verified"
            )
            phase = "missing-run-directory"
        else:
            orch = read_effective_orchestrator_status(run_dir)
            phase = infer_run_phase(run_dir=run_dir, row=selected, orchestrator_status=orch)
            logs = tail_recent_logs(run_dir, max_lines=log_lines)
    return MonitorSnapshot(
        rows=rows,
        selected=selected,
        target=target,
        generated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        orchestrator_status=orch,
        phase=phase,
        recent_logs=logs,
        hardware=collect_hardware_snapshot(probe_timeout=1.0),
        warnings=warnings,
    )


def _status_rows_for_target(
    target: MonitorTarget,
    *,
    include_peer_health: bool = True,
    scan_peer_result_artifacts: bool = False,
) -> list[status.StatusRow]:
    """Include an explicit offline run directory without registering it."""
    rows = status.collect_status_rows(
        include_peer_health=False,
        process_probe_timeout=1.0,
    )
    if target.run_dir:
        wanted = _norm_path_string(target.run_dir)
        if not any(row.run_dir and _norm_path_string(row.run_dir) == wanted for row in rows):
            run_dir = Path(target.run_dir).expanduser().resolve()
            if run_dir.is_dir():
                run_payload = _read_json_object(run_dir / "run.json")
                summary = _read_json_object(run_dir / "run_summary.json")
                startup = _read_json_object(run_dir / "startup_config.json")
                canonical = startup.get("canonical_args")
                canonical = canonical if isinstance(canonical, dict) else {}
                task_project = startup.get("task_project")
                task_project = task_project if isinstance(task_project, dict) else {}
                state = str(summary.get("status") or run_payload.get("status") or "offline")
                rows.append(
                    status.StatusRow(
                        pid=0,
                        ppid=0,
                        etime="-",
                        command="-",
                        run_dir=str(run_dir),
                        source="offline",
                        state=state,
                        run_id=str(run_payload.get("run_id") or run_dir.name),
                        task_path=(
                            str(canonical.get("task_path") or task_project.get("path") or "")
                            or None
                        ),
                        model=str(canonical.get("model") or "") or None,
                        model_provider_ref=str(canonical.get("model_provider") or "") or None,
                        started_at=str(run_payload.get("started_at") or "") or None,
                    )
                )
    if not include_peer_health:
        return rows
    selected, _warnings = select_status_row(rows, target)
    if selected is None or not selected.run_dir:
        return rows
    peer_health = status._read_peer_health(
        selected.run_dir,
        selected.task_path,
        selected.generation,
        scan_result_artifacts=scan_peer_result_artifacts,
    )
    enriched = replace(
        selected,
        peer_health_summary=(
            peer_health.summary
            if peer_health.peers or not selected.peers
            else selected.peer_health_summary
        ),
        peers=(
            [peer.to_dict() for peer in peer_health.peers]
            if peer_health.peers or not selected.peers
            else selected.peers
        ),
    )
    return [enriched if row is selected else row for row in rows]


def select_status_row(
    rows: Sequence[status.StatusRow],
    target: MonitorTarget,
) -> tuple[status.StatusRow | None, list[str]]:
    """Select a run row for detailed rendering."""

    warnings: list[str] = []
    if target.run_id:
        matches = [row for row in rows if row.run_id == target.run_id]
        if matches:
            return matches[0], warnings
        return None, [f"run id not found: {target.run_id}"]
    if target.run_dir:
        wanted = _norm_path_string(target.run_dir)
        matches = [row for row in rows if row.run_dir and _norm_path_string(row.run_dir) == wanted]
        if matches:
            return matches[0], warnings
        return None, [f"run dir not found in status rows: {target.run_dir}"]
    if target.task_path:
        wanted_task = _norm_path_string(target.task_path)
        matches = [
            row
            for row in rows
            if (row.task_path and _norm_path_string(row.task_path) == wanted_task)
            or (row.run_dir and _norm_path_string(row.run_dir).startswith(wanted_task + os.sep))
        ]
        active = [row for row in matches if row.state == STATE_RUNNING]
        pool = active or matches
        if len(pool) == 1:
            return pool[0], warnings
        if pool:
            if target.latest:
                return _latest_row(pool), warnings
            return None, ["multiple rows match task path; pass --latest, --run-id, or --run-dir"]
        return None, [f"no rows match task path: {target.task_path}"]
    active_rows = [row for row in rows if row.state == STATE_RUNNING]
    if target.latest:
        return (_latest_row(active_rows or rows), warnings) if rows else (None, warnings)
    if len(active_rows) == 1:
        return active_rows[0], warnings
    if len(rows) == 1:
        return rows[0], warnings
    if active_rows:
        return None, ["multiple active rows detected; pass --latest, --run-id, or --run-dir"]
    if rows:
        return None, ["multiple non-running rows detected; pass --latest, --run-id, or --run-dir"]
    return None, warnings


def infer_run_phase(
    *,
    run_dir: Path,
    row: status.StatusRow,
    orchestrator_status: dict[str, Any],
) -> str:
    """Best-effort human phase label from small run artifacts."""

    exit_condition = str(orchestrator_status.get("exit_condition") or "").strip()
    if exit_condition and exit_condition != "in_progress":
        return f"finished:{exit_condition}"
    current_gen = _safe_int(orchestrator_status.get("current_generation"), row.generation)
    if current_gen is None:
        return "starting/no-orchestrator-status"
    gen_dir = _generation_dir(run_dir, current_gen)
    if gen_dir is None:
        return f"gen{current_gen}:initializing"
    if (gen_dir / "generation_boundary.json").exists():
        return f"gen{current_gen}:boundary-committed"
    if (gen_dir / "generation_results.json").exists():
        return f"gen{current_gen}:boundary-pending"
    if (gen_dir / "STOP_SIGNAL").exists() or (gen_dir / "CLOSING_SIGNAL").exists():
        return f"gen{current_gen}:closing"
    dig_status = gen_dir / "dig" / "dig_stage_status.json"
    dig_payload = _read_json_object(dig_status)
    if dig_payload:
        last_phase = str(dig_payload.get("last_phase") or "").strip()
        last_status = str(dig_payload.get("last_status") or "").strip()
        if last_phase or last_status:
            return f"gen{current_gen}:dig:{last_phase or '?'}:{last_status or '?'}"
    return f"gen{current_gen}:running"


def tail_recent_logs(run_dir: Path, *, max_lines: int) -> list[str]:
    """Return a compact tail from the newest small log files under ``run_dir/logs``."""

    if max_lines <= 0:
        return []
    logs_dir = run_dir / "logs"
    try:
        candidates = [
            path for path in logs_dir.glob("*.log") if path.is_file() and not path.is_symlink()
        ]
    except OSError:
        return []
    if not candidates:
        return []
    candidates.sort(key=lambda path: _mtime(path), reverse=True)
    selected = candidates[:2]
    per_file = max(1, max_lines // len(selected))
    lines: list[str] = []
    for path in selected:
        tail = _tail_file(path, per_file)
        if not tail:
            continue
        lines.append(f"[{path.name}]")
        lines.extend(f"  {line}" for line in tail)
    return lines[-max_lines - len(selected) :]


def collect_hardware_snapshot(*, probe_timeout: float = 3.0) -> HardwareSnapshot:
    """Collect lightweight host load without broad scans."""

    warnings: list[str] = []
    loadavg = "-"
    with _suppress_os_error():
        loadavg = " ".join(f"{value:.2f}" for value in os.getloadavg())
    memory = _read_meminfo()
    gpus = _read_nvidia_smi(warnings, timeout_seconds=probe_timeout)
    return HardwareSnapshot(loadavg=loadavg, memory=memory, gpus=gpus, warnings=warnings)


def _latest_row(rows: Sequence[status.StatusRow]) -> status.StatusRow:
    return sorted(rows, key=_row_time_key, reverse=True)[0]


def _row_time_key(row: status.StatusRow) -> tuple[str, str, int]:
    return (row.updated_at or "", row.started_at or "", row.pid)


def _candidate_rows_message(rows: Sequence[status.StatusRow]) -> str:
    if not rows:
        return ""
    lines = ["candidate runs:\n"]
    for row in rows[:10]:
        line = (
            f"- run_id={row.run_id or '-'} pid={row.pid} state={row.state} "
            f"task={row.task_path or '-'} run_dir={row.run_dir or '-'}"
        )
        lines.append(_terminal_safe_text(line) + "\n")
    if len(rows) > 10:
        lines.append(f"- ... {len(rows) - 10} more rows\n")
    return "".join(lines)


def _should_use_tui(*, once: bool, clear: bool, plain: bool) -> bool:
    if once or plain or not clear:
        return False
    return os.environ.get("TERM", "").strip().lower() != "dumb"


def _style(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def _truecolor(text: str, rgb: tuple[int, int, int], enabled: bool) -> str:
    if not enabled:
        return text
    red, green, blue = rgb
    return f"\033[38;2;{red};{green};{blue}m{text}\033[0m"


def _pulse_glyph(frame_index: int, *, unicode_glyphs: bool) -> str:
    pulses = ("●", "◉", "○", "◉") if unicode_glyphs else ("*", "o", ".", "o")
    return pulses[frame_index % len(pulses)]


def _peer_health_rgb(health: str) -> tuple[int, int, int]:
    return {
        "green": LIVE_RGB,
        "yellow": WARNING_RGB,
        "red": ERROR_RGB,
    }.get(str(health).strip().casefold(), UNKNOWN_RGB)


def _render_brand_mark(
    pixel_rows: Sequence[str],
    *,
    color: bool,
    unicode_glyphs: bool = True,
) -> list[str]:
    """Render a binary logo mask with half blocks in the canonical brand color."""

    lines: list[str] = []
    for index in range(0, len(pixel_rows), 2):
        upper = pixel_rows[index]
        lower = pixel_rows[index + 1] if index + 1 < len(pixel_rows) else "0" * len(upper)
        glyphs: list[str] = []
        for top, bottom in zip(upper, lower, strict=True):
            if unicode_glyphs:
                glyph = (
                    "█"
                    if top == "1" and bottom == "1"
                    else "▀"
                    if top == "1"
                    else "▄"
                    if bottom == "1"
                    else " "
                )
            elif top == "1" and bottom == "1":
                glyph = "#"
            elif top == "1":
                glyph = "^"
            elif bottom == "1":
                glyph = "_"
            else:
                glyph = " "
            glyphs.append(glyph)
        lines.append(_truecolor("".join(glyphs), BRAND_RGB, color))
    return lines


def _stream_supports_tui_glyphs(stream: Any) -> bool:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "█▀▄◆●◉○".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _encode_for_stream(value: str, stream: Any) -> str:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        value.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        try:
            return value.encode(encoding, errors="replace").decode(encoding)
        except LookupError:
            return value.encode("ascii", errors="replace").decode("ascii")
    return value


def _strip_ansi(value: str) -> str:
    return ANSI_STYLE_RE.sub("", value)


def _terminal_safe_text(value: str) -> str:
    """Remove terminal controls while preserving ordinary Unicode text."""

    output: list[str] = []
    for char in value:
        codepoint = ord(char)
        if char in "\t\r\n":
            output.append(" ")
        elif (
            codepoint < 0x20
            or 0x7F <= codepoint <= 0x9F
            or codepoint in {0x061C, 0x200E, 0x200F}
            or 0x202A <= codepoint <= 0x202E
            or 0x2066 <= codepoint <= 0x2069
        ):
            continue
        else:
            output.append(char)
    return "".join(output)


def _visible_len(value: str) -> int:
    return _cell_width(_strip_ansi(value))


def _fit_plain(value: str, width: int) -> str:
    width = max(0, int(width))
    if width == 0:
        return ""
    value = _terminal_safe_text(value)
    if _cell_width(value) <= width:
        return value
    if width <= 1:
        return _truncate_cells(value, width)
    return _truncate_cells(value, width - 1) + "~"


def _wrap_plain(value: str, width: int) -> list[str]:
    """Soft-wrap terminal-safe text without splitting wide characters."""

    width = max(1, int(width))
    remaining = _terminal_safe_text(value)
    if not remaining:
        return [""]
    lines: list[str] = []
    while remaining:
        line = _truncate_cells(remaining, width)
        if not line:
            # A printable cell can be wider than a deliberately tiny viewport.
            line, remaining = remaining[0], remaining[1:]
        else:
            remaining = remaining[len(line) :]
        lines.append(line)
    return lines


def _fit_ansi(value: str, width: int) -> str:
    plain_width = _visible_len(value)
    if plain_width > width:
        return _fit_plain(_strip_ansi(value), width)
    return value + " " * max(0, width - plain_width)


def _fit_generated_ansi(value: str, width: int) -> str:
    """Fit trusted SGR-styled text without dropping a leading status color."""

    width = max(0, int(width))
    if width == 0:
        return ""
    visible = _visible_len(value)
    if visible <= width:
        return value + " " * (width - visible)

    target = max(0, width - 1)
    output: list[str] = []
    used = 0
    index = 0
    saw_style = False
    while index < len(value):
        match = ANSI_STYLE_RE.match(value, index)
        if match is not None:
            output.append(match.group(0))
            saw_style = True
            index = match.end()
            continue
        char = _terminal_safe_text(value[index])
        index += 1
        if not char:
            continue
        char_width = _cell_width(char)
        if used + char_width > target:
            break
        output.append(char)
        used += char_width
    output.append("~")
    if saw_style:
        output.append("\033[0m")
    return "".join(output) + " " * max(0, width - used - 1)


def _join_left_right(left: str, right: str, width: int) -> str:
    left = _fit_plain(left, width)
    right = _fit_plain(right, width)
    gap = width - _cell_width(left) - _cell_width(right)
    if gap <= 0:
        return _fit_plain(left + " " + right, width)
    return left + " " * gap + right


def _columns(specs: Sequence[tuple[str, int, str]]) -> str:
    cells: list[str] = []
    for value, width, align in specs:
        clipped = _fit_plain(str(value), width)
        pad = " " * max(0, width - _cell_width(clipped))
        if align == "right":
            cells.append(pad + clipped)
        else:
            cells.append(clipped + pad)
    return " ".join(cells)


def _pad_cells(value: str, width: int) -> str:
    return value + " " * max(0, width - _cell_width(value))


def _cell_width(value: str) -> int:
    width = 0
    for char in value:
        if char in "\n\r":
            continue
        category = unicodedata.category(char)
        if category.startswith("C") or unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def _truncate_cells(value: str, width: int) -> str:
    if width <= 0:
        return ""
    used = 0
    output: list[str] = []
    for char in value:
        char_width = _cell_width(char)
        if char_width == 0:
            output.append(char)
            continue
        if used + char_width > width:
            break
        output.append(char)
        used += char_width
    return "".join(output)


def _hstack(left: list[str], right: list[str], *, gap: int) -> list[str]:
    left_width = max((_visible_len(line) for line in left), default=0)
    right_width = max((_visible_len(line) for line in right), default=0)
    height = max(len(left), len(right))
    lines: list[str] = []
    for idx in range(height):
        left_line = left[idx] if idx < len(left) else ""
        right_line = right[idx] if idx < len(right) else ""
        lines.append(
            _fit_ansi(left_line, left_width) + " " * gap + _fit_ansi(right_line, right_width)
        )
    return lines


def _norm_path_string(path: str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        st = path.lstat()
        if st.st_size > MAX_JSON_BYTES or not path.is_file() or path.is_symlink():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _generation_dir(run_dir: Path, generation: int) -> Path | None:
    for name in (f"gen_{generation}", f"gen{generation}"):
        path = run_dir / name
        try:
            if path.is_dir() and not path.is_symlink():
                return path
        except OSError:
            continue
    return None


def _tail_file(path: Path, max_lines: int) -> list[str]:
    try:
        st = path.stat()
        offset = max(0, st.st_size - MAX_LOG_BYTES)
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(MAX_LOG_BYTES)
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines[-max_lines:]


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _read_meminfo() -> str:
    try:
        data: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                data[parts[0].rstrip(":")] = int(parts[1])
        total = data.get("MemTotal")
        available = data.get("MemAvailable")
        if total and available is not None:
            used = total - available
            return f"{_format_kib(used)} / {_format_kib(total)}"
    except (OSError, ValueError):
        pass
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available = int(os.sysconf("SC_AVPHYS_PAGES"))
        if pages > 0 and page_size > 0 and available >= 0:
            total_kib = pages * page_size // 1024
            used_kib = (pages - available) * page_size // 1024
            return f"{_format_kib(used_kib)} / {_format_kib(total_kib)}"
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return "-"


def _read_nvidia_smi(
    warnings: list[str],
    *,
    timeout_seconds: float = 3.0,
) -> list[str]:
    nvidia = shutil.which("nvidia-smi")
    if nvidia is None:
        return []
    try:
        result = subprocess.run(
            [
                nvidia,
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.1, float(timeout_seconds)),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        warnings.append(f"nvidia-smi unavailable: {exc}")
        return []
    if result.returncode != 0:
        warnings.append("nvidia-smi returned non-zero status")
        return []
    gpus: list[str] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            gpus.append(f"GPU {parts[0]}: util {parts[1]}%, mem {parts[2]}/{parts[3]} MiB")
    return gpus


def _format_kib(value: int) -> str:
    mib = value / 1024.0
    if mib >= 1024:
        return f"{mib / 1024.0:.1f} GiB"
    return f"{mib:.0f} MiB"


def _scoreless_limit_lines(orch: dict[str, Any]) -> list[str]:
    if orch.get("research_mode") != "scoreless":
        return []
    return [
        f"{label}: {'unbounded' if orch[key] is None else orch[key]}"
        for key, label in (
            ("max_generations", "generation_limit"),
            ("per_generation_hours", "per_generation_hours"),
        )
        if key in orch
    ]


def _format_peer_health(summary: dict[str, int] | None) -> str:
    if not summary:
        return "-"
    red = int(summary.get("red", 0) or 0)
    yellow = int(summary.get("yellow", 0) or 0)
    green = int(summary.get("green", 0) or 0)
    if red + yellow + green == 0:
        return "-"
    return f"R{red}/Y{yellow}/G{green}"


def _format_float(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "-"
    return f"{number:.4g}"


def _result_summary(value: object, *, include_reason: bool = False) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    variant = str(value.get("variant_name") or "unknown")
    metric = str(value.get("metric_name") or "metric")
    score = _format_float(value.get("metric_value"))
    stage = str(value.get("evidence_stage") or "unknown-stage")
    relation = str(value.get("baseline_relation") or "")
    summary = f"{variant} | {metric}={score} | stage={stage}"
    if relation:
        summary += f" | {relation}"
    if include_reason:
        reason = str(value.get("validation_reason") or "").strip()
        if reason:
            summary += f" | needs validation: {reason}"
    return summary


def _scheduler_blocker_summary(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    reasons = value.get("queue_blocked_reasons")
    if isinstance(reasons, dict) and reasons:
        return ", ".join(f"{key}={count}" for key, count in sorted(reasons.items()))
    probe = value.get("accelerator_probe")
    if isinstance(probe, dict) and probe.get("state") not in {None, "", "available"}:
        state = str(probe.get("state"))
        reason = str(probe.get("reason") or "").strip()
        return f"accelerator {state}" + (f": {reason}" if reason else "")
    return ""


def _safe_int(value: object, fallback: int | None) -> int | None:
    if isinstance(value, int):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _display(value: object, fallback: object) -> str:
    if value is None or value == "":
        value = fallback
    if value is None or value == "":
        return "-"
    return str(value)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


class _suppress_os_error:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, _exc: object, _tb: object) -> bool:
        return isinstance(exc_type, type) and issubclass(exc_type, OSError)


if __name__ == "__main__":  # pragma: no cover - module execution path.
    parser = argparse.ArgumentParser(prog="python -m praxist.cli.monitor")
    subparsers = parser.add_subparsers(dest="command")
    register(subparsers)
    parsed = parser.parse_args(["monitor", *sys.argv[1:]])
    sys.exit(cmd_monitor(parsed))

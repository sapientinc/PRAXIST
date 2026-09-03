"""Tests for the dependency-free Praxist OOBE terminal controls."""

from __future__ import annotations

import io
import os
import pty
import unittest
from unittest.mock import patch

from praxist.cli import _terminal_ui
from praxist.cli._terminal_ui import (
    Choice,
    TerminalInteractionCancelled,
    TerminalInteractionError,
    _read_terminal_key,
    _terminal_key_reader,
    confirm_action,
    interactive_terminal_available,
    read_masked_secret,
    read_visible_text,
    select_choice,
    view_scrollable_text,
)


def _reader(*keys: str):
    iterator = iter(keys)
    return lambda: next(iterator)


class TerminalUiTest(unittest.TestCase):
    class TtyBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    def test_choice_rejects_invalid_configuration_and_interrupt(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            select_choice("Choose", (), read_key=_reader("\n"))
        with self.assertRaisesRegex(ValueError, "out of range"):
            select_choice("Choose", (Choice("a", "A"),), default=1, read_key=_reader("\n"))
        with self.assertRaises(TerminalInteractionCancelled):
            select_choice(
                "Choose",
                (Choice("a", "A"),),
                output_stream=io.StringIO(),
                read_key=_reader("INTERRUPT"),
            )

    def test_choice_accepts_number_and_ignores_out_of_range_number(self) -> None:
        choices = (Choice("a", "A"), Choice("b", "B"))
        self.assertEqual(
            select_choice(
                "Choose",
                choices,
                default=None,
                output_stream=io.StringIO(),
                read_key=_reader("9", "2", "\n"),
            ),
            "b",
        )

    def test_choice_navigation_and_escape(self) -> None:
        output = io.StringIO()
        selected = select_choice(
            "Choose",
            (Choice("a", "A"), Choice("b", "B"), Choice("c", "C")),
            output_stream=output,
            read_key=_reader("DOWN", "DOWN", "UP", "\n"),
        )
        self.assertEqual(selected, "b")
        self.assertIn("2/3", output.getvalue())
        with self.assertRaises(TerminalInteractionCancelled):
            select_choice(
                "Choose",
                (Choice("a", "A"),),
                output_stream=io.StringIO(),
                read_key=_reader("ESC"),
            )

    def test_tty_choice_uses_polished_focus_and_stacked_details(self) -> None:
        output = self.TtyBuffer()
        with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("CLICOLOR", None)
            selected = select_choice(
                "Choose a Praxist runtime profile",
                (
                    Choice("native", "Codex-native mode", "Use the saved Codex login"),
                    Choice("api", "DeepSeek API", "Cost-efficient long research"),
                ),
                output_stream=output,
                read_key=_reader("DOWN", "\n"),
            )

        rendered = output.getvalue()
        self.assertEqual(selected, "api")
        self.assertIn("\x1b[1;36m  > 2. DeepSeek API\x1b[0m", rendered)
        self.assertIn("\x1b[2;36m       Cost-efficient long research\x1b[0m", rendered)
        self.assertIn("\x1b[2m       Use the saved Codex login\x1b[0m", rendered)
        self.assertIn("\x1b[?25l", rendered)
        self.assertTrue(rendered.endswith("\x1b[?25h"))

    def test_tty_choice_respects_color_opt_out_and_bounds_long_text(self) -> None:
        plain_output = self.TtyBuffer()
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertEqual(
                select_choice(
                    "Choose",
                    (Choice("a", "A", "Explanation"),),
                    output_stream=plain_output,
                    read_key=_reader("\n"),
                ),
                "a",
            )
        self.assertNotIn("\x1b[1;36m", plain_output.getvalue())

        self.assertEqual(_terminal_ui._fit_terminal_text("abcdefgh", 6), "abc...")
        self.assertEqual(_terminal_ui._fit_terminal_text("abc", 2), "ab")

        narrow_output = self.TtyBuffer()
        with (
            patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False),
            patch.object(_terminal_ui, "_terminal_width", return_value=19),
        ):
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("CLICOLOR", None)
            select_choice(
                "Choose",
                (Choice("a", "A", "Explanation"),),
                output_stream=narrow_output,
                read_key=_reader("\n"),
            )
        self.assertNotIn("\x1b[?25l", narrow_output.getvalue())

    def test_tty_choice_restores_cursor_when_cancelled(self) -> None:
        output = self.TtyBuffer()
        with (
            patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False),
            self.assertRaises(TerminalInteractionCancelled),
        ):
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("CLICOLOR", None)
            select_choice(
                "Choose",
                (Choice("a", "A", "Explanation"),),
                output_stream=output,
                read_key=_reader("ESC"),
            )
        self.assertTrue(output.getvalue().endswith("\x1b[?25h"))

    def test_tty_confirmation_and_text_inputs_style_prompts(self) -> None:
        with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("CLICOLOR", None)

            confirmation = self.TtyBuffer()
            confirm_action("Continue?", output_stream=confirmation, read_key=_reader("\n"))
            self.assertIn("\x1b[1mContinue?\x1b[0m", confirmation.getvalue())
            self.assertIn("\x1b[2m[Enter: continue | Esc: back]\x1b[0m", confirmation.getvalue())

            secret_output = self.TtyBuffer()
            self.assertEqual(
                read_masked_secret(
                    "API key: ",
                    output_stream=secret_output,
                    read_key=_reader("x", "\n"),
                ),
                "x",
            )
            self.assertTrue(secret_output.getvalue().startswith("\x1b[1mAPI key: \x1b[0m"))

            text_output = self.TtyBuffer()
            self.assertEqual(
                read_visible_text(
                    "Project: ",
                    output_stream=text_output,
                    read_key=_reader("x", "\n"),
                ),
                "x",
            )
            self.assertTrue(text_output.getvalue().startswith("\x1b[1mProject: \x1b[0m"))

    def test_scrollable_view_uses_temporary_screen_and_navigation(self) -> None:
        output = self.TtyBuffer()
        view_scrollable_text(
            "Agreement",
            "\n".join(f"line {index}" for index in range(30)),
            output_stream=output,
            read_key=_reader("PAGE_DOWN", "END", "HOME", "q"),
            viewport_height=10,
        )
        rendered = output.getvalue()
        self.assertTrue(rendered.startswith("\x1b[?1049h\x1b[?25l"))
        self.assertIn("Lines 25-30 of 30", rendered)
        self.assertIn("Lines 1-6 of 30", rendered)
        self.assertTrue(rendered.endswith("\x1b[?25h\x1b[?1049l"))

    def test_scrollable_view_interrupt_restores_screen(self) -> None:
        output = self.TtyBuffer()
        with self.assertRaises(TerminalInteractionCancelled):
            view_scrollable_text(
                "Agreement",
                "text",
                output_stream=output,
                read_key=_reader("INTERRUPT"),
                viewport_height=8,
            )
        self.assertTrue(output.getvalue().endswith("\x1b[?25h\x1b[?1049l"))

    def test_scrollable_view_handles_incremental_navigation_and_unknown_keys(self) -> None:
        output = self.TtyBuffer()
        view_scrollable_text(
            "Agreement",
            "\n".join(f"line {index}" for index in range(30)),
            output_stream=output,
            read_key=_reader("UP", "DOWN", "PAGE_UP", "unknown", "q"),
            viewport_height=10,
        )
        self.assertIn("Lines 2-7 of 30", output.getvalue())

    def test_scrollable_view_rejects_unsupported_terminal_shapes(self) -> None:
        with (
            patch.object(
                _terminal_ui,
                "interactive_terminal_available",
                return_value=False,
            ),
            self.assertRaisesRegex(TerminalInteractionError, "interactive terminal"),
        ):
            view_scrollable_text("Agreement", "text", output_stream=io.StringIO())

        with (
            patch.dict(os.environ, {"TERM": "dumb"}),
            patch.object(
                _terminal_ui,
                "interactive_terminal_available",
                return_value=True,
            ),
            self.assertRaisesRegex(TerminalInteractionError, "does not support"),
        ):
            view_scrollable_text("Agreement", "text", output_stream=io.StringIO())

        with self.assertRaisesRegex(TerminalInteractionError, "too short"):
            view_scrollable_text(
                "Agreement",
                "text",
                output_stream=io.StringIO(),
                read_key=_reader("q"),
                viewport_height=7,
            )

    def test_unselected_choice_requires_navigation_before_enter(self) -> None:
        output = io.StringIO()
        selected = select_choice(
            "Consent",
            (Choice("Yes", "Share"), Choice("No", "Skip")),
            default=None,
            output_stream=output,
            read_key=_reader("\n", "DOWN", "\n"),
        )
        self.assertEqual(selected, "Yes")
        self.assertIn("No selection", output.getvalue())

    def test_masked_secret_supports_paste_backspace_and_hides_value(self) -> None:
        output = io.StringIO()
        secret = read_masked_secret(
            "Key: ",
            output_stream=output,
            read_key=_reader("a", "b", "c", "\x7f", "d", "\n"),
        )
        self.assertEqual(secret, "abd")
        rendered = output.getvalue()
        self.assertNotIn("abd", rendered)
        self.assertEqual(rendered.count("*"), 4)

    def test_secret_ignores_arrows_and_escape_cancels(self) -> None:
        secret = read_masked_secret(
            "Key: ",
            output_stream=io.StringIO(),
            read_key=_reader("a", "LEFT", "b", "\r"),
        )
        self.assertEqual(secret, "ab")
        with self.assertRaises(TerminalInteractionCancelled):
            read_masked_secret(
                "Key: ",
                output_stream=io.StringIO(),
                read_key=_reader("x", "ESC"),
            )

    def test_secret_overflow_drains_input_before_failing(self) -> None:
        keys = iter(("a", "b", "c", "\n"))
        with self.assertRaisesRegex(TerminalInteractionError, "safety limit"):
            read_masked_secret(
                "Key: ",
                max_length=2,
                output_stream=io.StringIO(),
                read_key=lambda: next(keys),
            )
        with self.assertRaises(StopIteration):
            next(keys)

    def test_secret_validates_limit_and_ignores_unsupported_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            read_masked_secret("Key: ", max_length=0, read_key=_reader("\n"))
        self.assertEqual(
            read_masked_secret(
                "Key: ",
                output_stream=io.StringIO(),
                read_key=_reader("", "ab", "\x01", "x", "\n"),
            ),
            "x",
        )
        with self.assertRaises(TerminalInteractionCancelled):
            read_masked_secret(
                "Key: ",
                output_stream=io.StringIO(),
                read_key=_reader("INTERRUPT"),
            )

    def test_visible_text_supports_backspace_and_escape(self) -> None:
        output = io.StringIO()
        value = read_visible_text(
            "Path: ",
            output_stream=output,
            read_key=_reader("/", "a", "x", "\x7f", "b", "\n"),
        )
        self.assertEqual(value, "/ab")
        self.assertIn("/ax", output.getvalue())
        with self.assertRaises(TerminalInteractionCancelled):
            read_visible_text(
                "Path: ",
                output_stream=io.StringIO(),
                read_key=_reader("/", "x", "ESC"),
            )

    def test_visible_text_validates_limit_and_bounds_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            read_visible_text("Path: ", max_length=0, read_key=_reader("\n"))
        self.assertEqual(
            read_visible_text(
                "Path: ",
                max_length=2,
                output_stream=io.StringIO(),
                read_key=_reader("LEFT", "", "\x01", "a", "b", "c", "\n"),
            ),
            "ab",
        )
        with self.assertRaises(TerminalInteractionCancelled):
            read_visible_text(
                "Path: ",
                output_stream=io.StringIO(),
                read_key=_reader("INTERRUPT"),
            )

    def test_public_controls_use_terminal_reader_when_not_injected(self) -> None:
        module = "praxist.cli._terminal_ui"
        choices = (Choice("a", "A"),)
        with (
            patch(f"{module}.interactive_terminal_available", return_value=True),
            patch(f"{module}._terminal_key_reader") as terminal_reader,
        ):
            terminal_reader.return_value.__enter__.side_effect = [
                _reader("\n"),
                _reader("\n"),
                _reader("x", "\n"),
                _reader("y", "\n"),
            ]
            self.assertEqual(select_choice("Choose", choices), "a")
            confirm_action("Continue?")
            self.assertEqual(read_masked_secret("Key: "), "x")
            self.assertEqual(read_visible_text("Path: "), "y")

    def test_public_controls_reject_noninteractive_streams(self) -> None:
        module = "praxist.cli._terminal_ui.interactive_terminal_available"
        with patch(module, return_value=False):
            with self.assertRaises(TerminalInteractionError):
                select_choice("Choose", (Choice("a", "A"),))
            with self.assertRaises(TerminalInteractionError):
                confirm_action("Continue?")
            with self.assertRaises(TerminalInteractionError):
                read_masked_secret("Key: ")
            with self.assertRaises(TerminalInteractionError):
                read_visible_text("Path: ")

    def test_interactive_terminal_detection_requires_both_ttys(self) -> None:
        class Stream(io.StringIO):
            def __init__(self, tty: bool) -> None:
                super().__init__()
                self.tty = tty

            def isatty(self) -> bool:
                return self.tty

        self.assertTrue(interactive_terminal_available(Stream(True), Stream(True)))
        self.assertFalse(interactive_terminal_available(Stream(True), Stream(False)))

    def test_terminal_key_reader_preserves_utf8_path_characters(self) -> None:
        character = chr(0x7814)
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, character.encode())
            self.assertEqual(_read_terminal_key(read_fd), character)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_terminal_key_reader_classifies_control_and_escape_sequences(self) -> None:
        def read(payload: bytes, *, close_writer: bool = False) -> str:
            read_fd, write_fd = os.pipe()
            try:
                if payload:
                    os.write(write_fd, payload)
                if close_writer:
                    os.close(write_fd)
                    write_fd = -1
                return _read_terminal_key(read_fd)
            finally:
                os.close(read_fd)
                if write_fd >= 0:
                    os.close(write_fd)

        self.assertEqual(read(b"", close_writer=True), "EOF")
        self.assertEqual(read(b"\x04"), "EOF")
        self.assertEqual(read(b"\x03"), "INTERRUPT")
        self.assertEqual(read(b"x"), "x")
        self.assertEqual(read(chr(0x1F642).encode()), chr(0x1F642))
        self.assertEqual(read(b"\xe2"), "UNKNOWN")
        self.assertEqual(read(b"\x1b[A"), "UP")
        self.assertEqual(read(b"\x1b[B"), "DOWN")
        self.assertEqual(read(b"\x1b[C"), "RIGHT")
        self.assertEqual(read(b"\x1b[D"), "LEFT")
        self.assertEqual(read(b"\x1b[5~"), "PAGE_UP")
        self.assertEqual(read(b"\x1b[6~"), "PAGE_DOWN")
        self.assertEqual(read(b"\x1b[H"), "HOME")
        self.assertEqual(read(b"\x1b[F"), "END")
        self.assertEqual(read(b"\x1b"), "ESC")
        self.assertEqual(read(b"\x1b[Z"), "UNKNOWN")

    def test_terminal_reader_enters_and_restores_real_pty(self) -> None:
        master, slave = pty.openpty()
        stream = os.fdopen(slave, "r", closefd=False)
        try:
            with _terminal_key_reader(stream) as reader:
                os.write(master, b"x")
                self.assertEqual(reader(), "x")
        finally:
            stream.close()
            os.close(master)
            os.close(slave)

    def test_terminal_reader_reports_stream_without_file_descriptor(self) -> None:
        with (
            self.assertRaisesRegex(TerminalInteractionError, "terminal mode"),
            _terminal_key_reader(io.StringIO()),
        ):
            self.fail("reader should not be yielded")

    def test_confirmation_accepts_enter_and_rejects_escape(self) -> None:
        confirm_action("Continue?", output_stream=io.StringIO(), read_key=_reader("\n"))
        with self.assertRaises(TerminalInteractionCancelled):
            confirm_action("Continue?", output_stream=io.StringIO(), read_key=_reader("ESC"))
        with self.assertRaises(TerminalInteractionCancelled):
            confirm_action("Continue?", output_stream=io.StringIO(), read_key=_reader("INTERRUPT"))

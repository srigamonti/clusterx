# Copyright (c) 2015-2024, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

import io
from contextlib import redirect_stdout

commands = []  # helper module, exposes no CLI commands (required by cli/commands.py)


class MarkdownReport:
    """Collect a Markdown report, optionally echoing it to the console.

    Parameters
    ----------
    echo : bool, default True
        If True, everything added to the report is also printed to stdout.
    """

    def __init__(self, echo: bool = True):
        self.echo = echo
        self._lines: list[str] = []

    def _emit(self, text: str = ""):
        self._lines.append(text)
        if self.echo:
            print(text)

    def title(self, text: str):
        self._emit(f"# {text}\n")

    def section(self, text: str, level: int = 2):
        self._emit(f"\n{'#' * level} {text}\n")

    def text(self, text: str):
        self._emit(f"{text}\n")

    def bullets(self, items):
        for item in items:
            self._emit(f"- {item}")
        self._emit()

    def code(self, text: str, lang: str = ""):
        self._emit(f"```{lang}\n{str(text).rstrip()}\n```\n")

    def table(self, headers, rows):
        self._emit("| " + " | ".join(str(h) for h in headers) + " |")
        self._emit("|" + "|".join("---" for _ in headers) + "|")
        for row in rows:
            self._emit("| " + " | ".join(str(c) for c in row) + " |")
        self._emit()

    def capture(self, func, *args, **kwargs):
        """Run ``func``, add whatever it prints to the report as a code block."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = func(*args, **kwargs)
        out = buf.getvalue()
        if out.strip():
            self.code(out)
        return result

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(self._lines) + "\n")

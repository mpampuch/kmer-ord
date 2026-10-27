import os
import re
import subprocess
from pathlib import Path

from rich.console import Console

console = Console(highlight=False)

# Bound before the instance overrides below. Console.rule renders by calling
# self.print, so the print wrapper must not mirror again while rule() is
# already copying that line onto the file console.
_terminal_print = console.print
_terminal_rule = console.rule
_log_file = None
_log_console: Console | None = None
_inside_rule = False


def _print(*args, **kwargs):
    _terminal_print(*args, **kwargs)
    if _log_console is None or _log_file is None or _inside_rule:
        return
    _log_console.print(*args, **kwargs)
    _log_file.flush()


def _rule(*args, **kwargs):
    global _inside_rule
    _inside_rule = True
    try:
        _terminal_rule(*args, **kwargs)
    finally:
        _inside_rule = False
    if _log_console is None or _log_file is None:
        return
    _log_console.rule(*args, **kwargs)
    _log_file.flush()


console.print = _print
console.rule = _rule


def close_run_log() -> None:
    """Stop mirroring and close the log file. Safe when no log is open."""
    global _log_file, _log_console
    _log_console = None
    if _log_file is not None:
        _log_file.flush()
        _log_file.close()
        _log_file = None


def enable_run_log(path) -> None:
    """Append a plain-text copy of console output to path, flushed each line.

    A second call closes the previous file first. The parent directory is
    created so the banner can be logged before Context builds the output tree.
    """
    global _log_file, _log_console
    close_run_log()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = path.open("a", encoding="utf-8", buffering=1)
    _log_console = Console(
        file=_log_file,
        force_terminal=False,
        highlight=False,
        width=console.width,
    )


def section(title: str) -> None:
    console.print()
    console.rule(title, align="left", style="Dim")

def info(message: str) -> None:
    console.print(f"    {message}", markup=False, highlight=False)

def warn(message: str) -> None:
    console.print(f"    WARNING  {message}", style="Dim", markup=False, highlight=False)

def divider() -> None:
    console.print("    " + "─" * 52, style="Dim")


# tqdm redraws the current line with carriage returns. A pipe delivers those
# updates as one chunk, and writing them would fill the run log with bars.
_PROGRESS_BAR = re.compile(r"^\s*\d+%\|")


def _iter_log_lines(stream):
    for chunk in stream:
        for piece in str(chunk).split("\n"):
            # A trailing \r is the newline half of \r\n, not a redraw.
            piece = piece.rstrip("\r")
            if "\r" in piece:
                piece = piece.rsplit("\r", 1)[-1]
            if not piece.strip() or _PROGRESS_BAR.match(piece):
                continue
            yield piece


def run_logged_command(cmd, *, check: bool = False, verbose: bool = False, keep=None) -> int:
    """Run cmd and copy its output onto the kmer-ord console.

    The command should include ``conda run --no-capture-output`` so conda
    does not hold the child's output until it exits. PYTHONUNBUFFERED makes
    Python children flush each print through the pipe.

    ``verbose`` prints every complete line. Otherwise ``keep(line)`` decides,
    and ``keep=None`` keeps every complete line. Progress-bar redraws are
    always dropped. Returns the exit code. ``check=True`` raises
    ``CalledProcessError`` after the stream has been printed.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    stream = proc.stdout
    if stream is None:
        raise RuntimeError("logged command was started without a stdout pipe")
    try:
        for line in _iter_log_lines(stream):
            if verbose or keep is None or keep(line):
                console.print(line, markup=False, highlight=False)
    finally:
        if hasattr(stream, "read"):
            stream.read()
        code = proc.wait()
    if check and code != 0:
        raise subprocess.CalledProcessError(code, cmd)
    return code

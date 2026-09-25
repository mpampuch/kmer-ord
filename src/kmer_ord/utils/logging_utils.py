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

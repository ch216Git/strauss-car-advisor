"""Load KEY=VALUE lines from .env files into the environment; Python does not do this by itself.

Variables already set in the environment win; values are never printed, logged or put in errors.
Entry points (scripts, later the web app) call load_env_files() once at start-up.
"""

import os
import re
from collections.abc import MutableMapping, Sequence
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent
# backend/.env first, then the repository root's .env.
DEFAULT_ENV_FILES = (BACKEND_DIR / ".env", BACKEND_DIR.parent / ".env")

_LINE = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)")


def load_env_files(
    paths: Sequence[Path] = DEFAULT_ENV_FILES, environ: MutableMapping[str, str] | None = None
) -> tuple[str, ...]:
    """Set variables from the files that exist and are not already set; return their names only."""
    environ = os.environ if environ is None else environ
    loaded: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        for name, value in _parse(path):
            if name not in environ:
                environ[name] = value
                loaded.append(name)
    return tuple(loaded)


def _parse(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _LINE.fullmatch(stripped)
        if not match:
            # The line itself is not quoted: it may hold a secret.
            raise ValueError(f"{path}: line {number}: expected NAME=value")
        pairs.append((match.group(1), _value(match.group(2))))
    return pairs


def _value(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return raw.split(" #", 1)[0].strip()  # unquoted: drop a trailing " # comment"

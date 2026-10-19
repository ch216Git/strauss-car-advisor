"""Reading and checking JSON config files, with errors that name the file and the exact key path."""

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

from app.errors import ConfigError

TOP_LEVEL = "<top level>"  # key path for errors about the file as a whole

T = TypeVar("T")


def load_json_config(path: Path, parse: Callable[["Section"], T]) -> T:
    """Read a JSON object from path and parse it; any ConfigError raised carries the file name."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(TOP_LEVEL, "an existing config file", "no such file", file=path) from None
    except UnicodeDecodeError as e:
        raise ConfigError(TOP_LEVEL, "UTF-8 text", f"undecodable byte at position {e.start}", file=path) from None
    except json.JSONDecodeError as e:
        raise ConfigError(TOP_LEVEL, "valid JSON", f"{e.msg} at line {e.lineno}, column {e.colno}", file=path) from None

    try:
        return parse(Section(raw, ""))
    except ConfigError as e:
        e.file = path
        raise


def parse_str(value: Any, where: str) -> str:
    return check_type(value, where, str)


def parse_positive_int(value: Any, where: str) -> int:
    number = check_type(value, where, int)
    if number <= 0:
        raise ConfigError(where, "an integer > 0", shown(value))
    return number


def parse_rate(value: Any, where: str) -> float:
    number = check_type(value, where, float)
    if not 0 <= number <= 1:
        raise ConfigError(where, "rate 0..1", shown(value))
    return number


def shown(value: Any, limit: int = 80) -> str:
    """The value as it appears in JSON, shortened so a whole section doesn't flood the message."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


_TYPE_NAMES = {
    int: "an integer",
    float: "a number",
    str: "a string",
    bool: "true or false",
    list: "a list",
    dict: "an object",
}


def check_type(value: Any, where: str, kind: type) -> Any:
    """Return value if it is of the expected JSON type (ints are accepted as numbers)."""
    is_bool = isinstance(value, bool)
    if kind is float and isinstance(value, int) and not is_bool:
        return float(value)
    if not isinstance(value, kind) or (is_bool and kind is not bool):
        raise ConfigError(where, _TYPE_NAMES[kind], shown(value))
    return value


class Section:
    """A JSON object plus its dotted path, so errors can name the exact key."""

    def __init__(self, data: Any, path: str) -> None:
        self._data: dict[str, Any] = check_type(data, path or TOP_LEVEL, dict)
        self._path = path

    def parse(self, key: str, parse: Callable[[Any, str], T]) -> T:
        where = self._where(key)
        if key not in self._data:
            raise ConfigError(where, "a required key", "nothing (key is missing)")
        return parse(self._data[key], where)

    def get(self, key: str, kind: type) -> Any:
        return self.parse(key, lambda value, where: check_type(value, where, kind))

    def section(self, key: str) -> "Section":
        return Section(self.get(key, dict), self._where(key))

    def list_of(self, key: str, parse: Callable[[Any, str], T], *, non_empty: bool = False) -> tuple[T, ...]:
        where = self._where(key)
        values = self.get(key, list)
        if non_empty and not values:
            raise ConfigError(where, "a non-empty list", "[]")
        return tuple(parse(value, f"{where}[{i}]") for i, value in enumerate(values))

    def mapping_of(self, key: str, parse: Callable[[Any, str], T]) -> Mapping[str, T]:
        section = self.section(key)
        return MappingProxyType({name: section.parse(name, parse) for name in section._data})

    def _where(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

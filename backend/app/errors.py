from pathlib import Path


class VehicleNotFoundError(LookupError):
    """No record in the use-value dataset matches the requested key."""


class AmbiguousVehicleError(LookupError):
    """More than one record matches the requested key."""


class RulesConfigError(ValueError):
    """The rules file is missing, unreadable, or a value in it is invalid.

    Message: "<file>: <key path>: expected <expected>, found <found>".
    """

    def __init__(self, path: str, expected: str, found: str, file: Path | None = None) -> None:
        super().__init__(path, expected, found)
        self.file = file  # filled in by the loader, which is the only one that knows it
        self.path = path
        self.expected = expected
        self.found = found

    def __str__(self) -> str:
        return f"{self.file}: {self.path}: expected {self.expected}, found {self.found}"

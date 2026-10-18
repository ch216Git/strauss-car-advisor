class VehicleNotFoundError(LookupError):
    """No record in the use-value dataset matches the requested key."""


class AmbiguousVehicleError(LookupError):
    """More than one record matches the requested key."""


class RulesConfigError(ValueError):
    """The rules file is missing, unreadable, or does not have the expected shape."""

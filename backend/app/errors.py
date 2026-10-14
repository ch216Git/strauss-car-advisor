class VehicleNotFoundError(LookupError):
    """No record in the use-value dataset matches the requested key."""


class AmbiguousVehicleError(LookupError):
    """More than one record matches the requested key."""

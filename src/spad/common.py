from pint import Quantity, UnitRegistry

units = UnitRegistry()


def default_on_none(obj, default):
    """Return obj or default if obj is None"""
    return default if obj is None else obj


class SPADError(Exception):
    pass


def to_travel_time(distance: Quantity, speed: Quantity) -> Quantity:
    return distance / speed

def default_on_none(obj, default):
    """Return obj or default if obj is None"""
    return default if obj is None else obj


class SPADError(Exception):
    pass

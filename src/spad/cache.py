from struct import Struct
from typing import Tuple
import plyvel

KEY_SCHEMA_KEY = b'__schema_key__'
VALUE_SCHEMA_KEY = b'__schema_value__'


class BytesConverter:
    def __init__(self, fmt: str) -> None:
        self.struct = Struct(fmt)

    def encode(self, *values) -> bytes:
        return self.struct.pack(*values)

    def decode(self, value: bytes) -> Tuple:
        return self.struct.unpack(value)


class Raise:
    pass


class LevelDBWriter:
    def __init__(self, db: plyvel.DB, key_fmt: str = None, val_fmt: str = None) -> None:
        self.db = db
        self.key = BytesConverter(key_fmt) if key_fmt else None
        self.value = BytesConverter(val_fmt) if val_fmt else None

    def put(self, key: Tuple, value: Tuple, batch=None):
        writer = self.db if batch is None else batch
        writer.put(
            self.key.encode(*key),
            self.value.encode(*value),
        )

    def get(self, key: Tuple, default=KeyError):
        res = self.db.get(self.key.encode(*key))
        if res is None:
            if isinstance(default, Exception):
                raise default(key)
            else:
                return default
        else:
            return self.value.decode(res)
import pytest
from spad import cache
import struct
import plyvel
import uuid


def test_store_retrieve_ints(tmp_path):
    db = plyvel.DB(str(tmp_path / str(uuid.uuid4())), create_if_missing=True)
    key = b"key"
    converter = cache.BytesConverter("i")
    value = 4
    db.put(key, converter.encode(value))
    assert converter.decode(db.get(key)) == (value,)


def test_store_retrieve_multiple(tmp_path):
    db = plyvel.DB(str(tmp_path / str(uuid.uuid4())), create_if_missing=True)
    key = b"key"
    converter = cache.BytesConverter("id")
    value = (4, 3.5)
    db.put(key, converter.encode(*value))
    assert converter.decode(db.get(key)) == value


def test_cache(tmp_path):
    db = plyvel.DB(str(tmp_path / str(uuid.uuid4())), create_if_missing=True)
    rw = cache.LevelDBWriter(db, "ii", "dd")
    rw.put((1, 2), (3.0, 5.0))
    assert rw.get((1, 2)) == (3.0, 5.0)
    with pytest.raises(KeyError):
        rw.get((2, 3))
    assert rw.get((2, 3), None) is None

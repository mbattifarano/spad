from spad import cache
import struct
import plyvel
import uuid


def test_store_retrieve_ints(tmp_path):
    db = plyvel.DB((tmp_path / str(uuid.uuid4())), create_if_missing=True)
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

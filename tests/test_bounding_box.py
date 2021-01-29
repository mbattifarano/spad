from spad.bounding_box import BoundingBox


def test_bounding_box():
    bbox = BoundingBox(
        min_lat=0,
        min_lon=1,
        max_lat=2,
        max_lon=3
    )
    assert bbox.north == 2
    assert bbox.south == 0
    assert bbox.east == 3
    assert bbox.west == 1

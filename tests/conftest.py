import os
import gzip
from pytest import fixture


@fixture(scope="session")
def fixtures_directory():
    return os.path.join("tests", "fixtures")


@fixture(scope="session")
def mock_gps_data(fixtures_directory):
    return os.path.join(
        fixtures_directory,
        "mock_gps.csv"
    )

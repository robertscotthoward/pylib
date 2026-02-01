import time
from lib.tools import *


def test_to_date():
    d = to_datetime("2025-03-28")
    assert str(d) == "2025-03-28 00:00:00"
    
    d = to_datetime("2025-03-28T18:26:06.948-04:00")
    assert str(d) == "2025-03-28 18:26:06.948000-04:00"

    d = to_datetime_utc("2025-03-28T18:26:06.948-04:00")
    assert str(d) == "2025-03-28 22:26:06.948000+00:00"

    d = to_datetime("2025-03-28T18:26:06.948Z")
    assert str(d) == "2025-03-28 18:26:06.948000+00:00"

    d = to_date("/Date(1234567890)/")
    assert str(d) == "2009-02-13"

    d = to_date("2009-02-13T00:00:00")
    assert str(d) == "2009-02-13"

    d = to_date("2009-02-13")
    assert str(d) == "2009-02-13"

    d = to_date("03/28/2025")
    assert str(d) == "2025-03-28 00:00:00"


def test_getNewTemporaryFilePath():
    fn = getNewTemporaryFilePath('zycloan')
    writeText(fn, 'test')
    assert readText(fn) == 'test'
    os.remove(fn)


def test_spy():
    with Spy("Spy Test") as spy:
        time.sleep(1)


def test_to_seconds():
    assert to_seconds('1s') == 1
    assert to_seconds('1m') == 60
    assert to_seconds('1h') == 3600
    assert to_seconds('1d') == 86400
    assert to_seconds('2d3h4m5s') == 183845
    assert to_seconds('2d5s') == 172805
    assert to_seconds(1) == 1
    assert to_seconds(1.0) == 1.0
    assert to_seconds('1') == 1
    assert to_seconds('1.0') == 1.0
    assert to_seconds('1.0s') == 1.0
    assert to_seconds('1.0m') == 60.0
    assert to_seconds('1.0h') == 3600.0
    assert to_seconds('1.5h') == 5400.0
    assert to_seconds('1.0d') == 86400.0
    assert to_seconds('1.0w') == 604800.0
    assert to_seconds('1.0mo') == 2629746.0
    assert to_seconds('1.0y') == 86400.0
    assert to_seconds('1.0s1m') == 61.0
    assert to_seconds('1.0m1h') == 3660.0
        



if __name__ == "__main__":
    test_to_seconds()
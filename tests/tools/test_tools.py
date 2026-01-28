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
        
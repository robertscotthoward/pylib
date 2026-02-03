import pytest
import lib.ai.raptor

def test_raptor():
    lib.ai.raptor.test_raptor()

# Skip all tests in this file
pytestmark = pytest.mark.skip(reason="Skip all tests in this file")

if __name__ == "__main__":
    test_raptor()
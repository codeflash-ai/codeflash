import pytest
import time


@pytest.fixture(autouse=True)
def fixture1(request):
    start_time = time.time()
    time.sleep(0.1)
    yield
    print(f"Took {time.time() - start_time} seconds")


@pytest.fixture(autouse=True)
def fixture2(request):  # We don't need this fixture during testing
    print("not doing anything")
    yield
    print("did nothing")

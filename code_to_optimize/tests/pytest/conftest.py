import pytest
import time

# @pytest.fixture(autouse=True)
# def fixture(request):
#     if request.node.get_closest_marker("no_autouse"):
#         # Skip the fixture logic
#         yield
#     else:
#         start_time = time.time()
#         time.sleep(0.1)
#         yield
#         print(f"Took {time.time() - start_time} seconds")


@pytest.fixture(autouse=True)
def fixture1(request):  # We don't need this fixture during testing
    start_time = time.time()
    time.sleep(0.1)
    yield
    print(f"Took {time.time() - start_time} seconds")


@pytest.fixture(autouse=True)
def fixture2(request):  # We need it
    yield

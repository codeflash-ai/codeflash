import datetime

import pytest
from codeflash.verification.comparator import comparator
from codeflash.verification.equivalence import compare_results
from codeflash.verification.test_results import (
    TestResults,
    FunctionTestInvocation,
    InvocationId,
    TestType,
)


def test_basic_python_objects():
    a = 5
    b = 5
    c = 6
    d = None
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    a = 5.0
    b = 5.0
    c = 6.0
    d = None
    e = None
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)
    assert not comparator(d, a)
    assert comparator(d, e)

    a = "Hello"
    b = "Hello"
    c = "World"
    assert comparator(a, b)
    assert not comparator(a, c)

    a = [1, 2, 3]
    b = [1, 2, 3]
    c = [1, 2, 4]
    assert comparator(a, b)
    assert not comparator(a, c)

    a = {"a": 1, "b": 2}
    b = {"a": 1, "b": 2}
    c = {"a": 1, "b": 3}
    d = {"c": 1, "b": 2}
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)

    a = (1, 2, "str")
    b = (1, 2, "str")
    c = (1, 2, "str2")
    d = [1, 2, "str"]
    assert comparator(a, b)
    assert not comparator(a, c)
    assert not comparator(a, d)


def test_standard_python_library_objects():
    a = datetime.datetime(2020, 2, 2, 2, 2, 2)
    b = datetime.datetime(2020, 2, 2, 2, 2, 2)
    c = datetime.datetime(2020, 2, 2, 2, 2, 3)
    assert comparator(a, b)
    assert not comparator(a, c)

    a = datetime.date(2020, 2, 2)
    b = datetime.date(2020, 2, 2)
    c = datetime.date(2020, 2, 3)
    assert comparator(a, b)
    assert not comparator(a, c)

    a = datetime.timedelta(days=1)
    b = datetime.timedelta(days=1)
    c = datetime.timedelta(days=2)
    assert comparator(a, b)
    assert not comparator(a, c)


def test_numpy():
    try:
        import numpy as np
    except ImportError:
        pytest.skip()
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    c = np.array([1, 2, 4])
    assert comparator(a, b)
    assert not comparator(a, c)

    d = np.array([[1, 2], [3, 4]])
    e = np.array([[1, 2], [3, 4]])
    f = np.array([[1, 2], [3, 5]])
    assert comparator(d, e)
    assert not comparator(d, f)
    assert not comparator(a, d)

    g = np.array([1.0, 2.0, 3.0])
    assert not comparator(a, g)

    h = np.float32(1.0)
    i = np.float32(1.0)
    assert comparator(h, i)

    j = np.float64(1.0)
    k = np.float64(1.0)
    assert not comparator(h, j)
    print(comparator(j, k))
    assert comparator(j, k)

    l = np.int32(1)
    m = np.int32(1)
    assert comparator(l, m)
    assert not comparator(l, h)
    assert not comparator(l, j)

    n = np.int64(1)
    o = np.int64(1)
    assert not comparator(n, l)
    assert comparator(n, o)

    p = np.uint32(1)
    q = np.uint32(1)
    assert comparator(p, q)
    assert not comparator(p, l)

    r = np.uint64(1)
    s = np.uint64(1)
    assert not comparator(r, p)
    assert comparator(r, s)

    t = np.bool_(True)
    u = np.bool_(True)
    assert comparator(t, u)
    assert not comparator(t, r)

    v = np.complex64(1.0 + 1.0j)
    w = np.complex64(1.0 + 1.0j)
    assert comparator(v, w)
    assert not comparator(v, t)

    x = np.complex128(1.0 + 1.0j)
    y = np.complex128(1.0 + 1.0j)
    assert not comparator(x, v)
    assert comparator(x, y)

    # Create numpy array with mixed type object
    z = np.array([1, 2, "str"], dtype=np.object_)
    aa = np.array([1, 2, "str"], dtype=np.object_)
    ab = np.array([1, 2, "str2"], dtype=np.object_)
    assert comparator(z, aa)
    assert not comparator(z, ab)

    ac = np.array([1, 2, "str2"])
    ad = np.array([1, 2, "str2"])
    assert comparator(ac, ad)


def test_custom_object():
    class TestClass:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return self.value == other.value

    a = TestClass(5)
    b = TestClass(5)
    c = TestClass(6)
    assert comparator(a, b)
    assert not comparator(a, c)

    class TestClass2:
        def __init__(self, value):
            self.value = value

    a = TestClass(5)
    b = TestClass2(5)
    c = TestClass2(5)
    assert not comparator(a, b)
    assert comparator(
        b, c
    )  # This is a fallback to True right now since we don't know how to compare them. This can be improved later

    class TestClass3(TestClass):
        def print(self):
            print(self.value)

    a = TestClass2(5)
    b = TestClass3(5)
    c = TestClass3(5)
    assert not comparator(a, b)
    assert comparator(b, c)



def test_compare_results_fn():
    original_results = TestResults(
        [
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test_module_path",
                    test_class_name="test_class_name",
                    test_function_name="test_function_name",
                    function_getting_tested="function_getting_tested",
                    iteration_id="0",
                ),
                file_name="file_name",
                did_pass=True,
                runtime=5,
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=5,
            )
        ]
    )

    new_results_1 = TestResults(
        [
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test_module_path",
                    test_class_name="test_class_name",
                    test_function_name="test_function_name",
                    function_getting_tested="function_getting_tested",
                    iteration_id="0",
                ),
                file_name="file_name",
                did_pass=True,
                runtime=10,
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=5,
            )
        ]
    )

    assert compare_results(original_results, new_results_1)

    new_results_2 = TestResults(
        [
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test_module_path",
                    test_class_name="test_class_name",
                    test_function_name="test_function_name",
                    function_getting_tested="function_getting_tested",
                    iteration_id="0",
                ),
                file_name="file_name",
                did_pass=True,
                runtime=10,
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=[5],
            )
        ]
    )

    assert not compare_results(original_results, new_results_2)

    new_results_3 = TestResults(
        [
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test_module_path",
                    test_class_name="test_class_name",
                    test_function_name="test_function_name",
                    function_getting_tested="function_getting_tested",
                    iteration_id="0",
                ),
                file_name="file_name",
                did_pass=True,
                runtime=10,
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=5,
            ),
            FunctionTestInvocation(
                id=InvocationId(
                    test_module_path="test_module_path",
                    test_class_name="test_class_name",
                    test_function_name="test_function_name",
                    function_getting_tested="function_getting_tested",
                    iteration_id="2",
                ),
                file_name="file_name",
                did_pass=True,
                runtime=10,
                test_framework="unittest",
                test_type=TestType.EXISTING_UNIT_TEST,
                return_value=5,
            ),
        ]
    )

    assert not compare_results(original_results, new_results_3)

    assert not compare_results(TestResults(), TestResults())
